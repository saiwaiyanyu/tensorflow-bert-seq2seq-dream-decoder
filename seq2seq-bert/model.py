# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/21 8:24 AM
# @Author: wuchenglong

import tensorflow as tf
import os,random,json
from seq2seq import Seq2Seq
from utils import pad_data,len_check
from args_helper import args
from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)

def train():
    model = Seq2Seq(args.size_layer,
                    args.num_layers,
                    args.learning_rate,
                    args.vocab_file,
                    args.bert_config,
                    args.is_training,
                    )

    lines = [line.strip() for line in open("data/data.csv", "r").readlines()]
    lines = [(json.loads(line)["dream"], json.loads(line)["decode"]) for line in lines]
    inputs = [" ".join(list(q)) for q, a in lines]
    outputs = [" ".join(list(a)) for q, a in lines]

    dictionary_output, rev_dictionary_output = model.tokenizer.vocab,  model.tokenizer.inv_vocab
    dictionary_input, rev_dictionary_input = model.tokenizer.vocab,  model.tokenizer.inv_vocab

    min_line_length = 2
    max_line_length = 100

    data_filter = [(q, a) for q, a in zip(inputs, outputs) if
                              len_check(q, min_line_length, max_line_length) and len_check(a, min_line_length,
                                                                                           max_line_length)]
    random.shuffle(data_filter)
    inputs = [['[SEP]']+  q.split() + ['[CLS]'] for q, a in data_filter]
    outputs = [['[SEP]']+  a.split() + ['[CLS]'] for q, a in data_filter]

    tf.logging.info("sample size: %s", len(inputs))

    data = []
    for i in range(len(inputs)):
        tokens = inputs[i]
        inputs_ids = model.tokenizer.convert_tokens_to_ids(inputs[i])
        segment_ids = [0] * len(inputs_ids)
        input_mask = [1] * len(inputs_ids)
        tag_ids = model.tokenizer.convert_tokens_to_ids(outputs[i])
        data.append([tokens, tag_ids, inputs_ids, segment_ids, input_mask])


    data_dev = data[0:100]
    data_train = data[100:200]

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                tf.logging.info("restore model from patch: %s", ckpt.model_checkpoint_path)  # 加载预训练模型
                saver = tf.train.Saver(max_to_keep=4)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver = tf.train.Saver(max_to_keep=4)
                sess.run(tf.global_variables_initializer())

            global_step = 0
            for epoch_index in range(args.epoch):
                total_loss, total_accuracy = 0, 0
                batch_num = 0
                for k in range(0, len(data_train), args.batch_size):
                    batch_num = batch_num + 1
                    index = min(k + args.batch_size, len(data_train))
                    batch = pad_data(data_train[k: index])
                    tokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)
                    predicted, accuracy, loss, _, global_step = sess.run(fetches = [model.predicting_ids,
                                                                       model.accuracy,
                                                                       model.cost,
                                                                       model.optimizer,
                                                                       model.global_step
                                                             ],
                                                            feed_dict={
                                                                model.input_ids: inputs_ids,
                                                                model.input_mask: input_mask,
                                                                model.segment_ids: segment_ids,
                                                                model.Y: tag_ids,
                                                                model.dropout: 1.0
                                                            })
                    total_loss += loss
                    total_accuracy += accuracy


                    if global_step % 100 == 0:
                        print('%s  epoch: %d, global_step: %d, loss: %f, accuracy: %f' % (datetime.now().strftime( '%Y-%m-%d %H:%M:%S' ), epoch_index + 1, global_step, loss, accuracy))
                        saver.save(sess, os.path.join(args.checkpoint_dir, "seq2seq.ckpt"), global_step=global_step)


                        print("+" * 20)
                        for i in range(4):
                            print('row %d' % (i + 1))
                            print('dream:',
                                  ''.join([rev_dictionary_input[n] for n in inputs_ids[i] if n not in [0, 1, 2, 3]]))
                            print('real   meaning:',
                                  ''.join([rev_dictionary_output[n] for n in tag_ids[i] if n not in [0, 1, 2, 3]]))
                            print('dream decoding:',
                                  ''.join([rev_dictionary_output[n] for n in predicted[i] if n not in [0, 1, 2, 3]]),
                                  '')


                        index = list(range(len((data_dev))))
                        random.shuffle(index)
                        batch =pad_data([data_dev[i] for i in index ][:args.batch_size])
                        tokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)

                        predicted = sess.run(model.predicting_ids, feed_dict={
                            model.input_ids: inputs_ids,
                            model.input_mask: input_mask,
                            model.segment_ids: segment_ids,
                            model.dropout:1.0

                                                                              })
                        print("-" * 20)
                        for i in range(4):
                            print('row %d' % (i + 1))
                            print('dream:',
                                  ''.join([rev_dictionary_input[n] for n in inputs_ids[i] if n not in [0, 1, 2, 3]]))
                            print('real   meaning:',
                                  ''.join([rev_dictionary_output[n] for n in tag_ids[i] if n not in [0, 1, 2, 3]]))
                            print('dream decoding:',
                                  ''.join([rev_dictionary_output[n] for n in predicted[i] if n not in [0, 1, 2, 3]]), '')

                total_loss /= batch_num
                total_accuracy /= batch_num
                print('***%s epoch: %d, global_step: %d, avg loss: %f, avg accuracy: %f' % (datetime.now().strftime( '%Y-%m-%d %H:%M:%S' ),epoch_index + 1, global_step, total_loss, total_accuracy))



def predict():
    model = Seq2Seq(args.size_layer,
                    args.num_layers,
                    args.learning_rate,
                    args.vocab_file,
                    args.bert_config,
                    args.is_training,
                    )
    dictionary_output, rev_dictionary_output = model.tokenizer.vocab,  model.tokenizer.inv_vocab
    dictionary_input, rev_dictionary_input = model.tokenizer.vocab,  model.tokenizer.inv_vocab

    with tf.Session() as sess:
        with tf.device("/cpu:0"):

            ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                tf.logging.info("restore model from patch: %s", ckpt.model_checkpoint_path)  # 加载预训练模型
                saver = tf.train.Saver(max_to_keep=4)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                tf.logging.error("model path wrong !!")
                return

            while True:
                text = input("input your dream: ")
                inputs = ['[SEP]'] + list(text) + ['[CLS]']

                inputs_ids = model.tokenizer.convert_tokens_to_ids(inputs)
                segment_ids = [0] * len(inputs_ids)
                input_mask = [1] * len(inputs_ids)



                predicted2 = sess.run(model.predicting_ids, feed_dict={
                    model.input_ids: [inputs_ids],
                    model.input_mask: [input_mask],
                    model.segment_ids: [segment_ids],
                    model.dropout: 1.0
                })

                print('dream:', ''.join([rev_dictionary_input[n] for n in inputs_ids if n not in [0, 1, 2, 3]]))
                print('dream decoding:',
                          ''.join([rev_dictionary_output[n] for n in predicted2[0] if n not in [0, 1, 2, 3]]), '')
                print("*" * 20)

if __name__ == "__main__":
    if args.task == "train":
        train()
    if args.task == "predict":
        predict()