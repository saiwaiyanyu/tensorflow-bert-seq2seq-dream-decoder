# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/21 8:24 AM
# @Author: wuchenglong




import tensorflow as tf
import os,random,json
import argparse
from seq2seq import Seq2Seq
from utils import build_vocab,pad_sentence_batch,str_idx,len_check,read_vocab
from args_helper import args
from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)

def train():
    lines = [line.strip() for line in open("data/data.csv", "r").readlines()]
    lines = [(json.loads(line)["dream"], json.loads(line)["decode"]) for line in lines]
    inputs = [" ".join(list(q)) for q, a in lines]
    outputs = [" ".join(list(a)) for q, a in lines]
    all_info = ' '.join(inputs + outputs).split()
    if os.path.exists(args.vocab_file):
        dictionary_input, rev_dictionary_input = read_vocab(args.vocab_file)
    else:
        dictionary_input, rev_dictionary_input = build_vocab(all_info, args.vocab_file)

    dictionary_output, rev_dictionary_output = dictionary_input, rev_dictionary_input

    min_line_length = 2
    max_line_length = 100

    data_filter = [(q, a) for q, a in zip(inputs, outputs) if
                              len_check(q, min_line_length, max_line_length) and len_check(a, min_line_length,
                                                                                           max_line_length)]
    random.shuffle(data_filter)
    inputs = [q for q, a in data_filter]
    outputs = [a + ' EOS' for q, a in data_filter]

    tf.logging.info("sample size: %s", len(inputs))
    inputs_dev = inputs[0:100]
    outputs_dev = outputs[0:100]
    inputs_train = inputs[100:  ]
    outputs_train = outputs[100: ]



    inputs_train = str_idx(inputs_train, dictionary_input, dictionary_input['UNK'])
    print(inputs_train[:2])
    outputs_train = str_idx(outputs_train, dictionary_output, dictionary_output['UNK'])
    print(outputs_train[:2])
    inputs_dev = str_idx(inputs_dev, dictionary_input, dictionary_input['UNK'])
    outputs_dev = str_idx(outputs_dev, dictionary_output, dictionary_output['UNK'])

    model = Seq2Seq(args.size_layer,
                    args.num_layers,
                    args.embedded_size,
                    len(dictionary_input),
                    len(dictionary_output),
                    args.learning_rate,
                    dictionary_input
                    )

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
                for k in range(0, len(inputs_train), args.batch_size):
                    batch_num = batch_num + 1
                    index = min(k + args.batch_size, len(inputs_train))
                    batch_x, seq_x = pad_sentence_batch(inputs_train[k: index], dictionary_input["PAD"])
                    batch_y, seq_y = pad_sentence_batch(outputs_train[k: index], dictionary_input["PAD"])
                    predicted, accuracy, loss, _, global_step = sess.run(fetches = [model.predicting_ids,
                                                                       model.accuracy,
                                                                       model.cost,
                                                                       model.optimizer,
                                                                       model.global_step
                                                             ],
                                                            feed_dict={model.X: batch_x,
                                                                       model.Y: batch_y})
                    total_loss += loss
                    total_accuracy += accuracy


                    if global_step % 100 == 0:
                        print('%s epoch: %d, global_step: %d, loss: %f, accuracy: %f' % (datetime.now().strftime( '%Y-%m-%d %H:%M:%S' ),epoch_index + 1, global_step, loss, accuracy))
                        saver.save(sess, os.path.join(args.checkpoint_dir, "seq2seq.ckpt"), global_step=global_step)


                        print("+" * 20)
                        for i in range(4):
                            print('row %d' % (i + 1))
                            print('dream:',
                                  ''.join([rev_dictionary_input[n] for n in batch_x[i] if n not in [0,1,2,3]]))
                            print('real   meaning:',
                                  ''.join([rev_dictionary_output[n] for n in batch_y[i] if n not in [0,1,2,3]]))
                            print('dream decoding:',
                                  ''.join([rev_dictionary_output[n] for n in predicted[i] if n not in [0,1,2,3] ]),
                                  '')


                        index = list(range(len((inputs_dev))))
                        random.shuffle(index)
                        batch_x, _ = pad_sentence_batch([inputs_dev[i] for i in index ][:args.batch_size], dictionary_input["PAD"])
                        batch_y, _ = pad_sentence_batch([outputs_dev[i] for i in index ][:args.batch_size], dictionary_input["PAD"])
                        predicted = sess.run(model.predicting_ids, feed_dict={model.X: batch_x})
                        print("-" * 20)
                        for i in range(4):
                            print('row %d' % (i + 1))
                            # print(batch_x[i])
                            # print(predicted[i])
                            print('dream:',
                                  ''.join([rev_dictionary_input[n] for n in batch_x[i] if n not in [0,1,2,3]]))
                            print('real   meaning:',
                                  ''.join([rev_dictionary_output[n] for n in batch_y[i] if n not in [0,1,2,3]]))
                            print('dream decoding:',
                                  ''.join([rev_dictionary_output[n] for n in predicted[i] if n not in [0,1,2,3]]), '')

                total_loss /= batch_num
                total_accuracy /= batch_num
                print('***%s epoch: %d, global_step: %d, avg loss: %f, avg accuracy: %f' % (datetime.now().strftime( '%Y-%m-%d %H:%M:%S' ), epoch_index + 1, global_step, total_loss, total_accuracy))



def predict():

    dictionary_input, rev_dictionary_input = read_vocab(args.vocab_file)
    dictionary_output, rev_dictionary_output = dictionary_input, rev_dictionary_input

    model = Seq2Seq(args.size_layer,
                    args.num_layers,
                    args.embedded_size,
                    len(dictionary_input),
                    len(dictionary_output),
                    args.learning_rate,
                    dictionary_input)

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
                input_test = [" ".join(list(text))]
                input_test = str_idx(input_test, dictionary_input,dictionary_input['UNK'])
                batch_x, _ = pad_sentence_batch(input_test, dictionary_input["PAD"])
                predicted2 = sess.run(model.predicting_ids, feed_dict={model.X: batch_x})
                for i in range(len(batch_x)):
                    print('dream:', ''.join([rev_dictionary_input[n] for n in batch_x[i] if n not in [0,1,2,3]]) )
                    print('dream decoding:',
                          ''.join([rev_dictionary_output[n] for n in predicted2[i] if n not in [0,1,2,3]]), '\n')
                print("*" * 20)

if __name__ == "__main__":
    if args.task == "train":
        train()
    if args.task == "predict":
        predict()