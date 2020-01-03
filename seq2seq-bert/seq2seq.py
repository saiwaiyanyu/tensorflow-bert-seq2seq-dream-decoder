# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/19 3:10 PM
# @Author: wuchenglong


import tensorflow as tf
from bert import modeling
import tokenization

class Seq2Seq:
    def __init__(self, size_layer, num_layers,learning_rate,
                 vocab_file, bert_config, is_training):

        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,)

        self.is_training = is_training

        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="segment_ids"
        )
        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="dropout"
        )

        self.Y = tf.placeholder(tf.int32, [None, None],name = "Y")
        self.X_seq_len = tf.count_nonzero(self.input_ids, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        self.global_step = tf.Variable(1, name="global_step", trainable=False)
        batch_size = tf.shape(self.input_ids)[0]

        ###################################
        # 不用bert
        # self.encoder_embedding = tf.Variable(tf.random_uniform([len(self.tokenizer.vocab), 768], -1, 1))
        # self.decoder_embedding = tf.Variable(tf.random_uniform([len(self.tokenizer.vocab), 768], -1, 1))
        ###################################

        ###################################
        # 使用 bert，将bert的word_embedding提取出来
        self.bert_config = modeling.BertConfig.from_json_file(bert_config)
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        # self.encoder_embedding = model.get_embedding_table()
        # self.decoder_embedding = model.get_embedding_table()
        with tf.variable_scope("bert",reuse=True):
            with tf.variable_scope("embeddings",reuse=True):
                self.encoder_embedding = tf.get_variable(name="word_embeddings")
                self.decoder_embedding = tf.get_variable(name="word_embeddings")
        ###################################



        self.input_embedding = tf.nn.embedding_lookup(self.encoder_embedding, self.input_ids)
        with tf.variable_scope("encoder"):
            _, encoder_state = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)]),
                inputs=tf.nn.embedding_lookup(self.encoder_embedding, self.input_ids),
                # inputs=model.get_sequence_output(),
                sequence_length=self.X_seq_len,
                dtype=tf.float32)

        with tf.variable_scope("decoder"):

            main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], self.tokenizer.vocab["[unused1]"]), main], 1)
            dense = tf.layers.Dense(len(self.tokenizer.vocab))
            decoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])

            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(self.decoder_embedding, decoder_input),
                sequence_length=self.Y_seq_len,
                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cells,
                helper=training_helper,
                initial_state=encoder_state,
                output_layer=dense)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output

            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.decoder_embedding,
                start_tokens=tf.tile(tf.constant([self.tokenizer.vocab["[unused1]"]], dtype=tf.int32), [batch_size]),
                end_token=self.tokenizer.vocab["[unused2]"])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cells,
                helper=predicting_helper,
                initial_state=encoder_state,
                output_layer=dense)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=True,
                # maximum_iterations= 100,
                maximum_iterations = 3 * tf.reduce_max(self.X_seq_len)
            )

        self.predicting_ids = predicting_decoder_output.sample_id
        self.masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                     targets=self.Y,
                                                     weights=self.masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(self.cost,global_step=self.global_step, name = "optimizer")
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, self.masks, name = "prediction")
        self.mask_label = tf.boolean_mask(self.Y, self.masks,name = "mask_label")
        self.correct_pred = tf.equal(self.prediction, self.mask_label)
        self.correct_index = tf.cast(self.correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

