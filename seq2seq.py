# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/19 3:10 PM
# @Author: wuchenglong


import tensorflow as tf


class Seq2Seq:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate,
                 dictionary_input):

        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        self.dictionary_input = dictionary_input

        self.X = tf.placeholder(tf.int32, [None, None],name = "X")
        self.Y = tf.placeholder(tf.int32, [None, None],name = "Y")
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        self.global_step = tf.Variable(1, name="global_step", trainable=False)

        with tf.variable_scope("encoder"):
            self.encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1),name ="encoder_embedding")
            batch_size = tf.shape(self.X)[0]

            _, encoder_state = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)]),
                inputs=tf.nn.embedding_lookup(self.encoder_embedding, self.X),
                sequence_length=self.X_seq_len,
                dtype=tf.float32)

        with tf.variable_scope("decoder"):
            self.decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1),
                                                 name="decoder_embedding")

            main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], self.dictionary_input["GO"]), main], 1)
            dense = tf.layers.Dense(to_dict_size)
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
                start_tokens=tf.tile(tf.constant([self.dictionary_input["GO"]], dtype=tf.int32), [batch_size]),
                end_token=self.dictionary_input["EOS"])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cells,
                helper=predicting_helper,
                initial_state=encoder_state,
                output_layer=dense)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=True,
                maximum_iterations= 128
                # maximum_iterations=2 * tf.reduce_max(self.X_seq_len)
            )

        self.predicting_ids = predicting_decoder_output.sample_id
        self.masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                     targets=self.Y,
                                                     weights=self.masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost,global_step=self.global_step, name = "optimizer")
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, self.masks, name = "prediction")
        self.mask_label = tf.boolean_mask(self.Y, self.masks,name = "mask_label")
        self.correct_pred = tf.equal(self.prediction, self.mask_label)
        self.correct_index = tf.cast(self.correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

