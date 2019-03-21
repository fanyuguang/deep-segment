#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import tensorflow.contrib.crf as crf
from config import FLAGS
from utils.tensorflow_utils import TensorflowUtils


class SequenceLabelingModel(object):

    def __init__(self):
        self.raw_data_path = FLAGS.raw_data_path
        self.vocab_path = FLAGS.vocab_path

        self.use_stored_embedding = FLAGS.use_stored_embedding
        self.use_lstm = FLAGS.use_lstm
        self.use_dynamic_rnn = FLAGS.use_dynamic_rnn
        self.use_bidirectional_rnn = FLAGS.use_bidirectional_rnn

        self.vocab_size = FLAGS.vocab_size
        self.batch_size = FLAGS.batch_size
        self.num_steps = FLAGS.num_steps
        self.num_layers = FLAGS.num_layers
        self.embedding_size = FLAGS.embedding_size
        self.hidden_size = FLAGS.hidden_size
        self.keep_prob = FLAGS.keep_prob

        self.tensorflow_utils = TensorflowUtils()


    def inference(self, inputs, inputs_sequence_length, num_classes, is_training):
        """
        Bilstm + crf model
        :param inputs:
        :param inputs_sequence_length:
        :param num_classes:
        :param is_training:
        :return:
        """
        with tf.device('/cpu:0'):
            if self.use_stored_embedding:
                embedding = self.tensorflow_utils.load_embedding(os.path.join(self.raw_data_path, 'char_embedding.txt'),
                                                                 os.path.join(self.vocab_path, 'words_vocab.txt'))
            else:
                embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size],
                                            initializer=tf.random_uniform_initializer(), dtype=tf.float32)
            inputs_embedding = tf.nn.embedding_lookup(embedding, inputs)
        if is_training and self.keep_prob < 1:
            inputs_embedding = tf.nn.dropout(inputs_embedding, self.keep_prob)

        rnn_cell_collection = []
        bi_flag = 2 if self.use_bidirectional_rnn else 1
        for _ in range(bi_flag):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            if self.use_lstm:
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, initializer=initializer, forget_bias=1.0)
            else:
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_size)
            if is_training and self.keep_prob < 1.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(self.num_layers)])
            rnn_cell_collection.append(multi_cell)

        if self.use_dynamic_rnn:
            if self.use_bidirectional_rnn:
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_collection[0], rnn_cell_collection[1], inputs_embedding,
                                                             dtype=tf.float32, sequence_length=inputs_sequence_length)
                outputs = tf.concat(outputs, axis=2)
            else:
                outputs, _ = tf.nn.dynamic_rnn(rnn_cell_collection[0], inputs_embedding, dtype=tf.float32,
                                               sequence_length=inputs_sequence_length)
        else:
            inputs_embedding = tf.unstack(inputs_embedding, axis=1)
            if self.use_bidirectional_rnn:
                outputs, _, _ = tf.nn.static_bidirectional_rnn(rnn_cell_collection[0], rnn_cell_collection[1], inputs_embedding,
                                                               dtype=tf.float32, sequence_length=inputs_sequence_length)
            else:
                outputs, _ = tf.nn.static_rnn(rnn_cell_collection[0], inputs_embedding, dtype=tf.float32,
                                              sequence_length=inputs_sequence_length)
            outputs = tf.stack(outputs, axis=1)
        outputs = tf.reshape(outputs, shape=[-1, bi_flag * self.hidden_size])
        weights = tf.get_variable('weights', [bi_flag * self.hidden_size, num_classes], dtype=tf.float32)
        biases = tf.get_variable('biases', [num_classes], dtype=tf.float32)
        logits = tf.matmul(outputs, weights) + biases
        return logits


    def loss(self, logits, labels):
        """
        Loss of cross entropy between logits and labels
        :param logits:
        :param labels:
        :return:
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        return loss


    def accuracy(self, logits, labels):
        """
        Computer the accuracy of rnn model
        :param logits:
        :param labels:
        :return:
        """
        props = tf.nn.softmax(logits)
        prediction_labels = tf.argmax(props, 1)
        correct_prediction = tf.equal(prediction_labels, labels)
        accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy_value


    def slice_seq(self, logits, labels, sequence_lengths):
        """
        Slice sequence, used by accuracy method
        :param logits:
        :param labels:
        :param words_len:
        :return:
        """
        labels = tf.reshape(labels, shape=[-1])
        slice_indices = tf.constant([], dtype=tf.int64)
        for index in range(self.batch_size):
            sub_slice_indices = tf.range(sequence_lengths[index])
            sub_slice_indices = tf.add(tf.constant(index * self.num_steps, dtype=tf.int64), sub_slice_indices)
            slice_indices = tf.concat([slice_indices, sub_slice_indices], axis=0)
        slice_logits = tf.gather(logits, slice_indices)
        slice_labels = tf.gather(labels, slice_indices)
        return slice_logits, slice_labels


    def crf_loss(self, logits, labels, sequence_lengths, num_classes):
        """
        Loss of crf
        :param logits:
        :param labels:
        :param sequence_lengths:
        :param num_classes:
        :return:
        """
        logits = tf.reshape(logits, shape=[self.batch_size, self.num_steps, num_classes])
        labels = tf.cast(labels, tf.int32)
        log_likelihood, transition_params = crf.crf_log_likelihood(logits, labels, sequence_lengths)
        loss = tf.reduce_mean(-log_likelihood, name='loss')
        return loss, transition_params


    def crf_accuracy(self, logits, labels, sequence_length, transition_params, num_classes):
        """
        Computer the accuracy of rnn + crf model
        :param logits:
        :param labels:
        :param sequence_length:
        :param transition_params:
        :param num_classes:
        :return:
        """
        logits = tf.reshape(logits, shape=[self.batch_size, self.num_steps, num_classes])
        sequence_length = tf.to_int32(sequence_length)
        predict_indices, _ = crf.crf_decode(logits, transition_params, sequence_length)
        predict_indices = tf.to_int64(predict_indices, name='predict_label_indices')
        correct_prediction = tf.equal(predict_indices, labels)
        accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy_value