#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import shutil
import tensorflow as tf
import tensorflow.contrib.crf as crf

from tensorflow.contrib import lookup
from config import FLAGS
from utils.data_utils import DataUtils
from utils.tensorflow_utils import TensorflowUtils
from model import SequenceLabelingModel


class Predict(object):

    def __init__(self):
        self.vocab_path = FLAGS.vocab_path
        self.checkpoint_path = FLAGS.checkpoint_path
        self.freeze_graph_path = FLAGS.freeze_graph_path
        self.saved_model_path = FLAGS.saved_model_path

        self.use_crf = FLAGS.use_crf
        self.num_steps = FLAGS.num_steps

        self.default_label = FLAGS.default_label
        self.default_score = FLAGS.default_predict_score

        self.data_utils = DataUtils()
        self.tensorflow_utils = TensorflowUtils()
        self.num_classes = self.data_utils.load_num_classes()
        self.sequence_labeling_model = SequenceLabelingModel()
        self.init_predict_graph()


    def init_predict_graph(self):
        """
        init predict model graph
        :return:
        """
        # split 1-D String dense Tensor to words SparseTensor
        self.input_sentences = tf.placeholder(dtype=tf.string, shape=[None], name='input_sentences')
        sparse_words = tf.string_split(self.input_sentences, delimiter=' ')

        # slice SparseTensor
        valid_indices = tf.less(sparse_words.indices, tf.constant([self.num_steps], dtype=tf.int64))
        valid_indices = tf.reshape(tf.split(valid_indices, [1, 1], axis=1)[1], [-1])
        valid_sparse_words = tf.sparse_retain(sparse_words, valid_indices)

        excess_indices = tf.greater_equal(sparse_words.indices, tf.constant([self.num_steps], dtype=tf.int64))
        excess_indices = tf.reshape(tf.split(excess_indices, [1, 1], axis=1)[1], [-1])
        excess_sparse_words = tf.sparse_retain(sparse_words, excess_indices)

        # compute sentences lengths
        int_values = tf.ones(shape=tf.shape(valid_sparse_words.values), dtype=tf.int64)
        int_valid_sparse_words = tf.SparseTensor(indices=valid_sparse_words.indices, values=int_values,
                                                 dense_shape=valid_sparse_words.dense_shape)
        input_sentences_lengths = tf.sparse_reduce_sum(int_valid_sparse_words, axis=1)

        # sparse to dense
        default_padding_word = self.data_utils._START_VOCAB[0]
        words = tf.sparse_to_dense(sparse_indices=valid_sparse_words.indices,
                                   output_shape=[valid_sparse_words.dense_shape[0], self.num_steps],
                                   sparse_values=valid_sparse_words.values,
                                   default_value=default_padding_word)

        # dict words to ids
        with open(os.path.join(self.vocab_path, 'words_vocab.txt'), encoding='utf-8', mode='rt') as data_file:
            words_table_list = [line.strip() for line in data_file if line.strip()]
        words_table_tensor = tf.constant(words_table_list, dtype=tf.string)
        words_table = lookup.index_table_from_tensor(mapping=words_table_tensor, default_value=self.data_utils._START_VOCAB_ID[3])
        # words_table = lookup.index_table_from_file(os.path.join(vocab_path, 'words_vocab.txt'), default_value=3)
        words_ids = words_table.lookup(words)

        # blstm model predict
        with tf.variable_scope('model', reuse=None):
            logits = self.sequence_labeling_model.inference(words_ids, input_sentences_lengths, self.num_classes, is_training=False)

        if self.use_crf:
            logits = tf.reshape(logits, shape=[-1, self.num_steps, self.num_classes])
            transition_params = tf.get_variable("transitions", [self.num_classes, self.num_classes])
            input_sentences_lengths = tf.to_int32(input_sentences_lengths)
            predict_labels_ids, sequence_scores = crf.crf_decode(logits, transition_params, input_sentences_lengths)
            predict_labels_ids = tf.to_int64(predict_labels_ids)
            sequence_scores = tf.reshape(sequence_scores, shape=[-1, 1])
            normalized_sequence_scores = self.tensorflow_utils.score_normalize(sequence_scores)
            predict_scores = tf.matmul(normalized_sequence_scores, tf.ones(shape=[1, self.num_steps], dtype=tf.float32))
        else:
            props = tf.nn.softmax(logits)
            max_prop_values, max_prop_indices = tf.nn.top_k(props, k=1)
            predict_labels_ids = tf.reshape(max_prop_indices, shape=[-1, self.num_steps])
            predict_labels_ids = tf.to_int64(predict_labels_ids)
            predict_scores = tf.reshape(max_prop_values, shape=[-1, self.num_steps])
        predict_scores = tf.as_string(predict_scores, precision=3)

        # dict ids to labels
        with open(os.path.join(self.vocab_path, 'labels_vocab.txt'), encoding='utf-8', mode='rt') as data_file:
            labels_table_list = [line.strip() for line in data_file if line.strip()]
        labels_table_tensor = tf.constant(labels_table_list, dtype=tf.string)
        labels_table = lookup.index_to_string_table_from_tensor(mapping=labels_table_tensor, default_value=self.default_label)
        # labels_table = lookup.index_to_string_table_from_file(os.path.join(vocab_path, 'labels_vocab.txt'), default_value='O')
        predict_labels = labels_table.lookup(predict_labels_ids)

        sparse_predict_labels = self.tensorflow_utils.sparse_concat(predict_labels, valid_sparse_words, excess_sparse_words, self.default_label)
        sparse_predict_scores = self.tensorflow_utils.sparse_concat(predict_scores, valid_sparse_words, excess_sparse_words, '0.0')

        self.format_predict_labels = self.tensorflow_utils.sparse_string_join(sparse_predict_labels, 'predict_labels')
        self.format_predict_scores = self.tensorflow_utils.sparse_string_join(sparse_predict_scores, 'predict_scores')

        saver = tf.train.Saver()
        tables_init_op = tf.tables_initializer()

        self.sess = tf.Session()
        self.sess.run(tables_init_op)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('read model from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found at %s' % self.checkpoint_path)
            return


    def predict(self, words_list):
        """
        Predict labels, the operation of transfer words to ids is processed by tensorflow tensor
        Input words list
        :param words_list:
        :return:
        """
        split_words_list = []
        map_split_indexes = []
        for index in range(len(words_list)):
            temp_words_list = self.data_utils.split_long_sentence(words_list[index], self.num_steps)
            map_split_indexes.append(list(range(len(split_words_list), len(split_words_list) + len(temp_words_list))))
            split_words_list.extend(temp_words_list)

        predict_labels, predict_scores = self.sess.run([self.format_predict_labels, self.format_predict_scores], feed_dict={self.input_sentences: split_words_list})
        predict_labels_str = [predict_label.decode('utf-8') for predict_label in predict_labels]
        predict_scores_str = [predict_score.decode('utf-8') for predict_score in predict_scores]

        merge_predict_labels_str = []
        merge_predict_scores_str = []
        for indexes in map_split_indexes:
            merge_predict_label_str = ' '.join([predict_labels_str[index] for index in indexes])
            merge_predict_labels_str.append(merge_predict_label_str)
            merge_predict_score_str = ' '.join([predict_scores_str[index] for index in indexes])
            merge_predict_scores_str.append(merge_predict_score_str)

        return merge_predict_labels_str, merge_predict_scores_str


    def file_predict(self, data_filename, predict_filename):
        """
        Predict data_filename, save the predict result into predict_filename
        :param data_filename:
        :param predict_filename:
        :return:
        """
        print('Predict file ' + data_filename)
        words_list = []
        labels_list = []
        predict_labels_list = []
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                words, labels = self.data_utils.split(line)
                if words and labels:
                    words_list.append(' '.join(words))
                    labels_list.append(' '.join(labels))
                    predict_labels, _ = self.predict([' '.join(words)])
                    predict_labels_list.append(predict_labels[0])
        with open(predict_filename, encoding='utf-8', mode='wt') as predict_file:
            for (words, labels, predict_labels) in zip(words_list, labels_list, predict_labels_list):
                predict_file.write('Passage: ' + words + '\n')
                predict_file.write('Label: ' + labels + '\n')
                predict_file.write('PredictLabel: ' + predict_labels + '\n\n')


    def single_word_file_predict(self, data_filename, predict_filename):
        """
        Predict data_filename, save the predict result into predict_filename
        The label is split into single word, -B -M -E -S
        :param data_filename:
        :param predict_filename:
        :return:
        """
        print('Predict file ' + data_filename)
        sentence_list = []
        words_list = []
        labels_list = []
        predict_labels_list = []
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                words, labels = self.data_utils.split(line)
                if words and labels:
                    sentence_list.append(''.join(words))
                    words_list.append(' '.join(words))
                    labels_list.append(' '.join(labels))
                    predict_labels, _ = self.predict([' '.join(words)])
                    predict_labels_list.append(predict_labels[0])
        word_predict_label_list = []
        word_category_list = []
        word_predict_category_list = []
        for (words, labels, predict_labels) in zip(words_list, labels_list, predict_labels_list):
            word_list = words.split()
            label_list = labels.split()
            predict_label_list = predict_labels.split()
            word_predict_label = ' '.join([word + '/' + predict_label for (word, predict_label) in zip(word_list, predict_label_list)])
            word_predict_label_list.append(word_predict_label)
            # merge label
            merge_word_list, merge_label_list = self.data_utils.merge_label(word_list, label_list)
            word_category = ' '.join([word + '/' + label for (word, label) in zip(merge_word_list, merge_label_list) if label != self.default_label])
            word_category_list.append(word_category)
            # merge predict label
            merge_predict_word_list, merge_predict_label_list = self.data_utils.merge_label(word_list, predict_label_list)
            word_predict_category = ' '.join([predict_word + '/' + predict_label for (predict_word, predict_label) in
                                              zip(merge_predict_word_list, merge_predict_label_list) if predict_label != 'O'])
            word_predict_category_list.append(word_predict_category)
        with open(predict_filename, encoding='utf-8', mode='wt') as predict_file:
            for (sentence, word_predict_label, word_category, word_predict_category) in \
                    zip(sentence_list, word_predict_label_list, word_category_list, word_predict_category_list):
                predict_file.write('Passage: ' + sentence + '\n')
                predict_file.write('SinglePredict: ' + word_predict_label + '\n')
                predict_file.write('Merge: ' + word_category + '\n')
                predict_file.write('MergePredict: ' + word_predict_category + '\n\n')


    def freeze_graph(self):
        """
        Save graph into .pb file
        :return:
        """
        graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ['init_all_tables', 'predict_labels', 'predict_scores'])
        tf.train.write_graph(graph, self.freeze_graph_path, 'frozen_graph.pb', as_text=False)
        print('Successfully freeze model to %s' % self.freeze_graph_path)


    def saved_model_pb(self):
        """
        Saved model into .ph and variables files, loading it by tensorflow serving,
        :return:
        """
        saved_model_path = os.path.join(self.saved_model_path, '1')
        if os.path.exists(saved_model_path):
            shutil.rmtree(saved_model_path)
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
        input_tensor_info = tf.saved_model.utils.build_tensor_info(self.input_sentences)
        output_labels_tensor_info = tf.saved_model.utils.build_tensor_info(self.format_predict_labels)
        output_scores_tensor_info = tf.saved_model.utils.build_tensor_info(self.format_predict_scores)
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_sentences': input_tensor_info},
            outputs={'predict_labels': output_labels_tensor_info, 'predict_scores': output_scores_tensor_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict_segment': prediction_signature},
            legacy_init_op=legacy_init_op
        )
        builder.save()
        print('Successfully exported model to %s' % saved_model_path)


def main(_):
    predict = Predict()

    sentence = '张伟在6月16号会去一趟丹棱街中国移动营业厅'
    sentence = ''.join(sentence.split())
    words = ' '.join([char for char in sentence])
    predict_labels, predict_scores = predict.predict([words, '你 好'])
    print(predict_labels)
    print(predict_scores)

    predict.freeze_graph()
    predict.saved_model_pb()

    # file predict
    # ner_predict.file_predict(os.path.join(FLAGS.datasets_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'test_predict.txt'))

    # file predict, and the label is split into single word
    # ner_predict.single_word_file_predict(os.path.join(FLAGS.datasets_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'test_predict.txt'))


if __name__ == '__main__':
    tf.app.run()
