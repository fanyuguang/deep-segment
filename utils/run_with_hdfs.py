#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from config import FLAGS
from utils.data_utils import DataUtils
from utils.hdfs_utils import HdfsUtils
from predict import Predict
from utils.tensorflow_utils import TensorflowUtils
from train import Train

tf.app.flags.DEFINE_string('hdfs_host', 'hdfs-bizaistca.corp.microsoft.com', 'hdfs host')
tf.app.flags.DEFINE_integer('hdfs_port', 8020, 'hdfs port')
tf.app.flags.DEFINE_string('hdfs_user', 'hadoop', 'hdfs user')

tf.app.flags.DEFINE_string('input_path', '/user/hadoop/data/input/', 'input data path')
tf.app.flags.DEFINE_string('output_path', '/user/hadoop/data/output/', 'output data path')


def main():
    hdfs_client = HdfsUtils(FLAGS.hdfs_host, FLAGS.hdfs_port, FLAGS.hdfs_user)
    hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'data.txt'), os.path.join(FLAGS.raw_data_path, 'data.txt'))
    data_utils = DataUtils()
    # datasets process
    data_utils.prepare_datasets(os.path.join(FLAGS.raw_data_path, 'data_demo.txt'), FLAGS.train_percent,
                                FLAGS.val_percent, FLAGS.datasets_path)
    words_vocab, labels_vocab, _ = data_utils.create_vocabulary(os.path.join(FLAGS.datasets_path, 'train.txt'),
                                                                FLAGS.vocab_path, FLAGS.vocab_size)

    train_word_ids_list, train_label_ids_list = data_utils.file_to_word_ids(
        os.path.join(FLAGS.datasets_path, 'train.txt'), words_vocab, labels_vocab)
    validation_word_ids_list, validation_label_ids_list = data_utils.file_to_word_ids(
        os.path.join(FLAGS.datasets_path, 'validation.txt'), words_vocab, labels_vocab)
    test_word_ids_list, test_label_ids_list = data_utils.file_to_word_ids(os.path.join(FLAGS.datasets_path, 'test.txt'),
                                                                          words_vocab, labels_vocab)

    tensorflow_utils = TensorflowUtils()

    tensorflow_utils.create_record(train_word_ids_list, train_label_ids_list,
                                   os.path.join(FLAGS.tfrecords_path, 'train.tfrecords'))
    tensorflow_utils.create_record(validation_word_ids_list, validation_label_ids_list,
                                   os.path.join(FLAGS.tfrecords_path, 'validate.tfrecords'))
    tensorflow_utils.create_record(test_word_ids_list, test_label_ids_list,
                                   os.path.join(FLAGS.tfrecords_path, 'test.tfrecords'))

    segment_train = Train()
    segment_train.train()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        predict = Predict()

        sentence = '张伟在6月16号会去一趟丹棱街中国移动营业厅'
        words = ' '.join([char for char in sentence])
        predict_labels, predict_scores = predict.predict([words])
        print(predict_labels)
        print(predict_scores)

        predict.saved_model_pb()

    hdfs_upload(FLAGS.tensorboard_path, os.path.join(FLAGS.output_path, 'tensorboard'))
    hdfs_upload(FLAGS.saved_model_path, os.path.join(FLAGS.output_path, 'saved-model-data'))


if __name__ == '__main__':
    main()
