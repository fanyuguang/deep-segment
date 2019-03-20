#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import shutil

from config import FLAGS
from data_utils import DataUtils
from tensorflow_utils import TensorflowUtils
from train import Train
from predict import Predict
from hdfs3 import HDFileSystem
from shutil import copytree

tf.app.flags.DEFINE_string('hdfs_host', 'hdfs-bizaistca.corp.microsoft.com', 'hdfs host')
tf.app.flags.DEFINE_integer('hdfs_port', 8020, 'hdfs port')
tf.app.flags.DEFINE_string('hdfs_user', 'hadoop', 'hdfs user')

tf.app.flags.DEFINE_string('input_path', '/user/hadoop/data/input/', 'input data path')
tf.app.flags.DEFINE_string('output_path', '/user/hadoop/data/output/', 'output data path')

hdfs = HDFileSystem(host=FLAGS.hdfs_host, port=FLAGS.hdfs_port, user=FLAGS.hdfs_user)


def hdfs_download(hdfs_path, local_path):
    hdfs_path = os.path.normpath(hdfs_path)
    local_path = os.path.normpath(local_path)
    local_parent_path = os.path.dirname(local_path)
    if not hdfs.exists(hdfs_path):
        raise Exception('hdfs file not exists: ' + hdfs_path)
    if not os.path.exists(local_parent_path):
        raise Exception('local parent folder not exists: ' + local_parent_path)
    if os.path.exists(local_path):
        raise Exception('local file exists: ' + local_path)

    if hdfs.isfile(hdfs_path):
        print('is file')
        hdfs.get(hdfs_path, local_path)
    elif hdfs.isdir(hdfs_path):
        print('is dir')
        os.mkdir(local_path)
        for (root, dirnames, filenames) in hdfs.walk(hdfs_path):
            relative_path = os.path.relpath(root, hdfs_path)
            for dirname in dirnames:
                current_local_dir_path = os.path.join(local_path, relative_path, dirname)
                os.makedirs(current_local_dir_path)
            for filename in filenames:
                current_hdfs_file_path = os.path.join(root, filename)
                current_local_file_path = os.path.join(local_path, relative_path, filename)
                hdfs.get(current_hdfs_file_path, current_local_file_path)
    else:
        raise Exception('parameters invalid')
    print('Done.')


def hdfs_upload(local_path, hdfs_path):
    local_path = os.path.normpath(local_path)
    hdfs_path = os.path.normpath(hdfs_path)
    hdfs_parent_path = os.path.dirname(hdfs_path)
    if not os.path.exists(local_path):
        raise Exception('local file not exists: ' + local_path)
    if not hdfs.exists(hdfs_parent_path):
        raise Exception('hdfs parent folder not exists: ' + hdfs_parent_path)
    if hdfs.exists(hdfs_path):
        raise Exception('hdfs file exists: ' + hdfs_path)

    if os.path.isfile(local_path):
        print('is file')
        hdfs.put(local_path, hdfs_path)
    elif os.path.isdir(local_path):
        print('is dir')
        hdfs.mkdir(hdfs_path)
        for (root, dirnames, filenames) in os.walk(local_path):
            relative_path = os.path.relpath(root, local_path)
            for dirname in dirnames:
                current_hdfs_dir_path = os.path.join(hdfs_path, relative_path, dirname)
                hdfs.mkdir(current_hdfs_dir_path)
            for filename in filenames:
                current_local_file_path = os.path.join(root, filename)
                current_hdfs_file_path = os.path.join(hdfs_path, relative_path, filename)
                hdfs.put(current_local_file_path, current_hdfs_file_path)
    else:
        raise Exception('parameters invalid')
    print('Done.')


def main():
    hdfs_download(os.path.join(FLAGS.input_path, 'data.txt'), os.path.join(FLAGS.raw_data_path, 'data.txt'))
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
