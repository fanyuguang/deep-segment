#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import tensorflow as tf

from config import FLAGS
from evaluate import Evaluate
from predict import Predict
from train import Train
from utils.data_utils import DataUtils
from utils.hdfs_utils import HdfsUtils
from utils.tensorflow_utils import TensorflowUtils

tf.app.flags.DEFINE_string('train_evaluate_export', 'train', 'train, evaluate or export')

tf.app.flags.DEFINE_string('hdfs_host', 'hdfs-bizaistca.corp.microsoft.com', 'hdfs host')
tf.app.flags.DEFINE_integer('hdfs_port', 8020, 'hdfs port')
tf.app.flags.DEFINE_string('hdfs_user', 'hadoop', 'hdfs user')

tf.app.flags.DEFINE_string('input_path', '/user/hadoop/data/input/', 'input data path')
tf.app.flags.DEFINE_string('log_path', '/user/hadoop/data/log/', 'log data path')
tf.app.flags.DEFINE_string('model_path', '/user/hadoop/data/model/', 'export model data path')

def update_config(config_path):
    try:
        with open(config_path, encoding='utf-8', mode='r') as data_file:
            config_json = json.load(data_file)
            if 'use_lstm' in config_json:
                FLAGS.use_lstm = config_json['use_lstm']
            elif'use_dynamic_rnn' in config_json:
                FLAGS.use_dynamic_rnn = config_json['use_dynamic_rnn']
            elif 'use_bidirectional_rnn' in config_json:
                FLAGS.use_bidirectional_rnn = config_json['use_bidirectional_rnn']
            elif 'vocab_drop_limit' in config_json:
                FLAGS.vocab_drop_limit = config_json['vocab_drop_limit']
            elif 'batch_size' in config_json:
                FLAGS.batch_size = config_json['batch_size']
            elif 'num_steps' in config_json:
                FLAGS.num_steps = config_json['num_steps']
            elif 'num_layer' in config_json:
                FLAGS.num_layer = config_json['num_layer']
            elif 'embedding_size' in config_json:
                FLAGS.embedding_size = config_json['embedding_size']
            elif 'learning_rate' in config_json:
                FLAGS.learning_rate = config_json['learning_rate']
            elif 'learning_rate_decay_factor' in config_json:
                FLAGS.learning_rate_decay_factor = config_json['learning_rate_decay_factor']
            elif 'keep_prob' in config_json:
                FLAGS.keep_prob = config_json['keep_prob']
            elif 'clip_norm' in config_json:
                FLAGS.clip_norm = config_json['clip_norm']
    except:
        raise Exception('ERROR: config.json content invalid')


def main():
    hdfs_client = HdfsUtils(FLAGS.hdfs_host, FLAGS.hdfs_port, FLAGS.hdfs_user)
    data_utils = DataUtils()

    hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'config.json'), os.path.join(FLAGS.raw_data_path, 'config.json'))
    update_config(os.path.join(FLAGS.raw_data_path, 'config.json'))

    if FLAGS.train_evaluate_export == 'train':
        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'train.txt'), os.path.join(FLAGS.datasets_path, 'train.txt'))
        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'test.txt'))

        data_utils = DataUtils()

        data_utils.label_segment_file(os.path.join(FLAGS.datasets_path, 'train.txt'), os.path.join(FLAGS.datasets_path, 'label_train.txt'))
        data_utils.label_segment_file(os.path.join(FLAGS.datasets_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'label_test.txt'))

        data_utils.split_label_file(os.path.join(FLAGS.datasets_path, 'label_train.txt'), os.path.join(FLAGS.datasets_path, 'split_train.txt'))
        data_utils.split_label_file(os.path.join(FLAGS.datasets_path, 'label_test.txt'), os.path.join(FLAGS.datasets_path, 'split_test.txt'))

        words_vocab, labels_vocab = data_utils.create_vocabulary(os.path.join(FLAGS.datasets_path, 'split_train.txt'), FLAGS.vocab_path, FLAGS.vocab_drop_limit)

        train_word_ids_list, train_label_ids_list = data_utils.file_to_word_ids(os.path.join(FLAGS.datasets_path, 'split_train.txt'), words_vocab, labels_vocab)
        test_word_ids_list, test_label_ids_list = data_utils.file_to_word_ids(os.path.join(FLAGS.datasets_path, 'split_test.txt'), words_vocab, labels_vocab)

        tensorflow_utils = TensorflowUtils()
        tensorflow_utils.create_record(train_word_ids_list, train_label_ids_list, os.path.join(FLAGS.tfrecords_path, 'train.tfrecords'))
        tensorflow_utils.create_record(test_word_ids_list, test_label_ids_list, os.path.join(FLAGS.tfrecords_path, 'test.tfrecords'))

        segment_train = Train()
        segment_train.train()


        hdfs_client.hdfs_upload(FLAGS.vocab_path, FLAGS.log_path)
        hdfs_client.hdfs_upload(FLAGS.tensorboard_path, FLAGS.log_path)
        hdfs_client.hdfs_upload(FLAGS.tensorboard_path, FLAGS.log_path)
    elif FLAGS.train_evaluate_export == 'evaluate':
        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'evaluate.txt'), os.path.join(FLAGS.raw_data_path, 'evaluate.txt'))

        data_utils.label_segment_file(os.path.join(FLAGS.datasets_path, 'evaluate.txt'),os.path.join(FLAGS.datasets_path, 'label_evaluate.txt'))
        data_utils.split_label_file(os.path.join(FLAGS.datasets_path, 'label_evaluate.txt'),os.path.join(FLAGS.datasets_path, 'split_evaluate.txt'))
        predict = Predict()
        predict.file_predict(os.path.join(FLAGS.datasets_path, 'split_evaluate.txt'), os.path.join(FLAGS.datasets_path, 'evaluate_predict.txt'))

        evaluate = Evaluate()
        evaluate.evaluate(os.path.join(FLAGS.datasets_path, 'evaluate_predict.txt'))

        hdfs_client.hdfs_upload(FLAGS.tensorboard_path, os.path.join(FLAGS.output_path, 'tensorboard'))
    elif FLAGS.train_evaluate_export == 'export':
        predict = Predict()
        sentence = '张伟在6月16号会去一趟丹棱街中国移动营业厅'
        words = ' '.join([char for char in sentence])
        _, _ = predict.predict([words])
        predict.saved_model_pb()

        hdfs_client.hdfs_upload(FLAGS.saved_model_path, os.path.join(FLAGS.output_path, 'saved-model-data'))

if __name__ == '__main__':
    main()
