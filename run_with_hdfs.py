#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import threading
import time
import tensorflow as tf

from config import FLAGS
from evaluate import Evaluate
from predict import Predict
from train import Train
from utils.data_utils import DataUtils
from utils.hdfs_utils import HdfsUtils
from utils.tensorflow_utils import TensorflowUtils

tf.app.flags.DEFINE_string('train_evaluate', 'evaluate', 'train or evaluate')

tf.app.flags.DEFINE_string('hdfs_host', 'hdfs-bizaistca.corp.microsoft.com', 'hdfs host')
tf.app.flags.DEFINE_integer('hdfs_port', 8020, 'hdfs port')
tf.app.flags.DEFINE_string('hdfs_user', 'hadoop', 'hdfs user')

# tf.app.flags.DEFINE_string('input_path', '/user/hadoop/data/input/', 'input data path')
# tf.app.flags.DEFINE_string('output_path', '/user/hadoop/data/output_path/', 'output_path data path')
tf.app.flags.DEFINE_string('input_path', '/user/hadoop/fanyuguang/input/', 'input data path')
tf.app.flags.DEFINE_string('output_path', '/user/hadoop/fanyuguang/output/', 'output data path')


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


class TrainMonitor(object):
    def __init__(self):
        self.train_is_alive = False
        self.train = Train()
        self.predict = Predict()


    def train(self):
        self.train.train()


    def upload_log_model_data(self, hdfs_client, flags):
        self.predict.saved_model_pb()

        hdfs_tensorboard_path = os.path.join(FLAGS.output_path, os.path.basename(os.path.normpath(flags.tensorboard_path)))
        hdfs_checkpoint_path = os.path.join(FLAGS.output_path, os.path.basename(os.path.normpath(flags.checkpoint_path)))
        hdfs_saved_model_path = os.path.join(FLAGS.output_path, os.path.basename(os.path.normpath(flags.saved_model_path)))

        temp_hdfs_tensorboard_path = hdfs_tensorboard_path + '-temp'
        temp_hdfs_checkpoint_path = hdfs_checkpoint_path + '-temp'
        temp_hdfs_saved_model_path = hdfs_saved_model_path + '-temp'

        hdfs_client.hdfs_upload(flags.tensorboard_path, temp_hdfs_tensorboard_path)
        hdfs_client.hdfs_upload(flags.checkpoint_path, temp_hdfs_checkpoint_path)
        hdfs_client.hdfs_upload(flags.saved_model_path, temp_hdfs_saved_model_path)

        hdfs_client.hdfs_delete(hdfs_tensorboard_path)
        hdfs_client.hdfs_delete(hdfs_checkpoint_path)
        hdfs_client.hdfs_delete(hdfs_saved_model_path)

        hdfs_client.hdfs_mv(temp_hdfs_tensorboard_path, hdfs_tensorboard_path)
        hdfs_client.hdfs_mv(temp_hdfs_checkpoint_path, hdfs_checkpoint_path)
        hdfs_client.hdfs_mv(temp_hdfs_saved_model_path, hdfs_saved_model_path)


    def upload(self, hdfs_client, flags):
        while(self.train_is_alive):
            time.sleep(120)
            self.upload_log_model_data(hdfs_client, flags)


def main():
    hdfs_client = HdfsUtils(FLAGS.hdfs_host, FLAGS.hdfs_port, FLAGS.hdfs_user)

    hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'config.json'), os.path.join(FLAGS.raw_data_path, 'config.json'))
    update_config(os.path.join(FLAGS.raw_data_path, 'config.json'))

    data_utils = DataUtils()

    if FLAGS.train_evaluate == 'train':
        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'train.txt'), os.path.join(FLAGS.datasets_path, 'train.txt'))
        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'test.txt'))

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

        hdfs_client.hdfs_upload(FLAGS.vocab_path, os.path.join(FLAGS.output_path, os.path.basename(FLAGS.vocab_path)))

        threads = []
        train_monitor = TrainMonitor()
        threads.append(threading.Thread(target=train_monitor.train))
        threads.append(threading.Thread(target=train_monitor.predict, args=(data_utils, FLAGS)))
        for thread in threads:
            thread.start()
        thread.join()
        train_monitor.upload_log_model_data(hdfs_client, FLAGS)
    elif FLAGS.train_evaluate == 'evaluate':
        shutil.rmtree(FLAGS.vocab_path)
        shutil.rmtree(FLAGS.checkpoint_path)

        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, os.path.basename(FLAGS.vocab_path)), FLAGS.vocab_path)
        hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'test.txt'))
        hdfs_checkpoint_path = os.path.join(FLAGS.input_path, os.path.basename(FLAGS.checkpoint_path))
        hdfs_client.hdfs_download(hdfs_checkpoint_path, FLAGS.checkpoint_path)

        data_utils.label_segment_file(os.path.join(FLAGS.datasets_path, 'test.txt'), os.path.join(FLAGS.datasets_path, 'label_test.txt'))
        data_utils.split_label_file(os.path.join(FLAGS.datasets_path, 'label_test.txt'), os.path.join(FLAGS.datasets_path, 'split_test.txt'))

        predict = Predict()
        predict.file_predict(os.path.join(FLAGS.datasets_path, 'split_test.txt'), os.path.join(FLAGS.datasets_path, 'test_predict.txt'))

        evaluate = Evaluate()
        evaluate.evaluate(os.path.join(FLAGS.datasets_path, 'test_predict.txt'), os.path.join(FLAGS.datasets_path, 'test_evaluate.txt'))

        hdfs_client.hdfs_delete(os.path.join(FLAGS.output_path, 'test_evaluate.txt'))
        hdfs_client.hdfs_upload(os.path.join(FLAGS.datasets_path, 'test_evaluate.txt'), os.path.join(FLAGS.input_path, 'test_evaluate.txt'))


if __name__ == '__main__':
    main()
