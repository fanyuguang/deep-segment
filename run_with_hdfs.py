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


class Segmenter(object):
    def __init__(self, hdfs_client, flags):
        self.train_is_alive = False
        self.hdfs_client = hdfs_client
        self.flags = flags
        self.data_utils = DataUtils()


    def update_config(self):
        config_path = os.path.join(self.flags.raw_data_path, 'config.json')
        try:
            with open(config_path, encoding='utf-8', mode='r') as data_file:
                config_json = json.load(data_file)
                if 'use_lstm' in config_json:
                    self.flags.use_lstm = config_json['use_lstm']
                elif 'use_dynamic_rnn' in config_json:
                    self.flags.use_dynamic_rnn = config_json['use_dynamic_rnn']
                elif 'use_bidirectional_rnn' in config_json:
                    self.flags.use_bidirectional_rnn = config_json['use_bidirectional_rnn']
                elif 'vocab_drop_limit' in config_json:
                    self.flags.vocab_drop_limit = config_json['vocab_drop_limit']
                elif 'batch_size' in config_json:
                    self.flags.batch_size = config_json['batch_size']
                elif 'num_steps' in config_json:
                    self.flags.num_steps = config_json['num_steps']
                elif 'num_layer' in config_json:
                    self.flags.num_layer = config_json['num_layer']
                elif 'embedding_size' in config_json:
                    self.flags.embedding_size = config_json['embedding_size']
                elif 'learning_rate' in config_json:
                    self.flags.learning_rate = config_json['learning_rate']
                elif 'learning_rate_decay_factor' in config_json:
                    self.flags.learning_rate_decay_factor = config_json['learning_rate_decay_factor']
                elif 'keep_prob' in config_json:
                    self.flags.keep_prob = config_json['keep_prob']
                elif 'clip_norm' in config_json:
                    self.flags.clip_norm = config_json['clip_norm']
        except:
            raise Exception('ERROR: config.json content invalid')


    def train(self):
        self.hdfs_client.hdfs_download(os.path.join(self.flags.input_path, 'train.txt'), os.path.join(self.flags.datasets_path, 'train.txt'))
        self.hdfs_client.hdfs_download(os.path.join(self.flags.input_path, 'test.txt'), os.path.join(self.flags.datasets_path, 'test.txt'))

        self.data_utils.label_segment_file(os.path.join(self.flags.datasets_path, 'train.txt'), os.path.join(self.flags.datasets_path, 'label_train.txt'))
        self.data_utils.label_segment_file(os.path.join(self.flags.datasets_path, 'test.txt'), os.path.join(self.flags.datasets_path, 'label_test.txt'))

        self.data_utils.split_label_file(os.path.join(self.flags.datasets_path, 'label_train.txt'), os.path.join(self.flags.datasets_path, 'split_train.txt'))
        self.data_utils.split_label_file(os.path.join(self.flags.datasets_path, 'label_test.txt'), os.path.join(self.flags.datasets_path, 'split_test.txt'))

        words_vocab, labels_vocab = self.data_utils.create_vocabulary(os.path.join(self.flags.datasets_path, 'split_train.txt'), self.flags.vocab_path, self.flags.vocab_drop_limit)

        train_word_ids_list, train_label_ids_list = self.data_utils.file_to_word_ids(os.path.join(self.flags.datasets_path, 'split_train.txt'), words_vocab, labels_vocab)
        test_word_ids_list, test_label_ids_list = self.data_utils.file_to_word_ids(os.path.join(self.flags.datasets_path, 'split_test.txt'), words_vocab, labels_vocab)

        tensorflow_utils = TensorflowUtils()
        tensorflow_utils.create_record(train_word_ids_list, train_label_ids_list, os.path.join(self.flags.tfrecords_path, 'train.tfrecords'))
        tensorflow_utils.create_record(test_word_ids_list, test_label_ids_list, os.path.join(self.flags.tfrecords_path, 'test.tfrecords'))

        self.hdfs_client.hdfs_upload(self.flags.vocab_path, os.path.join(self.flags.output_path, os.path.basename(self.flags.vocab_path)))

        train = Train()
        train.train()


    def upload_tensorboard(self):
        hdfs_tensorboard_path = os.path.join(self.flags.output_path, os.path.basename(os.path.normpath(self.flags.tensorboard_path)))
        temp_hdfs_tensorboard_path = hdfs_tensorboard_path + '-temp'
        self.hdfs_client.hdfs_upload(self.flags.tensorboard_path, temp_hdfs_tensorboard_path)
        self.hdfs_client.hdfs_delete(hdfs_tensorboard_path)
        self.hdfs_client.hdfs_mv(temp_hdfs_tensorboard_path, hdfs_tensorboard_path)


    def log_monitor(self):
        while(self.train_is_alive):
            time.sleep(120)
            self.upload_tensorboard()


    def upload_model(self):
        predict = Predict()
        predict.saved_model_pb()

        hdfs_checkpoint_path = os.path.join(self.flags.output_path, os.path.basename(os.path.normpath(self.flags.checkpoint_path)))
        hdfs_saved_model_path = os.path.join(self.flags.output_path, os.path.basename(os.path.normpath(self.flags.saved_model_path)))

        temp_hdfs_checkpoint_path = hdfs_checkpoint_path + '-temp'
        temp_hdfs_saved_model_path = hdfs_saved_model_path + '-temp'

        self.hdfs_client.hdfs_upload(self.flags.checkpoint_path, temp_hdfs_checkpoint_path)
        self.hdfs_client.hdfs_upload(self.flags.saved_model_path, temp_hdfs_saved_model_path)

        self.hdfs_client.hdfs_delete(hdfs_checkpoint_path)
        self.hdfs_client.hdfs_delete(hdfs_saved_model_path)

        self.hdfs_client.hdfs_mv(temp_hdfs_checkpoint_path, hdfs_checkpoint_path)
        self.hdfs_client.hdfs_mv(temp_hdfs_saved_model_path, hdfs_saved_model_path)


    def evaluate(self):
        shutil.rmtree(self.flags.vocab_path)
        shutil.rmtree(self.flags.checkpoint_path)

        self.hdfs_client.hdfs_download(os.path.join(self.flags.input_path, os.path.basename(self.flags.vocab_path)), self.flags.vocab_path)
        self.hdfs_client.hdfs_download(os.path.join(self.flags.input_path, 'test.txt'), os.path.join(self.flags.datasets_path, 'test.txt'))
        hdfs_checkpoint_path = os.path.join(self.flags.input_path, os.path.basename(self.flags.checkpoint_path))
        self.hdfs_client.hdfs_download(hdfs_checkpoint_path, self.flags.checkpoint_path)

        self.data_utils.label_segment_file(os.path.join(self.flags.datasets_path, 'test.txt'), os.path.join(self.flags.datasets_path, 'label_test.txt'))
        self.data_utils.split_label_file(os.path.join(self.flags.datasets_path, 'label_test.txt'), os.path.join(self.flags.datasets_path, 'split_test.txt'))

        predict = Predict()
        predict.file_predict(os.path.join(self.flags.datasets_path, 'split_test.txt'), os.path.join(self.flags.datasets_path, 'test_predict.txt'))

        self.model_evaluate = Evaluate()
        self.model_evaluate.evaluate(os.path.join(self.flags.datasets_path, 'test_predict.txt'), os.path.join(self.flags.datasets_path, 'test_evaluate.txt'))

        self.hdfs_client.hdfs_delete(os.path.join(self.flags.output_path, 'test_evaluate.txt'))
        self.hdfs_client.hdfs_upload(os.path.join(self.flags.datasets_path, 'test_evaluate.txt'), os.path.join(self.flags.input_path, 'test_evaluate.txt'))


def main():
    hdfs_client = HdfsUtils(FLAGS.hdfs_host, FLAGS.hdfs_port, FLAGS.hdfs_user)
    hdfs_client.hdfs_download(os.path.join(FLAGS.input_path, 'config.json'), os.path.join(FLAGS.raw_data_path, 'config.json'))

    segmenter = Segmenter(hdfs_client, FLAGS)
    segmenter.update_config()

    if FLAGS.train_evaluate == 'train':
        threads = []
        threads.append(threading.Thread(target=segmenter.train))
        threads.append(threading.Thread(target=segmenter.log_monitor))
        for thread in threads:
            thread.start()
        thread.join()
        time.sleep(5)
        segmenter.upload_data()
    elif FLAGS.train_evaluate == 'evaluate':
        segmenter.evaluate()

if __name__ == '__main__':
    main()
