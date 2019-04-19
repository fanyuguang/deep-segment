#!/usr/bin/python3
# -*- coding: utf-8 -*-

import datetime
import os
import pytz
import tensorflow as tf

from config import FLAGS
from utils.data_utils import DataUtils
from utils.tensorflow_utils import TensorflowUtils
from model import SequenceLabelingModel


class Train(object):

    def __init__(self):
        self.tfrecords_path = FLAGS.tfrecords_path
        self.checkpoint_path = FLAGS.checkpoint_path
        self.tensorboard_path = FLAGS.tensorboard_path

        self.use_crf = FLAGS.use_crf
        self.learning_rate = FLAGS.learning_rate
        self.learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
        self.decay_steps = FLAGS.decay_steps
        self.clip_norm = FLAGS.clip_norm
        self.max_training_step = FLAGS.max_training_step

        self.train_tfrecords_filename = os.path.join(self.tfrecords_path, 'train.tfrecords')
        self.test_tfrecords_filename = os.path.join(self.tfrecords_path, 'test.tfrecords')

        self.data_utils = DataUtils()
        self.num_classes = self.data_utils.get_vocabulary_size(os.path.join(FLAGS.vocab_path, 'labels_vocab.txt'))
        self.tensorflow_utils = TensorflowUtils()
        self.sequence_labeling_model = SequenceLabelingModel()


    def train(self):
        """
        train bilstm + crf model
        :return:
        """
        train_data = self.tensorflow_utils.read_and_decode(self.train_tfrecords_filename)
        train_batch_features, train_batch_labels, train_batch_features_lengths = train_data
        test_data = self.tensorflow_utils.read_and_decode(self.test_tfrecords_filename)
        test_batch_features, test_batch_labels, test_batch_features_lengths = test_data

        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_steps, self.learning_rate_decay_factor, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(lr)

        with tf.variable_scope('model'):
            logits = self.sequence_labeling_model.inference(train_batch_features, train_batch_features_lengths, self.num_classes, is_training=True)
        train_batch_labels = tf.to_int64(train_batch_labels)

        if self.use_crf:
            loss, transition_params = self.sequence_labeling_model.crf_loss(logits, train_batch_labels, train_batch_features_lengths, self.num_classes)
        else:
            slice_logits, slice_train_batch_labels = self.sequence_labeling_model.slice_seq(logits, train_batch_labels, train_batch_features_lengths)
            loss = self.sequence_labeling_model.loss(slice_logits, slice_train_batch_labels)

        with tf.variable_scope('model', reuse=True):
            accuracy_logits = self.sequence_labeling_model.inference(test_batch_features, test_batch_features_lengths, self.num_classes, is_training=False)
        test_batch_labels = tf.to_int64(test_batch_labels)
        if self.use_crf:
            accuracy = self.sequence_labeling_model.crf_accuracy(accuracy_logits, test_batch_labels, test_batch_features_lengths,
                                                   transition_params, self.num_classes)
        else:
            slice_accuracy_logits, slice_test_batch_labels = self.sequence_labeling_model.slice_seq(accuracy_logits, test_batch_labels,
                                                                                          test_batch_features_lengths)
            accuracy = self.sequence_labeling_model.accuracy(slice_accuracy_logits, slice_test_batch_labels)

        # summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('lr', lr)

        # compute and update gradient
        # train_op = optimizer.minimize(loss, global_step=global_step)

        # computer, clip and update gradient
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        train_op = optimizer.apply_gradients(zip(clip_gradients, variables), global_step=global_step)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=None)
        checkpoint_filename = os.path.join(self.checkpoint_path, 'model.ckpt')

        with tf.Session() as sess:
            summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.tensorboard_path, sess.graph)
            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('Continue training from the model {}'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            max_accuracy = 0.0
            min_loss = 100000000.0
            try:
                while not coord.should_stop():
                    _, loss_value, step = sess.run([train_op, loss, global_step])
                    if step % 100 == 0:
                        accuracy_value, summary_value, lr_value = sess.run([accuracy, summary_op, lr])
                        china_tz = pytz.timezone('Asia/Shanghai')
                        current_time = datetime.datetime.now(china_tz)
                        print('[{}] Step: {}, loss: {}, accuracy: {}, lr: {}'.format(current_time, step, loss_value, accuracy_value, lr_value))
                        if accuracy_value > max_accuracy and loss_value < min_loss:
                            writer.add_summary(summary_value, step)
                            saver.save(sess, checkpoint_filename, global_step=step)
                            print('save model to %s-%d' % (checkpoint_filename, step))
                            max_accuracy = accuracy_value
                            min_loss = loss_value
                    if step >= self.max_training_step:
                        print('Done training after %d step' % step)
                        break
            except tf.errors.OutOfRangeError:
                print('Done training after reading all data')
            finally:
                coord.request_stop()
            coord.join(threads)


def main(_):
    data_utils = DataUtils()
    data_utils.prepare_datasets(os.path.join(FLAGS.raw_data_path, 'data_demo.txt'), FLAGS.test_percent,
                                os.path.join(FLAGS.datasets_path, 'train.txt'), os.path.join(FLAGS.datasets_path, 'test.txt'))

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

    tensorflow_utils.print_all(os.path.join(FLAGS.tfrecords_path, 'train.tfrecords'))
    # tensorflow_utils.print_shuffle(os.path.join(FLAGS.tfrecords_path, 'train.tfrecords'))

    segment_train = Train()
    segment_train.train()


if __name__ == '__main__':
    tf.app.run()
