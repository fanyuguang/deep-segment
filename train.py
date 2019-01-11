import datetime
import os
import tensorflow as tf
from config import FLAGS
from data_utils import DataUtils
from tensorflow_utils import TensorflowUtils
from model import NerModel


class Train(object):

    def __init__(self):
        self.vocab_path = FLAGS.vocab_path
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
        self.validate_tfrecords_filename = os.path.join(self.tfrecords_path, 'validate.tfrecords')

        self.data_utils = DataUtils()
        self.num_classes = self.data_utils.load_num_classes()
        self.tensorflow_utils = TensorflowUtils()
        self.ner_model = NerModel()


    def train(self):
        """
        train bilstm + crf model
        :return:
        """
        train_data = self.tensorflow_utils.read_and_decode(self.train_tfrecords_filename)
        train_batch_features, train_batch_labels, train_batch_features_lengths = train_data
        validate_data = self.tensorflow_utils.read_and_decode(self.validate_tfrecords_filename)
        validate_batch_features, validate_batch_labels, validate_batch_features_lengths = validate_data

        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_steps, self.learning_rate_decay_factor, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(lr)

        with tf.variable_scope('model'):
            logits = self.ner_model.inference(train_batch_features, train_batch_features_lengths, self.num_classes, is_training=True)
        train_batch_labels = tf.to_int64(train_batch_labels)

        if self.use_crf:
            loss, transition_params = self.ner_model.crf_loss(logits, train_batch_labels, train_batch_features_lengths, self.num_classes)
        else:
            slice_logits, slice_train_batch_labels = self.ner_model.slice_seq(logits, train_batch_labels, train_batch_features_lengths)
            loss = self.ner_model.loss(slice_logits, slice_train_batch_labels)

        with tf.variable_scope('model', reuse=True):
            accuracy_logits = self.ner_model.inference(validate_batch_features, validate_batch_features_lengths, self.num_classes, is_training=False)
        validate_batch_labels = tf.to_int64(validate_batch_labels)
        if self.use_crf:
            accuracy = self.ner_model.crf_accuracy(accuracy_logits, validate_batch_labels, validate_batch_features_lengths,
                                                   transition_params, self.num_classes)
        else:
            slice_accuracy_logits, slice_validate_batch_labels = self.ner_model.slice_seq(accuracy_logits, validate_batch_labels,
                                                                                          validate_batch_features_lengths)
            accuracy = self.ner_model.accuracy(slice_accuracy_logits, slice_validate_batch_labels)

        # summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('lr', lr)
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        # compute and update gradient
        # train_op = optimizer.minimize(loss, global_step=global_step)

        # computer, clip and update gradient
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        train_op = optimizer.apply_gradients(zip(clip_gradients, variables), global_step=global_step)

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=None)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
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

            start_time = datetime.datetime.now()
            try:
                while not coord.should_stop():
                    _, loss_value, step = sess.run([train_op, loss, global_step])
                    if step % 100 == 0:
                        accuracy_value, summary_value, lr_value = sess.run([accuracy, summary_op, lr])
                        end_time = datetime.datetime.now()
                        print('[{}] Step: {}, loss: {}, accuracy: {}, lr: {}'.format(end_time - start_time, step, loss_value, accuracy_value, lr_value))
                        if step % 1000 == 0:
                            writer.add_summary(summary_value, step)
                            saver.save(sess, checkpoint_filename, global_step=step)
                            print('save model to %s-%d' % (checkpoint_filename, step))
                        start_time = end_time
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
    
    # format datasets
    # data_utils.format_data(os.path.join(FLAGS.raw_data_path, 'data.txt'), os.path.join(FLAGS.raw_data_path, 'format_data.txt'))
    # if FLAGS.use_char_based:
    #     data_utils.split_label_file(os.path.join(FLAGS.raw_data_path, 'data.txt'), os.path.join(FLAGS.raw_data_path, 'split_data.txt'))
    
    # # datasets process
    # data_utils.prepare_datasets(os.path.join(FLAGS.raw_data_path, 'split_data.txt'), FLAGS.train_percent, FLAGS.val_percent, FLAGS.datasets_path)
    # words_vocab, labels_vocab, _ = data_utils.create_vocabulary(os.path.join(FLAGS.datasets_path, 'train.txt'), FLAGS.vocab_path, FLAGS.vocab_size)
    # 
    # train_word_ids_list, train_label_ids_list = data_utils.data_to_token_ids(os.path.join(FLAGS.datasets_path, 'train.txt'), words_vocab, labels_vocab)
    # validation_word_ids_list, validation_label_ids_list = data_utils.data_to_token_ids(os.path.join(FLAGS.datasets_path, 'validation.txt'), words_vocab, labels_vocab)
    # test_word_ids_list, test_label_ids_list = data_utils.data_to_token_ids(os.path.join(FLAGS.datasets_path, 'test.txt'), words_vocab, labels_vocab)
    # 
    # tensorflow_utils = TensorflowUtils()
    # 
    # tensorflow_utils.create_record(train_word_ids_list, train_label_ids_list, os.path.join(FLAGS.tfrecords_path, 'train.tfrecords'))
    # tensorflow_utils.create_record(validation_word_ids_list, validation_label_ids_list, os.path.join(FLAGS.tfrecords_path, 'validate.tfrecords'))
    # tensorflow_utils.create_record(test_word_ids_list, test_label_ids_list, os.path.join(FLAGS.tfrecords_path, 'test.tfrecords'))
    # 
    # tensorflow_utils.print_all(os.path.join(FLAGS.tfrecords_path, 'train.tfrecords'))
    # tensorflow_utils.print_shuffle(os.path.join(FLAGS.tfrecords_path, 'test.tfrecords'))

    segment_train = Train()
    segment_train.train()


if __name__ == '__main__':
    tf.app.run()
