import tensorflow as tf
from config import FLAGS
from data_utils import DataUtils


class TensorflowUtils(object):

    def __init__(self):
        self.vocab_path = FLAGS.vocab_path

        self.batch_size = FLAGS.batch_size
        self.num_steps = FLAGS.num_steps
        self.min_after_dequeue = FLAGS.min_after_dequeue
        self.num_threads = FLAGS.num_threads

        self.data_utils = DataUtils()
        self.default_word_padding_token = self.data_utils._START_VOCAB_ID[0]
        self.default_label_padding_token = self.data_utils.load_default_label_token()


    def load_embedding(self, vocab_filename, embedding_filename):
        embedding_filename = 'data/raw-data/embedding_vectors.txt'
        # embedding_dict = dict()
        # with open(embedding_filename, encoding='utf-8', mode='rt') as data_file:
        #     is_title = True
        #     for line in data_file:
        #         words = line.split()
        #         if is_title:
        #             word_count = int(words[0])
        #             embedding_size = int(words[1])
        #             is_tile = False
        #         else:
        #             word = words[0]
        #             embedding = [float(num) for num in words[1:]]
        #             if len(embedding) != embedding_size:
        #                 raise Exception('The content of file : %s is error' %(embedding_filename))
        #             embedding_matrix.append(embedding)


    def create_record(self, words_list, labels_list, tfrecords_filename):
        """
        store data into tfrecords file
        :param words_list:
        :param labels_list:
        :param tfrecords_filename:
        :return:
        """
        print('Create record to ' + tfrecords_filename)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        assert len(words_list) == len(labels_list)
        for (word_ids, label_ids) in zip(words_list, labels_list):
            word_list = [int(word) for word in word_ids.strip().split()]
            label_list = [int(label) for label in label_ids.strip().split()]
            assert len(word_list) == len(label_list)
            example = tf.train.Example(features=tf.train.Features(feature={
                'words': tf.train.Feature(int64_list=tf.train.Int64List(value=word_list)),
                'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=label_list)),
            }))
            writer.write(example.SerializeToString())
        writer.close()


    def read_and_decode(self, tfrecords_filename):
        """
        shuffled read batch data from tfrecords file
        :param tfrecords_filename:
        :return:
        """
        print('Read record from ' + tfrecords_filename)
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feature_configs = {
            # 'words': tf.FixedLenFeature(shape=[num_steps], dtype=tf.int64, default_value=0),
            'words': tf.VarLenFeature(dtype=tf.int64),
            'labels': tf.VarLenFeature(dtype=tf.int64),
        }
        features = tf.parse_single_example(serialized_example, features=feature_configs)
        words = features['words']
        words_len = words.dense_shape[0]
        words_len = tf.minimum(words_len, tf.constant(self.num_steps, tf.int64))
        words = tf.sparse_to_dense(sparse_indices=words.indices[: self.num_steps], output_shape=[self.num_steps],
                                   sparse_values=words.values[: self.num_steps], default_value=self.default_word_padding_token)
        labels = features['labels']
        labels = tf.sparse_to_dense(sparse_indices=labels.indices[: self.num_steps], output_shape=[self.num_steps],
                                    sparse_values=labels.values[: self.num_steps], default_value=self.default_label_padding_token)
        capacity = self.min_after_dequeue + 3 * self.batch_size
        words_batch, labels_batch, words_len_batch = tf.train.shuffle_batch([words, labels, words_len],
                                                                            batch_size=self.batch_size, capacity=capacity,
                                                                            min_after_dequeue=self.min_after_dequeue,
                                                                            num_threads=self.num_threads)
        return words_batch, labels_batch, words_len_batch


    def print_all(self, tfrecords_filename):
        """
        print all data from tfrecords file
        :param tfrecords_filename:
        :return:
        """
        number = 1
        for serialized_example in tf.python_io.tf_record_iterator(tfrecords_filename):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            words = example.features.feature['words'].int64_list.value
            labels = example.features.feature['labels'].int64_list.value
            word_list = [word for word in words]
            label_list = [label for label in labels]
            print('Number:{}, labels: {}, features: {}'.format(number, label_list, word_list))
            number += 1


    def print_shuffle(self, tfrecords_filename):
        """
        print shuffled data from tfrecords file calling read_and_decode method
        :param tfrecords_filename:
        :return:
        """
        words_batch, labels_batch, words_len_batch = self.read_and_decode(tfrecords_filename)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    batch_words_r, batch_labels_r, batch_words_len_r = sess.run([words_batch, labels_batch, words_len_batch])
                    print('batch_words_r : ', batch_words_r.shape)
                    print(batch_words_r)
                    print('batch_labels_r : ', batch_labels_r.shape)
                    print(batch_labels_r)
                    print('batch_words_len_r : ', batch_words_len_r.shape)
                    print(batch_words_len_r)
            except tf.errors.OutOfRangeError:
                print('Done reading')
            finally:
                coord.request_stop()
            coord.join(threads)


    def sparse_concat(self, sparse_tensor_input, base_tensor, excess_tensor, default_value):
        """
        extend sparse_tensor_input using base_indices and excess_indices
        :param sparse_tensor_input:
        :param base_indices:
        :param base_shape:
        :param excess_indices:
        :param excess_value_shape:
        :param excess_shape:
        :param default_value:
        :return:
        """
        # extract real blstm predict in dense and save to sparse
        base_sparse_tensor = tf.SparseTensor(indices=base_tensor.indices,
                                             values=tf.gather_nd(sparse_tensor_input, base_tensor.indices),
                                             dense_shape=base_tensor.dense_shape)

        # create excess SparseTensor with default_value
        excess_sparse_tensor = tf.SparseTensor(indices=excess_tensor.indices,
                                               values=tf.fill(tf.shape(excess_tensor.values), default_value),
                                               dense_shape=excess_tensor.dense_shape)

        # concat SparseTensor
        concat_sparse_tensor = tf.SparseTensor(
            indices=tf.concat(axis=0, values=[base_sparse_tensor.indices, excess_sparse_tensor.indices]),
            values=tf.concat(axis=0, values=[base_sparse_tensor.values, excess_sparse_tensor.values]),
            dense_shape=excess_sparse_tensor.dense_shape)
        concat_sparse_tensor = tf.sparse_reorder(concat_sparse_tensor)
        return concat_sparse_tensor


    def sparse_string_join(self, sparse_tensor_input, name):
        """
        join SparseTensor to 1-D String dense Tensor
        :param sparse_tensor_input:
        :param name:
        :return:
        """
        dense_tensor_input = tf.sparse_to_dense(sparse_indices=sparse_tensor_input.indices,
                                                output_shape=sparse_tensor_input.dense_shape,
                                                sparse_values=sparse_tensor_input.values,
                                                default_value='')
        dense_tensor_input_join = tf.reduce_join(dense_tensor_input, axis=1, separator=' ')
        format_predict_labels = tf.string_strip(dense_tensor_input_join, name=name)
        return format_predict_labels
