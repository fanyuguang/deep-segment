import os
import shutil
import tensorflow as tf
from tensorflow.contrib import lookup
import tensorflow.contrib.crf as crf
from config import FLAGS
from data_utils import DataUtils
from tensorflow_utils import TensorflowUtils
from model import NerModel


class NerPredict(object):

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
        self.ner_model = NerModel()
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

        # dict words to token ids
        with open(os.path.join(self.vocab_path, 'words_vocab.txt'), encoding='utf-8', mode='rt') as data_file:
            words_table_list = [line.strip() for line in data_file if line.strip()]
        words_table_tensor = tf.constant(words_table_list, dtype=tf.string)
        words_table = lookup.index_table_from_tensor(mapping=words_table_tensor, default_value=self.data_utils._START_VOCAB_ID[3])
        # words_table = lookup.index_table_from_file(os.path.join(vocab_path, 'words_vocab.txt'), default_value=3)
        words_ids = words_table.lookup(words)

        # blstm model predict
        with tf.variable_scope('model', reuse=None):
            logits = self.ner_model.inference(words_ids, input_sentences_lengths, self.num_classes, is_training=False)

        if self.use_crf:
            logits = tf.reshape(logits, shape=[-1, self.num_steps, self.num_classes])
            transition_params = tf.get_variable("transitions", [self.num_classes, self.num_classes])
            input_sentences_lengths = tf.to_int32(input_sentences_lengths)
            predict_labels_ids, sequence_scores = crf.crf_decode(logits, transition_params, input_sentences_lengths)
            predict_labels_ids = tf.to_int64(predict_labels_ids)
            sequence_scores = tf.reshape(sequence_scores, shape=[-1, 1])
            predict_scores = tf.matmul(sequence_scores, tf.ones(shape=[1, self.num_steps], dtype=tf.float32))
        else:
            props = tf.nn.softmax(logits)
            max_prop_values, max_prop_indices = tf.nn.top_k(props, k=1)
            predict_labels_ids = tf.reshape(max_prop_indices, shape=[-1, self.num_steps])
            predict_labels_ids = tf.to_int64(predict_labels_ids)
            predict_scores = tf.reshape(max_prop_values, shape=[-1, self.num_steps])
        predict_scores = tf.as_string(predict_scores, precision=3)

        # dict token ids to labels
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
        predict labels, the operation of transfer words to id token is processed by tensorflow tensor
        input words list, now, only support one element list
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
        predict data_filename, save the predict result into predict_filename
        the label is split into single word, -B -M -E -S
        :param data_filename:
        :param predict_filename:
        :return:
        """
        print('Predict file ' + data_filename)
        sentence_list = []
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                line = ''.join(line.strip().split())
                if line:
                    words = [char for char in line]
                    predict_labels, _ = self.predict([' '.join(words)])
                    predict_labels = predict_labels[0].split()
                    merge_word_list, _ = self.data_utils.merge_label(words, predict_labels)
                    sentence_list.append(' '.join(merge_word_list))
                else:
                    sentence_list.append('')
        with open(predict_filename, encoding='utf-8', mode='wt') as predict_file:
            for sentence in sentence_list:
                predict_file.write(sentence + '\n')


    def freeze_graph(self):
        # save graph into .pb file
        graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ['init_all_tables', 'predict_labels', 'predict_scores'])
        tf.train.write_graph(graph, self.freeze_graph_path, 'frozen_graph.pb', as_text=False)


    def saved_model_pb(self):
        # for tensorflow serving, saved model into .ph and variables files
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
            signature_def_map={'predict_ner': prediction_signature},
            legacy_init_op=legacy_init_op
        )
        builder.save()
        print('Successfully exported model to %s' % saved_model_path)


def main(_):
    ner_predict = NerPredict()

    sentence = '空间绝对够用，对于我这1.68米的个子就算1.80米 大个也绝对够用。后排座椅，也绝对够宽敞。拉4个都不算太挤'
    words = [char for char in ''.join(sentence.split())]
    predict_labels, predict_scores = ner_predict.predict([' '.join(words)])

    data_utils = DataUtils()
    merge_word_list, _ = data_utils.merge_label(words, predict_labels[0].split())
    print(' '.join(merge_word_list))

    ner_predict.file_predict(os.path.join(FLAGS.datasets_path, 'saic_gm_segmentation_20181122_random_sample_1000.tsv'),
                             os.path.join(FLAGS.datasets_path, 'saic_gm_segmentation_20181122_random_sample_1000_predict.tsv'))


if __name__ == '__main__':
    tf.app.run()
