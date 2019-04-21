#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import random

from config import FLAGS


class DataUtils(object):

    def __init__(self):
        self.vocab_path = FLAGS.vocab_path
        self.default_label = FLAGS.default_label

        # Special vocabulary symbols:
        # PAD is used to pad a sequence to a fixed size
        # GO is for the end of the encoding
        # EOS is for the end of decoding
        # UNK is for out of vocabulary words
        _PAD, _GO, _EOS, _UNK = '_PAD', '_GO', '_EOS', '_UNK'
        self._START_VOCAB = [_PAD, _GO, _EOS, _UNK]
        PAD_ID, GO_ID, EOS_ID, UNK_ID = range(4)
        self._START_VOCAB_ID = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

        specific_symbols_data_filename = os.path.join(FLAGS.raw_data_path, 'specific_symbols.txt')
        self.specific_symbols = self.load_specific_symbols(specific_symbols_data_filename)
        punctuations = '\,./<>?;\':"[]{}|`~!@#$%^&*()-=_+，。《》？；‘’：“”【】、{}·~！￥…（）—'
        self.punctuation_regex = re.escape(punctuations)


    def split(self, sentence):
        """
        Split sentence with format 'word1/label1||word2/label2' to word list and label list
        :param sentence:
        :return: word list and label list
        """
        word_label_list = re.split('\|{2}', sentence.strip())
        word_list = []
        label_list = []
        for word_label in word_label_list:
            if word_label:
                separator_index = word_label.rfind('/')
                if separator_index != -1:
                    word = word_label[: separator_index]
                    label = word_label[separator_index + 1:]
                    if word and label:
                        word_list.append(word)
                        label_list.append(label)
        return word_list, label_list


    def load_specific_symbols(self, data_filename):
        """
        Load specific_symbols from data_filename
        :param data_filename:
        :return: specific symbols list
        """
        specific_symbols = set()
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                symbol = ''.join(line.strip().split())
                if symbol:
                    specific_symbols.add(symbol)
        return specific_symbols


    def format_data(self, data_filename, format_data_filename):
        """
        Upper word, remove specific symbols
        Upper label
        :param data_filename:
        :param format_data_filename:
        :param specific_symbols_data_filename:
        :return:
        """
        print('Format data file ' + data_filename)
        format_data_list = []
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                word_list, label_list = self.split(line)
                if word_list and label_list:
                    word_label_list = []
                    for word, label in zip(word_list, label_list):
                        # format label
                        label = label.upper()
                        # format word, remove symbol, punctuation, other language and continuous space
                        word = word.lower()
                        for symbol in self.specific_symbols:
                            word = word.replace(symbol, ' ')
                        # word = re.sub('[%s]+' % self.punctuation_regex, ' ', word)
                        # word = re.sub('[^a-zA-Z0-9\u4e00-\u9fff]+', ' ', word)
                        word = re.sub('\s+', '', word)
                        if word and label:
                            word_label_list.append(word + '/' + label)
                    if word_label_list:
                        format_data_list.append('||'.join(word_label_list))

        with open(format_data_filename, encoding='utf-8', mode='wt') as format_data_file:
            for data in format_data_list:
                format_data_file.write(data + '\n')


    def split_label(self, sentence):
        """
        Split label of sentence to -B -M -E -S
        :param sentence:
        :return: labeled sentence with char-based
        """
        single_word_label_list = []
        word_list, label_list = self.split(sentence)
        if word_list and label_list:
            for word, label in zip(word_list, label_list):
                single_label_list = []
                word_length = len(word)
                if label == self.default_label:
                    single_label_list = [self.default_label for _ in range(word_length)]
                else:
                    if word_length == 1:
                        single_label_list = [label + '-S']
                    elif word_length == 2:
                        single_label_list = [label + '-B', label + '-E']
                    else:
                        single_label_list.append(label + '-B')
                        for index in range(word_length - 2):
                            single_label_list.append(label + '-M')
                        single_label_list.append(label + '-E')
                for (single_word, single_label) in zip(word, single_label_list):
                    single_word_label = single_word + '/' + single_label
                    single_word_label_list.append(single_word_label)
        split_sentence = '||'.join(single_word_label_list)
        return split_sentence


    def split_label_file(self, data_filename, split_data_filename):
        """
        Split label of file to -B -M -E -S
        :param data_filename:
        :param split_data_filename:
        :return:
        """
        print('Split label file ' + data_filename)
        with open(split_data_filename, encoding='utf-8', mode='wt') as new_data_file:
            with open(data_filename, encoding='utf-8', mode='rt') as raw_data_file:
                for line in raw_data_file:
                    sentence = self.split_label(line)
                    new_data_file.write(sentence + '\n')


    def label_segment_file(self, data_filename, label_data_filename):
        """
        Label of file for segment
        :param data_filename:
        :param split_data_filename:
        :return:
        """
        print('Label file ' + data_filename)
        with open(label_data_filename, encoding='utf-8', mode='wt') as new_data_file:
            with open(data_filename, encoding='utf-8', mode='rt') as raw_data_file:
                for line in raw_data_file:
                    words = line.strip().split()
                    sentence = '||'.join([word + '/WORD' for word in words if word])
                    new_data_file.write(sentence + '\n')


    def merge_label(self, word_list, label_list):
        """
        Merge split label, example label-B label-M label-E label-S to label
        :param word_list:
        :param label_list:
        :return: merged word list and label list with word-based
        """
        merge_word_list = []
        merge_label_list = []
        category = ''
        category_word_list = []
        self.default_segment_label = 'WORD'
        for (word, label) in zip(word_list, label_list):
            if word and label:
                if len(label) > 1 and label.find('-B') == len(label) - 2:
                    if category_word_list:
                        merge_word_list.extend(category_word_list)
                        merge_label_list.extend([self.default_segment_label] * len(category_word_list))
                        category_word_list = []
                    category = label[0:-2]
                    category_word_list.append(word)
                elif len(label) > 1 and label.find('-M') == len(label) - 2:
                    category_word_list.append(word)
                    if category != label[0:-2]:
                        merge_word_list.extend(category_word_list)
                        merge_label_list.extend([self.default_segment_label] * len(category_word_list))
                        category = ''
                        category_word_list = []
                elif len(label) > 1 and label.find('-E') == len(label) - 2:
                    category_word_list.append(word)
                    if category == label[0:-2]:
                        merge_word_list.append(''.join(category_word_list))
                        merge_label_list.append(category)
                    else:
                        merge_word_list.extend(category_word_list)
                        merge_label_list.extend([self.default_segment_label] * len(category_word_list))
                    category = ''
                    category_word_list = []
                elif len(label) > 1 and label.find('-S') == len(label) - 2:
                    if category_word_list:
                        merge_word_list.extend(category_word_list)
                        merge_label_list.extend([self.default_segment_label] * len(category_word_list))
                        category_word_list = []
                    category = label[0:-2]
                    merge_word_list.append(word)
                    merge_label_list.append(category)
                    category = ''
                elif label == self.default_label:
                    category_word_list.append(word)
                    merge_word_list.extend(category_word_list)
                    merge_label_list.extend([self.default_segment_label] * len(category_word_list))
                    category = ''
                    category_word_list = []
                else:
                    raise ValueError('Merge_label input exists invalid data.')
        if category and category_word_list:
            merge_word_list.append(''.join(category_word_list))
            merge_label_list.append(category)
        return merge_word_list, merge_label_list


    def prepare_datasets(self, raw_data_filename, test_percent, train_data_filename, test_data_filename):
        """
        Split dataset to train set, validation set, test set
        Store sets into datasets dir
        :param raw_data_filename:
        :param train_percent:
        :param val_percent:
        :param datasets_path:
        :return:
        """
        print('Prepare datasets ' + raw_data_filename)
        data_list = []
        with open(raw_data_filename, encoding='utf-8', mode='rt') as raw_data_file:
            for line in raw_data_file:
                line = line.strip('\n')
                if line:
                    data_list.append(line)
        random.shuffle(data_list)
        data_size = len(data_list)
        test_end_index = int(data_size * test_percent)

        train_data_list = data_list[test_end_index:]
        test_data_list = data_list[:test_end_index:]
        self.list_to_file(train_data_list, train_data_filename)
        self.list_to_file(test_data_list, test_data_filename)


    def count_vocabulary(self, data_filename):
        """
        Count word and label from file
        :param data_filename:
        :return: word count, label count and word count per label
        """
        words_count = {}
        labels_count = {}
        label_words_vocab = {}
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                word_list, label_list = self.split(line)
                for (word, label) in zip(word_list, label_list):
                    if word in words_count:
                        words_count[word] += 1
                    else:
                        words_count[word] = 1
                    if label in labels_count:
                        labels_count[label] += 1
                    else:
                        labels_count[label] = 1
                    if label in label_words_vocab:
                        if word not in label_words_vocab[label]:
                            label_words_vocab[label].append(word)
                    else:
                        label_words_vocab[label] = [word]
        return words_count, labels_count, label_words_vocab


    def list_to_file(self, data_list, data_filename):
        """
        Write list into file, one element per line
        :param data_list:
        :param data_filename:
        :return:
        """
        with open(data_filename, encoding='utf-8', mode='wt') as data_file:
            for data in data_list:
                if data:
                    data_file.write(data + '\n')


    def sort_vocabulary(self, vocab_count, vocab_filename, using_start_vocab=True):
        """
        Sort vocab from word and word count
        :param vocab_count:
        :param vocab_filename:
        :return: sorted vocab
        """
        vocab_list = sorted(vocab_count, key=vocab_count.get, reverse=True)
        if using_start_vocab:
            vocab_list = self._START_VOCAB + vocab_list
        self.list_to_file(vocab_list, vocab_filename)
        vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
        return vocab


    def create_vocabulary(self, data_filename, vocab_path, vocab_drop_limit):
        """
        Count and create vocabulary of word and label
        Store vocabulary into vocab dir
        :param data_filename:
        :param vocab_path:
        :return: word vocab, label vocab and word vocab per label
        """
        print('Creating vocabulary ' + data_filename)
        words_count, labels_count, _ = self.count_vocabulary(data_filename)
        words_count = {k: v for k, v in words_count.items() if v > vocab_drop_limit}
        words_vocab = self.sort_vocabulary(words_count, os.path.join(vocab_path, 'words_vocab.txt'))
        labels_vocab = self.sort_vocabulary(labels_count, os.path.join(vocab_path, 'labels_vocab.txt'), using_start_vocab=False)
        return words_vocab, labels_vocab


    def initialize_single_vocabulary(self, vocab_filename):
        """
        Restore vocabulary from vocab file
        :param vocab_filename:
        :return: vocab
        """
        if os.path.exists(vocab_filename):
            data_list = []
            with open(vocab_filename, encoding='utf-8', mode='rt') as vocab_file:
                for line in vocab_file:
                    line = line.strip()
                    if line:
                        data_list.append(line)
            vocab = dict([(x, y) for (y, x) in enumerate(data_list)])
            return vocab
        else:
            raise ValueError('Vocabulary file %s not found.', vocab_filename)


    def initialize_vocabulary(self, vocab_path):
        """
        Restore vocabulary of word and label from vocab file
        :param vocab_path:
        :return: word vocab and label vocab
        """
        print('Initialize vocabulary ' + vocab_path)
        words_vocab = self.initialize_single_vocabulary(os.path.join(vocab_path, 'words_vocab.txt'))
        labels_vocab = self.initialize_single_vocabulary(os.path.join(vocab_path, 'labels_vocab.txt'))
        return words_vocab, labels_vocab


    def get_vocabulary_size(self, vocab_filename):
        """
        Load num classes from labels vocab
        :return:
        """
        labels_vocab = self.initialize_single_vocabulary(vocab_filename)
        return len(labels_vocab)


    def load_default_label_id(self):
        """
        Init default word value and label value for read_and_decode padding
        :return:
        """
        labels_vocab = self.initialize_single_vocabulary(os.path.join(self.vocab_path, 'labels_vocab.txt'))
        try:
            default_label_id = labels_vocab[self.default_label]
        except:
            raise Exception('Can not find default_label : %s in labels vocab filename : ' % (self.default_label, labels_vocab))
        return default_label_id


    def word_to_id(self, word_list, vocab, default_id):
        """
        Transfer word to id
        :param word_list:
        :param vocab:
        :return: word ids
        """
        return [vocab.get(word, default_id) for word in word_list]


    def sentence_to_word_ids(self, sentence, words_vocab, labels_vocab):
        """
        Split sentence to words and labels, and transfer it to id
        :param sentence:
        :param words_vocab:
        :param labels_vocab:
        :return: word ids and label ids
        """
        word_list, label_list = self.split(sentence)
        word_ids = self.word_to_id(word_list, words_vocab, self._START_VOCAB_ID[3])
        label_ids = self.word_to_id(label_list, labels_vocab, labels_vocab[self.default_label])
        assert len(word_ids) == len(label_ids)
        return word_ids, label_ids


    def file_to_word_ids(self, data_filename, words_vocab, labels_vocab):
        """
        Transfer file to word ids
        :param data_filename:
        :param words_vocab:
        :param labels_vocab:
        :return: word ids list and label ids list
        """
        print('Tokenizing data in ' + data_filename)
        word_ids_list = []
        label_ids_list = []
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                word_ids, label_ids = self.sentence_to_word_ids(line, words_vocab, labels_vocab)
                if word_ids and label_ids:
                    word_ids_list.append(' '.join([str(tok) for tok in word_ids]))
                    label_ids_list.append(' '.join([str(tok) for tok in label_ids]))
        return word_ids_list, label_ids_list


    def align_word(self, words, fixed_size, padding_id):
        """
        Align length of words to a fixed size
        :param words:
        :param align_size:
        :return: padded words with align_size length
        """
        word_list = words.strip().split()
        words_count = len(word_list)
        if words_count < fixed_size:
            padding = ' '.join([padding_id for _ in range(fixed_size - words_count)])
            if words_count:
                return words + ' ' + padding
            else:
                return padding
        else:
            words_truncate = ' '.join(word_list[: fixed_size])
            return words_truncate


    def index_separator(self, words, start_index, separators):
        """
        Find the index of separators on words, index larger than start_index
        :param words:
        :param start_index:
        :param separators:
        :return:
        """
        index = start_index
        while index < len(words):
            if words[index] in separators:
                return index
            index += 1
        return -1


    def split_long_sentence(self, words, num_steps):
        """
        Split long sentence to short sentences
        :param words:
        :param num_steps:
        :return:
        """
        words = words.strip().split()
        if len(words) <= num_steps:
            return [' '.join(words)]

        words_list = []
        index = 0
        while index < len(words):
            separator_index = self.index_separator(words, index, ['。'])
            if separator_index == -1 or separator_index - index + 1 > num_steps:
                separator_index = self.index_separator(words, index, ['，', ',', '？', '?', '！', '!'])
            if separator_index == -1 or separator_index - index + 1 > num_steps:
                separator_index = min(index + num_steps - 1, len(words) - 1)
            words_list.append(' '.join(words[index: separator_index + 1]))
            index = separator_index + 1
        return words_list


    def judge_same_word(self, word, predict_word):
        """
        Judge two words is same
        :param word:
        :param predict_word:
        :return: if same return True, else return False
        """
        word_matrix = [[0 for _ in range(len(predict_word) + 1)] for _ in range(len(word) + 1)]
        common_max = 0
        for i in range(len(word)):
            for j in range(len(predict_word)):
                if word[i] == predict_word[j]:
                    word_matrix[i + 1][j + 1] = word_matrix[i][j] + 1
                    if word_matrix[i + 1][j + 1] > common_max:
                        common_max = word_matrix[i + 1][j + 1]
        min_len = len(word) if len(word) < len(predict_word) else len(predict_word)
        if common_max >= 6 or common_max == min_len:
            return True
        else:
            return False
