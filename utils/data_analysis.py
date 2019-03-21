#!/usr/bin/python3
# -*- coding: utf-8 -*-

from utils.data_utils import DataUtils


class DataAnalysis(object):

    def __init__(self):
        self.data_utils = DataUtils()


    def word_analysis(self, data_filename):
        """
        Count word frequency
        :param data_filename:
        :return:
        """
        words_count = 0
        words_count_map = dict()
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                words, _ = self.data_utils.split(line)
                words_count += len(words)
                for word in words:
                    if word in words_count_map:
                        words_count_map[word] += 1
                    else:
                        words_count_map[word] = 1
        words_type_count = len(words_count_map)
        return words_count, words_type_count


    def length_analysis(self, data_filename):
        """
        Count sentence length
        :param data_filename:
        :return:
        """
        sentences_count = 0
        max_length = 0
        length_count_map = dict()
        with open(data_filename, encoding='utf-8', mode='rt') as data_file:
            for line in data_file:
                words, _ = self.data_utils.split(line)
                length = len(''.join(words))
                # length = len(''.join(line.strip().split()))
                if length in length_count_map:
                    length_count_map[length] += 1
                else:
                    length_count_map[length] = 1
                if length > max_length:
                    max_length = length
                sentences_count += 1

        if sentences_count == 0:
            return dict()

        statistic_result = dict()
        accumulative_count = 0
        for i in range(max_length + 1):
            if i in length_count_map:
                accumulative_count += length_count_map[i]
            if i != 0 and (i % 50 == 0 or i == max_length):
                statistic_result[i] = '%.2f' % (accumulative_count / sentences_count * 100)
        return statistic_result


def main():
    data_analysis = DataAnalysis()

    data_filename = 'data/raw-data/data_demo.txt'
    result = data_analysis.length_analysis(data_filename)
    print(result)

    words_count, words_type_count = data_analysis.word_analysis(data_filename)
    print('All words count : %d, Word type count : %d' % (words_count, words_type_count))


if __name__ == '__main__':
    main()
