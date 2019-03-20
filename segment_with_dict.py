#!/usr/bin/python3
# -*- coding: utf-8 -*-

from math import log


class DagSegment(object):
    def __init__(self):
        self.extract_dict_freq = 30000000
        self.pre_segment_freq = 900000
        self.FREQ = {}
        self.total = 0
        base_dict_path = 'data/raw-data/base_dict.txt'
        extract_dict_path = 'data/raw-data/extract_dict.txt'
        self.gen_pfdict(base_dict_path)
        self.gen_pfdict(extract_dict_path)


    def gen_pfdict(self, dict_path):
        with open(dict_path, encoding='utf-8', mode='r') as data_file:
            for lineno, line in enumerate(data_file, 1):
                try:
                    words = line.strip().split('\t')
                    word = words[0]
                    if len(words) == 2:
                        freq = int(words[1])
                    elif len(words) == 1:
                        freq = self.extract_dict_freq
                    else:
                        raise ValueError('invalid dictionary entry in %s at Line %s: %s' % (dict_path, lineno, line))
                    self.FREQ[word] = freq
                    self.total += freq
                    # Similar to trie tree
                    for ch in range(len(word)):
                        wfrag = word[:ch + 1]
                        if wfrag not in self.FREQ:
                            self.FREQ[wfrag] = 0
                except ValueError:
                    raise ValueError('invalid dictionary entry in %s at Line %s: %s' % (dict_path, lineno, line))


    def add_pfdict(self, words, freq):
        lfreq = self.FREQ
        ltotal = self.total
        for word in words:
            if word not in lfreq:
                lfreq[word] = freq
                ltotal += freq
                # Similar to trie tree
                for ch in range(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
        return lfreq, ltotal


    def get_DAG(self, sentence, current_freq):
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in current_freq:
                if current_freq[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG


    def calc(self, sentence, DAG, route, current_total, out_dict_freq=1):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(current_total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or out_dict_freq) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])
            print('%d -------------' % idx)
            for x in DAG[idx]:
                if self.FREQ.get(sentence[idx:x + 1]):
                    temp4 = self.FREQ.get(sentence[idx:x + 1])
                temp1 = log(self.FREQ.get(sentence[idx:x + 1]) or out_dict_freq)
                temp2 = log(self.FREQ.get(sentence[idx:x + 1]) or out_dict_freq) - logtotal
                temp3 = log(self.FREQ.get(sentence[idx:x + 1]) or out_dict_freq) - logtotal + route[x + 1][0]
                print(str(temp3) + '    ' + str(x))


    def segment(self, sentence, pre_segments):
        current_freq, current_total = self.add_pfdict(pre_segments, self.pre_segment_freq)
        DAG = self.get_DAG(sentence, current_freq)
        route = {}
        self.calc(sentence, DAG, route, current_total)
        segments = []
        x = 0
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            word = sentence[x:y]
            segments.append(word)
            x = y
        return segments


def main():
    sentence = '全新英速亚旅行版在外观和内饰上均和此前曝光的三厢版基本一致'
    pre_segments = ['全新', '英速亚', '旅行版', '在', '外观和内饰', '上', '均', '和', '此前', '曝光', '的', '三厢版', '基本', '一致']
    # sentence = '相比14款取消了自动空调和后排出风口，后排座椅不能放倒，扶手箱太靠后，开长途手累'
    # pre_segments = ['相比', '14', '款', '取消', '了', '自动空调', '和', '后排', '出风口', '，', '后排', '座椅', '不能', '放倒', '，', '扶手箱', '太', '靠后', '，', '开', '长途', '手累']
    dagSegment = DagSegment()
    segments = dagSegment.segment(sentence, pre_segments)
    print(segments)


if __name__ == '__main__':
    main()
