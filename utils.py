# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/23 10:15 上午
# @Author: wuchenglong

import collections


def read_vocab(vocab_file):
    lines = [line.strip() for line in open(vocab_file, "r").readlines()]
    lines = [line.split(" ") for line in lines]
    dictionary = dict(zip([line[0] for line in lines], [int(line[2]) for line in lines]))
    reversed_dictionary = dict(zip([int(line[2]) for line in lines], [line[0] for line in lines]))
    return dictionary, reversed_dictionary


def build_vocab(words,vocab_file):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(100000)
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    with open(vocab_file,"w") as f:
        for i, c in enumerate(count):
            f.write("{word} {count} {id}\n".format(
                word = c[0],count=c[1],id=i))
    return dictionary, reversed_dictionary

def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


def str_idx(corpus, dic,default):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, default))
        X.append(ints)
    return X


def len_check(line,min_line_length,max_line_length):
    return len(line.split()) >= min_line_length and len(line.split()) <= max_line_length
