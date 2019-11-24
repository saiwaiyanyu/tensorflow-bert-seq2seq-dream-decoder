# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/9 10:49 AM
# @Author: wuchenglong

from bert.tokenization import  FullTokenizer as BertFullTokenizer


class FullTokenizer(BertFullTokenizer):
    """Runs end-to-end tokenziation."""
    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab.get(item,100))
    return output