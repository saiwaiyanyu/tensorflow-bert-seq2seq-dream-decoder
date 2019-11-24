# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/23 10:15 上午
# @Author: wuchenglong


import copy
def pad_data(data):
    c_data = copy.deepcopy(data)
    max_x_length = max([len(i[0]) for i in c_data])
    max_y_length = max([len(i[1]) for i in c_data])
    # print("max_x_length : {} ,max_y_length : {}".format( max_x_length,max_y_length))
    padded_data = []
    for i in c_data:
        tokens, tag_ids, inputs_ids, segment_ids, input_mask = i
        tag_ids = tag_ids + (max_y_length - len(tag_ids)) * [0]
        inputs_ids = inputs_ids + (max_x_length - len(inputs_ids)) * [0]
        segment_ids = segment_ids + (max_x_length - len(segment_ids)) * [0]
        input_mask = input_mask + (max_x_length - len(input_mask)) * [0]
        assert len(inputs_ids) == len(segment_ids) == len(input_mask)
        padded_data.append(
            [tokens, tag_ids, inputs_ids, segment_ids, input_mask]
        )
    return padded_data


def len_check(line,min_line_length,max_line_length):
    return len(line.split()) >= min_line_length and len(line.split()) <= max_line_length
