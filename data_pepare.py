# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/21 9:24 下午
# @Author: wuchenglong

import json,re



lines = [line.strip() for line in open("data.csv","r").readlines()] + [line.strip() for line in open("data2.csv","r").readlines()]
lines = [(json.loads(line)["url_2"], json.loads(line)["detail"])  for line in lines]

sample = []
for url,line in lines:
    # print(url)
    # print("=" * 20)
    # print(line)
    line = line.replace("\u3000","").split("原版周公")[0]
    # print(line)
    for elem in re.split("[\n。]",line):
        if any([ elem.__contains__(key) for key in ["梦到","梦见"]])\
                and all([ not elem.__contains__(key) for key  in ["：","（","）","梦境解说","明确地指出","案件分析","梦境解析","心理分析","梦境解说","详细解说","梦境描述","案例分析","是什么意思呢"]])\
            :
            res = re.search('(?P<dream>(梦见|梦到)[^,，]+)[,，](?P<decode>.+)$', elem)
            if res:
                result_dict = res.groupdict()
                if re.search("(暗示|预示|表示你)", result_dict["dream"] ):

                    result_dict["decode"] = "预示" + re.split("(暗示|预示|表示你)", result_dict["dream"] )[1] +result_dict["decode"]

                    result_dict["dream"] = re.split("(暗示|预示|表示你)", result_dict["dream"] )[0]

                if len(result_dict["decode"]) > 70:
                    # print(elem)
                    continue
                result_dict["decode"] = re.sub("[。，,.]?$","。",result_dict["decode"] )

                # result_dict["decode"] = re.sub("^(代表|预示|暗示|表明|那表示|意味着|表示)", "", result_dict["decode"])
                # result_dict["dream"] = re.sub("^(梦见|梦到)", "", result_dict["dream"])
                sample.append(result_dict)
            else:
                pass
        else:
            pass

with open("data/data_init.csv","w") as f:
    for elem in sample:
        f.write(json.dumps(elem,ensure_ascii=False)+"\n")


import collections
lines = [line.strip() for line in open("data/data_init.csv", "r").readlines()]
lines = [ (json.loads(line)["dream"],json.loads(line)["decode"]) for line in lines]
unique_line = {}

for question,answer in lines:
    if question in unique_line:
        if len(answer) > len(unique_line[question]):
            unique_line[question]=answer
    else:
        unique_line[question] = answer

with open("data/data.csv","w") as f:
    for  question,answer in unique_line.items():
        f.write(json.dumps({
            "dream":question,"decode":answer
        },ensure_ascii=False)+"\n")



