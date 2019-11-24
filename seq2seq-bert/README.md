
# tf-bert-seq2seq-dream-decoder
![](media/15744968371036.jpg)


`tensorflow-bert` 周公解梦。通过 `tensorflow bert seq2seq` 实现一个梦境解析模型，说出你的`梦境(dream)`，模型自动解析`decode`你梦境的征兆，用科学的视角看待玄学。 \^_^

## 依赖

    python >= 3.6
    tensorflow 1.14.0
    bert


## 下载bert预训练模型

    $ wget -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    $ unzip chinese_L-12_H-768_A-12.zip 
    

## 下载bert源代码
下载 [**bert**](https://github.com/google-research/bert) 放入项目目录**bert**下，

    $ git clone https://github.com/google-research/bert.git
    
## 数据样例

    {
        "dream": "梦到买自行车",
        "decode": "正在对某件事情做出决定，可能会带来不好的后果。"
    }
    {
        "dream": "梦到买筐子",
        "decode": "预示着近期生活上可能会有大的开销。"
    }
    

**33000+** 梦境解析[**样本**](data/data.csv)。

## 运行

### 训练

    $ python3 model.py --task=train \
        --is_training=True \
        --epoch=100 \
        --size_layer=256 \
        --bert_config=chinese_L-12_H-768_A-12/bert_config.json \
        --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
        --num_layers=2 \
        --learning_rate=0.001 \
        --batch_size=16 \
        --checkpoint_dir=result





![](media/15744775485612.jpg)

### 预测

    $ python3 model.py --task=predict \
        --is_training=False \
        --epoch=100 \
        --size_layer=256 \
        --bert_config=chinese_L-12_H-768_A-12/bert_config.json \
        --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
        --num_layers=2 \
        --learning_rate=0.001 \
        --batch_size=16 \
        --checkpoint_dir=result


**Just For Fun!!**

