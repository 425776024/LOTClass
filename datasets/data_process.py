#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 3:16 下午
# @File    : data_process.py

import pandas as pd
from config.configs_interface import configs
import os
from collections import Counter
from src.data_utils.train_test_split import train_test_split
from src.logers import LOGS
from transformers import BertTokenizer

stop_words = os.path.join(configs.data.DATASET, configs.data.stop_words)
LABEL_NAME_FILE = os.path.join(configs.data.DATASET, configs.data.LABEL_NAME_FILE)
TRAIN_CORPUS = os.path.join(configs.data.DATASET, configs.data.TRAIN_CORPUS)
TEST_CORPUS = os.path.join(configs.data.DATASET, configs.data.TEST_CORPUS)
TRAIN_LABEL = os.path.join(configs.data.DATASET, configs.data.TRAIN_LABEL)
TEST_LABEL = os.path.join(configs.data.DATASET, configs.data.TEST_LABEL)


def get_data(min_label_count=0):
    '''
    更具 min_label_count 进行过滤，（不分析特点类别，全部混合训练）
    :return: data_sentences, data_labels, label_to_index, index_to_label
    '''
    train_df = pd.read_csv('data.csv')
    LOGS.log.debug(f"Train set shape:{train_df.shape}")
    label_count = train_df['label'].value_counts()
    # data_sentences = train_df['text'].values
    LOGS.log.debug(f'原始类别数据：{label_count}')
    # 只分析这些label

    clear_labels = []
    for li in label_count.index:
        if label_count[li] < min_label_count:
            clear_labels.append(li)
    print('这些类别过滤删除：', clear_labels)

    df_clear = train_df[~train_df['label'].isin(clear_labels)]

    LOGS.log.debug(f"Train set shape:{df_clear.shape}")
    clear_label_count = df_clear['label'].value_counts()
    LOGS.log.debug(f'过滤后数据：{clear_label_count}')

    # 所有数据
    data_sentences = df_clear['text'].values.tolist()
    data_labels = df_clear['label'].values.tolist()
    labels = clear_label_count.keys()

    label_to_index = {
        str(k): i for i, k in enumerate(labels)
    }
    index_to_label = {
        i: str(k) for i, k in enumerate(labels)
    }
    data_labels = [label_to_index[i] for i in data_labels]

    assert len(data_sentences) == len(data_labels)
    label_counter = Counter(data_labels)
    count_vales = label_counter.values()
    # max_count = max(count_vales)
    min_count = min(count_vales)
    label_weight = []
    for label_i in range(len(label_to_index)):
        wi = min_count * 1.0 / label_counter[label_i]
        label_weight.append(wi)

    assert len(data_sentences) == len(data_labels)
    return data_sentences, data_labels, labels, label_to_index, index_to_label, label_weight


def write(datas, path):
    with open(path, mode='w', encoding='utf-8') as wf:
        if isinstance(datas, list):
            for di in datas:
                di = str(di)
                wf.write(di + '\n')
        if isinstance(datas, dict):
            for di in datas:
                di = str(datas[di])
                wf.write(di + '\n')


data_sentences, data_labels, labels, label_to_index, index_to_label, label_weight = get_data()

train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    data_sentences,
    data_labels,
    test_size=configs.data.test_date_rate,
    random_state=1,
    banance=None)

write(index_to_label, path=LABEL_NAME_FILE)
write(train_inputs, path=TRAIN_CORPUS)
write(test_inputs, path=TEST_CORPUS)
write(train_targets, path=TRAIN_LABEL)
write(test_targets, path=TEST_LABEL)
