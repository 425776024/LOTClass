#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 10:48 上午
# @File    : utils.py


def load_stop_words(path):
    stop_words = set()
    with open(path, mode='r', encoding='utf-8') as rf:
        for line in rf:
            stop_words.add(line.strip())
    return stop_words
