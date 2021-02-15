#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 下午4:09
# @File    : logers.py
import os
from loguru import logger


class LOGS:
    log = logger

    @classmethod
    def init(cls, log_file: str):
        if os.path.exists(log_file):
            os.remove(log_file)
        cls.log.add(log_file, rotation="00:00", retention='7 days')


