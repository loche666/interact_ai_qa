# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/7/4
import os
from importlib import import_module
from typing import List

import numpy as np
import torch

from src.constants.common_constants import CLS
from src.constants.config_constants import cuda
from src.model_training.business_type_model.utils import DatasetIterater

os.environ[cuda.device.key] = cuda.device.val


class DatasetIterator(object):
    """数据集迭代器"""

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def init_model(model_name):
    """设置模型配置，加载模型
    :param model_name:
    :return:
    """
    x = import_module('src.models.bert')
    config = x.ModelConfig(model_name)
    model = x.Model(config).to(config.device)
    return model, config


def predict(model, test_iter):
    """
    :param model:
    :param test_iter:
    :return:
    """
    pred_labels_all = np.array([], dtype=int)
    pred_probs_all = np.array([], dtype=float)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in test_iter:
            pred_labels, pred_probs = model.predict(texts)
            labels = labels.data.cpu().numpy()

            labels_all = np.append(labels_all, labels)
            pred_labels_all = np.append(pred_labels_all, pred_labels)
            pred_probs_all = np.append(pred_probs_all, pred_probs)

    return labels_all, pred_labels_all, pred_probs_all


def build_iterator(dataset, config):
    """
    :param dataset:
    :param config:
    :return:
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def load_dataset(config, pred_data: List):
    """
    :param config:
    :param pred_data:
    :return:
    """
    contents = []
    for line in pred_data:
        lin = line.strip()
        if not lin:
            continue
        content, label = lin.strip(), 0
        token = config.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        pad_size = config.pad_size

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, int(label), seq_len, mask))
    return contents
