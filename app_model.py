# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/8/8
import json
import os
import time
from os.path import dirname, abspath

import torch
from flask import Flask, request
from sentence_transformers import SentenceTransformer

from src.common_utils.bert_utils import init_model, predict, load_dataset, build_iterator
from src.constants.common_constants import CLS2TEXT
from src.constants.config_constants import service

app = Flask(__name__)

# 加载业务分类模型
business_type_model, config = init_model('business_type_model')
business_type_model.load_state_dict(torch.load(config.save_path))
business_type_model.eval()

# 加载语义相似度计算模型
model = SentenceTransformer(os.path.join(dirname(abspath(__file__)),
                                         'src/model_training/question_similarity_model/model/sbert'))
model.encode(['测试用例'])


@app.route('/{}/{}'.format(service.model.context, service.model.business_type.interface),
           methods=['post'])
def get_business_type():
    """获得一组问题的类别列表
    :return:
    """
    if request.json is not None:
        questions = request.json.get(service.model.business_type.params, '[]')
    else:
        questions = request.values.get(service.model.business_type.params, '[]')
    _start_time = time.time()

    # 基于问题构造数据集和迭代器
    pred_data = load_dataset(config, questions)
    pred_iter = build_iterator(pred_data, config)

    # 调用分类方法
    _, pred_labels, pred_probs = predict(business_type_model, pred_iter)
    _class = [CLS2TEXT.get(e) for e in pred_labels.tolist()]
    print('***问题分类时间：%ss' % str(round(time.time() - _start_time, 2)))
    return json.dumps(_class, ensure_ascii=False)


@app.route('/{}/{}'.format(service.model.context, service.model.sentence_encode.interface),
           methods=['POST'])
def sbert_encode():
    """ 获取给定问题列表的向量列表
    :return:
    """
    if request.json is not None:
        sentences = request.json.get('questions', '[]')
    else:
        sentences = request.values.get('questions', '[]')

    if not isinstance(sentences, list):
        sentences = json.loads(sentences)
    vectors = model.encode(sentences).tolist()
    return json.dumps(vectors, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=False, port=8500)
