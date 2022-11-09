# -*- coding: utf-8 -*-
# @Desc   : 构建业务领域分类模型数据集
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/6/30
import os
import re

from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split

from src.common_utils.common_utils import get_es_hosts
from src.common_utils.es_utils import query_existing_questions
from src.constants.common_constants import KW_PATTERNS, CLS_LST, QUESTION_MAX_LEN
from src.constants.config_constants import es
from src.constants.index_constants import interact_qa

# 其他类别编号及每个类别样本最大数
other_cls, sample_num = 12, 4000

# 伪标注数据用于训练的模型
corpus_for_model = 'business_type_model'

# 数据集基础路径
data_set_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  f'model_training/{corpus_for_model}/data')

# 加载手动标注样本词典
with open(os.path.join(data_set_base_path, 'annotation_list.txt'), 'r') as annotate_file:
    annotate_dict = [s.strip().split('$$$') for s in annotate_file.readlines()]
    annotate_dict = dict(zip([s[0] for s in annotate_dict], [s[1] for s in annotate_dict]))


def construct_data_set(ds_type, X, Y):
    """按规定格式构造数据集
    :param ds_type:
    :param X:
    :param Y:
    :return:
    """
    with open(os.path.join(data_set_base_path, f'{ds_type}.txt'), 'w') as ds_file:
        output_samples = []
        for x, y in zip(X, Y):
            manual_y = annotate_dict.get(x)
            # 如果样本标注被人工纠正，采用纠正后标注
            if manual_y:
                y = manual_y
            output_samples.append('{}\t{}\n'.format(x, y))

        if ds_type in ['train', 'dev']:
            # 将人工标注数据加入训练集和开发集
            output_samples.extend(['{}\t{}\n'.format(x, y) for x, y in annotate_dict.items()])

        ds_file.writelines(list(set(output_samples)))


if __name__ == '__main__':
    samples = []
    labels = []

    query_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match_all": {}
                    },
                    {
                        "range": {
                            interact_qa.ANSWER_TIME: {
                                "gte": "now-500d",
                                "lte": "now"
                            }
                        }
                    }
                ]
            }
        },
        "_source": [interact_qa.QUESTION]
    }

    # 获取全部原始数据
    es_client = Elasticsearch(get_es_hosts(), timeout=5000)
    query_res = query_existing_questions(es_client, es.index.interact_qa, query_body)
    questions = [q.get(interact_qa.QUESTION) for q in query_res]

    # todo: 用于验证模型分类“其他”类别的占比
    with open(os.path.join(data_set_base_path, 'test_corpus.txt'), 'w') as corpus_file:
        corpus_file.writelines(['{}\n'.format(re.sub('\n|\r|\t', ' ', q)) for q in questions])

    for q in questions:
        q = re.sub('\n|\r|\t', ' ', q)
        if len(q) > QUESTION_MAX_LEN:
            continue
        is_matched = False
        for k, v in KW_PATTERNS.items():
            if re.search(k, q):
                is_matched = True
                if labels.count(v) < sample_num:
                    samples.append(q)
                    labels.append(v)
                    break
        if not is_matched and labels.count(other_cls) < sample_num:
            samples.append(q)
            labels.append(other_cls)

    X_train, X_test, y_train, y_test = train_test_split(samples, labels,
                                                        train_size=0.8, random_state=2,
                                                        shuffle=True, stratify=labels)

    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test,
                                                    train_size=0.5, random_state=2,
                                                    shuffle=True, stratify=y_test)

    # 生成训练集
    construct_data_set('train', X_train, y_train)
    # 生成验证集
    construct_data_set('dev', X_dev, y_dev)
    # 生成测试集
    construct_data_set('test', X_test, y_test)

    # 生成类别文件
    with open(os.path.join(data_set_base_path, 'class.txt'), 'w') as cls_file:
        cls_file.writelines(['{}\n'.format(c) for c in CLS_LST])
