# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/7/4
import os
import sys
import time
import traceback
from typing import List

import numpy as np
import psutil
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from src.common_utils.logstash_utils import Logger
from src.common_utils.common_utils import sbert_encode, get_business_type, get_es_hosts
from src.common_utils.es_utils import query_existing_questions
from src.constants.common_constants import VECTORIZE_BATCH, DUMP_ES_BATCH, AVAILABLE_CLS, QUESTION_MAX_LEN
from src.constants.config_constants import es
from src.constants.index_constants import interact_qa, ai_qa, EMPTY_COL_VAL

logger = Logger(__name__)

es_sim_client = Elasticsearch(get_es_hosts(), timeout=5000)

call_dump_es_count = 0


def query_from_es() -> List:
    """ 查询过去N天互动问答的问题
    :return:
    """
    query_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            interact_qa.REPLY_FLAG: 1
                        }
                    },
                    {
                        "range": {
                            interact_qa.QUESTION_TIME: {
                                "gte": "now-1000d",
                                "lte": "now"
                            }
                        }
                    }
                ]
            }
        },
        "_source": [
            interact_qa.ID,
            interact_qa.QUESTION,
            interact_qa.QUESTION_TIME,
            interact_qa.SW_INDUSTRY_NAME,
            interact_qa.ZJH_INDUSTRY_NAME,
            interact_qa.PLATE,
        ]
    }

    # 获取全部原始数据
    query_res = query_existing_questions(es_sim_client, es.index.interact_qa, query_body)
    query_res = [q for q in query_res if len(q.get(interact_qa.QUESTION)) <= QUESTION_MAX_LEN]
    return query_res


def convert_to_vectors(question_lst: List) -> np.ndarray:
    """批量将问题转换为语义向量
    :param question_lst:
    :return:
    """
    # 切片调用模型服务
    start, end = 0, VECTORIZE_BATCH
    vectors = None
    # 如果问答对数小于batch数，则直接转换向量
    if len(question_lst) < end:
        vectors = sbert_encode(question_lst)
    else:
        while end < len(question_lst):
            _vectors = sbert_encode(question_lst[start: end])
            if start == 0:
                vectors = _vectors
            else:
                vectors = np.vstack((vectors, _vectors))
            start = end
            end += VECTORIZE_BATCH
        _vectors = sbert_encode(question_lst[start:])
        vectors = np.vstack((vectors, _vectors))
    return vectors


def dump_to_es(question_objs, classes, embeddings):
    """ 将互动问答类别和语义向量存入es
    :param question_objs:
    :param classes:
    :param embeddings:
    :return:
    """
    actions = []
    for _obj, _cls, _vector in zip(question_objs, classes, embeddings):
        if _cls in AVAILABLE_CLS:
            action = {
                "_index": es.index.intelligent_interact_qa,
                "_id": _obj.get(interact_qa.ID),
                "_source": {
                    ai_qa.ID: _obj.get(interact_qa.ID),
                    ai_qa.QUESTION: _obj.get(interact_qa.QUESTION),
                    ai_qa.QUESTION_TIME: _obj.get(interact_qa.QUESTION_TIME),
                    ai_qa.CLS: _cls,
                    ai_qa.PLATE: _obj.get(interact_qa.PLATE) if _obj.get(interact_qa.PLATE) else EMPTY_COL_VAL,
                    ai_qa.VECTOR: _vector.tolist()
                }
            }

            # 获取证监会和申万行业
            sw_industry_name = _obj.get(interact_qa.SW_INDUSTRY_NAME) if _obj.get(
                interact_qa.SW_INDUSTRY_NAME) else EMPTY_COL_VAL
            zjh_industry_name = _obj.get(interact_qa.ZJH_INDUSTRY_NAME) if _obj.get(
                interact_qa.ZJH_INDUSTRY_NAME) else EMPTY_COL_VAL

            action.get('_source').update({ai_qa.SW_INDUSTRY_NAME: sw_industry_name})
            action.get('_source').update({ai_qa.ZJH_INDUSTRY_NAME: zjh_industry_name})

            actions.append(action)

    try:
        global call_dump_es_count
        if call_dump_es_count == 0:
            es_sim_client.indices.delete(index=es.index.intelligent_interact_qa, ignore=[400, 404])
            es_sim_client.indices.create(index=es.index.intelligent_interact_qa, body=ai_qa.MAPPING)
        success, _ = bulk(es_sim_client, actions, index=es.index.intelligent_interact_qa, raise_on_error=True)
        call_dump_es_count += 1
        logger.info('共计导入 %s 个问答对' % success)
    except Exception:
        logger.error(traceback.format_exc())


def construct_es_data():
    """构建intelligent_interact_qa索引，并将互动问答索引数据导入该索引
    :return:
    """
    # 从es中查询问题及相关字段
    questions = query_from_es()

    segment_num = int(len(questions) / DUMP_ES_BATCH)

    for i in range(segment_num):
        try:
            _questions = questions[i * DUMP_ES_BATCH: (i + 1) * DUMP_ES_BATCH]

            # 将问题向量化
            logger.info('向量化中...')
            pred_data = [d.get(interact_qa.QUESTION) for d in _questions]
            vectors = convert_to_vectors(pred_data)
            # 预测问题类别
            logger.info('分类中...')
            pred_labels = get_business_type(pred_data)
            # 存入es
            logger.info('存入ES中...')
            dump_to_es(_questions, pred_labels, vectors)
        except Exception:
            logger.error(traceback.format_exc())
        finally:
            del vectors
            logger.info('内存使用：%sG' %
                        round(psutil.Process(os.getpid()).memory_info().rss / pow(1024, 3), 2))

    # 将剩余问题存入es中
    _questions = questions[segment_num * DUMP_ES_BATCH:]

    logger.info('向量化中...')
    pred_data = [d.get(interact_qa.QUESTION) for d in _questions]
    vectors = convert_to_vectors(pred_data)
    # 预测问题类别
    logger.info('分类中...')
    pred_labels = get_business_type(pred_data)
    # 存入es
    logger.info('存入ES中...')
    dump_to_es(_questions, pred_labels, vectors)
    del vectors
    logger.info('内存使用：%sG' %
                round(psutil.Process(os.getpid()).memory_info().rss / pow(1024, 3), 2))


def update_vectors_in_mini_batch(actions_tmp):
    """ 一个mini batch的向量更新
    :param actions_tmp:
    :return:
    """
    try:
        # 取出问题列表，并调用sentence-bert获取新的编码
        questions = [elem.get('doc').get(ai_qa.QUESTION) for elem in actions_tmp]
        q_vec = sbert_encode([q for q in questions]).tolist()

        # 更新已有向量
        for i, _action in enumerate(actions_tmp):
            doc_obj = _action.get('doc')
            doc_obj[ai_qa.VECTOR] = q_vec[i]
            doc_obj.pop(ai_qa.QUESTION)
            _action['doc'] = doc_obj
            actions_tmp[i] = _action

        return actions_tmp
    except Exception:
        return []


def update_vectors():
    """更新intelligent索引中的问题向量
    :return:
    """
    start_time = time.time()
    es_client = Elasticsearch(hosts=get_es_hosts(), timeout=5000)

    # 查询现有问答对
    query_body = {
        "query": {
            "match_all": {}
        }
    }
    pairs = query_existing_questions(es_cli=es_client,
                                     es_index=es.index.intelligent_interact_qa,
                                     query_body=query_body)

    actions = []
    actions_tmp = []
    for count, pair in enumerate(pairs):
        if count != 0 and count % 100 == 0:
            # 更新一个batch的操作
            actions.extend(update_vectors_in_mini_batch(actions_tmp))
            actions_tmp = []
            logger.info('%s / %s' % (count, len(pairs)))

        # 取出问答对中的问题
        question = pair.get(ai_qa.QUESTION)

        try:
            action = {
                '_op_type': 'update',
                '_index': es.index.intelligent_interact_qa,
                '_id': pair.get(interact_qa.ID),
                'doc': {ai_qa.QUESTION: question}
            }
            actions_tmp.append(action)

        except Exception:
            logger.error('更新字段异常： %s' % question)
            continue

    # 更新剩余的数据操作
    actions.extend(update_vectors_in_mini_batch(actions_tmp))
    logger.info('%s / %s' % (len(pairs), len(pairs)))

    # 批量更新问题向量
    success, _ = bulk(es_client, actions, index=es.index.intelligent_interact_qa, raise_on_error=True)
    logger.info('更新问题向量共耗时%s s。' % round(time.time() - start_time, 2))


if __name__ == '__main__':
    # 构建数据集标志位
    update_vector = False
    if not update_vector:
        construct_es_data()
    elif update_vector:
        update_vectors()
