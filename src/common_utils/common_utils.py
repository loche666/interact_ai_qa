# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/8/12
import json
import traceback
from typing import Dict, List

import numpy as np
import requests
from elasticsearch import Elasticsearch
from requests import ReadTimeout
from sklearn.metrics.pairwise import cosine_similarity

from src.common_utils.es_utils import query_existing_questions
from src.common_utils.logstash_utils import Logger
from src.constants.common_constants import CANDIDATE_NUM
from src.constants.config_constants import service, es, consul as consul_param, url
from src.constants.index_constants import interact_qa, ai_qa

logger = Logger(__name__)


def get_es_hosts() -> List:
    """ 获取es的host列表
    :return:
    """

    return [{'host': host, 'port': es.port} for host in es.ip.split(' ')]


def sbert_encode(questions: List) -> np.ndarray:
    """ 基于rest请求获取问题向量
    :param questions:
    :return:
    """
    # 初始化问题向量
    vector = []
    post_data = {service.model.sentence_encode.params: questions}

    sbert_service_url = '{}:{}/{}/{}'.format(service.model.url, service.model.port,
                                             service.model.context, service.model.sentence_encode.interface)

    try:
        result = requests.post(sbert_service_url, json=post_data, timeout=60)
    except ReadTimeout:
        logger.error('请求语义向量服务超时。%s' % traceback.format_exc())
        return np.asarray(vector)

    if result.status_code == 200:
        vector = json.loads(bytes.decode(result.content))
    else:
        logger.error('请求语义向量服务失败，状态码：%s。%s' % (result.status_code, traceback.format_exc()))
    return np.asarray(vector)


def get_business_type(questions: List) -> List:
    """获取问题所属业务分类
    :param questions:
    :return:
    """
    b_type = []
    business_type_service_url = '{}:{}/{}/{}'.format(service.model.url, service.model.port,
                                                     service.model.context, service.model.business_type.interface)
    try:
        result = requests.post(business_type_service_url,
                               json={service.model.business_type.params: questions},
                               headers={'Content-Type': 'application/json'},
                               timeout=60)
    except ReadTimeout:
        logger.error('请求问题类型超时。%s' % traceback.format_exc())
        return b_type
    if result.status_code == 200:
        b_type = json.loads(bytes.decode(result.content))
    else:
        logger.error('请求Business_type_model模型失败，状态码：%s。%s' % (result.status_code, traceback.format_exc()))
    return b_type


def construct_answers(question, sw_industry, zjh_industry, plate, answers) -> Dict:
    """ 构建返回问题对应的答案
    :param question:
    :param plate:
    :param zjh_industry:
    :param sw_industry:
    :param answers:
    :return:
    """
    return {
        service.app.param.question: question,
        service.app.param.sw_industry_name: sw_industry,
        service.app.param.zjh_industry_name: zjh_industry,
        service.app.param.plate: plate,
        service.app.param.answer: answers
    }


def query_interact_data(es_client: Elasticsearch, index: str, ids) -> List:
    """基于互动问答id，查询互动问答索引，获取相似问题的相关信息
    :param es_client:
    :param index:
    :param ids:
    :return:
    """
    query_id_lst = []
    for _id in ids:
        query_id_lst.append(
            {
                "match_phrase": {
                    interact_qa.ID: _id
                }
            }
        )
    query_body = {
        "query": {
            "bool": {
                "should": query_id_lst
            }
        },
        "_source": [
            interact_qa.ID,
            interact_qa.QUESTION,
            interact_qa.QUESTION_TYPE,
            interact_qa.QUESTION_TIME,
            interact_qa.COMPANY_CODE,
            interact_qa.COMPANY_SORT_NAME,
            interact_qa.ANSWER_LIST,
            interact_qa.SOURCE
        ]
    }

    # 获取全部原始数据
    query_res = query_existing_questions(es_client, index, query_body,
                                         scroll_maintain_time='10s', scroll_size=100)
    return query_res


def query_candidates(es_client: Elasticsearch, index: str, _class: str, sw_industry: List,
                     zjh_industry: List, plate: List, unmatched_pairs) -> List:
    """基于问题类别、问题公司所属行业检索候选历史问题
    :param es_client:
    :param index: 互动问答智能匹配索引
    :param _class: 问题所属类别
    :param sw_industry: 提问公司所属申万二级行业
    :param plate: 用户所选板块
    :param zjh_industry: 提问公司所属证监会行业
    :param unmatched_pairs: 用户标注不适用的历史问题列表
    :return:
    """

    def construct_should_block(key, terms):
        """ 构建行业查询语句列表
        :param key:
        :param terms:
        :return:
        """
        should_list = []
        for term in terms:
            should_list.append({
                "term": {
                    key: term
                }
            })
        return {
            "bool": {
                "should": should_list
            }
        }

    def construct_must_not_block(key, terms):
        """ 构建不适用匹配列表
        :param key:
        :param terms:
        :return:
        """
        must_not_list = []
        for term in terms:
            must_not_list.append({
                "term": {
                    key: term
                }
            })
        return {
            "bool": {
                "must_not": must_not_list
            }
        }

    query_body = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "term": {
                            ai_qa.CLS: _class
                        }
                    }
                ]
            }
        },
        "_source": [ai_qa.ID, ai_qa.VECTOR]
    }

    # 如果申万行业列表不为空
    if sw_industry:
        query_body['query']['bool']['filter'].append(construct_should_block(ai_qa.SW_INDUSTRY_NAME, sw_industry))

    # 如果证监会行业列表不为空
    if sw_industry:
        query_body['query']['bool']['filter'].append(construct_should_block(ai_qa.ZJH_INDUSTRY_NAME, zjh_industry))

    # 如果板块列表不为空
    if plate:
        query_body['query']['bool']['filter'].append(construct_should_block(ai_qa.PLATE, plate))

    # 如果不适用列表不为空
    if unmatched_pairs:
        query_body['query']['bool']['filter'].append(construct_must_not_block(ai_qa.ID, unmatched_pairs))

    # 获取全部原始数据
    query_res = query_existing_questions(es_client, index, query_body, scroll_maintain_time='10s', scroll_size=1000)
    return query_res


def calculate_cosine_similarity(question_embedding: np.ndarray, results: List) -> List:
    """ 在召回的问答对列表中增加cosine相似度字段
    :param question_embedding:
    :param results:
    :return:
    """

    def _calculate_cosine_similarity(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
        """ 计算两个向量的cosine相似度
        :param vec_1:
        :param vec_2:
        :return:
        """
        similarity = cosine_similarity(vec_1, vec_2)[0][0]
        return (similarity + 1) / 2

    new_results = []
    for res in results:
        vec = np.asarray(res.get('vector')).reshape(1, -1)
        cos = _calculate_cosine_similarity(question_embedding.reshape(1, -1), vec)
        res['cosine'] = cos
        # 减少系统开销
        res.pop('vector')
        new_results.append(res)
    return sorted(new_results, key=lambda x: x.get('cosine'), reverse=True)


def refine_res(id_2_score: List, pairs: List) -> List:
    """ 去重查询结果相同的问题，并按语义相似度得分排序
    :param id_2_score:
    :param pairs:
    :return:
    """
    # 补充语义相似度得分，用于最终排序
    id_2_score_dict = dict(zip([e.get(interact_qa.ID) for e in id_2_score],
                               [round(e.get('cosine'), 3) for e in id_2_score]))
    question_2_obj = {}
    for p in pairs:
        # 对问题内容一样的对象进行去重，并按得分、时间顺序排序
        if question_2_obj.get(p.get(interact_qa.QUESTION)):
            continue
        else:
            p.update({'cosine': id_2_score_dict.get(p.get(interact_qa.ID))})
            question_2_obj[p.get(interact_qa.QUESTION)] = p
    res = sorted(list(question_2_obj.values()), key=lambda x: (x.get('cosine'), x.get(interact_qa.QUESTION_TIME)),
                 reverse=True)[: CANDIDATE_NUM]

    for i, p in enumerate(res):
        p.update({'page_num': int(i / 10)})
    return res


def register_service():
    """ 在Consul中注册服务
    :return:
    """
    import consul
    from src.common_utils.consul_client import ConsulClient
    consul_ip = consul_param.host
    consul_port = consul_param.port
    server_name = consul_param.service_name
    local_ip = consul_param.local_host
    local_port = consul_param.local_port
    interval = consul_param.interval
    token = consul_param.token

    logger.info('注册服务：%s' % server_name)
    client = ConsulClient(consul_ip, consul_port, token)

    service_id = '{}{}:{}'.format(server_name, local_ip, local_port)
    http_check = url.HEALTH_CHECK_URL.format(local_ip, local_port)

    check = consul.Check.http(http_check, interval)
    client.register(server_name, service_id, local_ip, int(local_port), '', interval, check)
    logger.info('注册服务 %s 成功' % server_name)
