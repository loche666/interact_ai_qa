# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/8/12
import json
import platform
import time
import traceback
from typing import List

from elasticsearch import Elasticsearch
from flask import Flask, request

from src.common_utils.common_utils import calculate_cosine_similarity, query_candidates, query_interact_data, \
    construct_answers, get_business_type, sbert_encode, get_es_hosts, refine_res, register_service
from src.common_utils.db_utils import query_unmatched_pairs
from src.common_utils.logstash_utils import Logger
from src.constants.common_constants import DEFAULT_ANSWER, CANDIDATE_NUM, AVAILABLE_CLS, QUESTION_MAX_LEN
from src.constants.config_constants import es, service, url
from src.constants.http_constants import http_obj
from src.constants.index_constants import ai_qa

app = Flask(__name__)

logger = Logger(__name__)

es_client = Elasticsearch(get_es_hosts(), timeout=5000)

# 注册中心
if platform.system().lower() not in ['windows', 'darwin']:
    try:
        register_service()
    except Exception:
        logger.error('注册失败 %s' % traceback.format_exc())


@app.route('/{}/{}'.format(service.app.context, service.app.get_answers), methods=['post'])
def get_answers():
    """获取互动问答问题答案"""

    if request.json is not None:
        questions = request.json.get(service.app.param.questions, '[]')
    else:
        questions = request.values.get(service.app.param.questions, '[]')

    # 获取参数
    try:
        question_lst: List[str] = [q.get(service.app.param.question) for q in questions]
        question_ids: List[str] = [q.get(service.app.param.question_id) for q in questions]
        company_codes: List[str] = [q.get(service.app.param.company_code) for q in questions]
        sw_industry_names: List[List] = [q.get(service.app.param.sw_industry_name) for q in questions]
        zjh_industry_names: List[List] = [q.get(service.app.param.zjh_industry_name) for q in questions]
        plates: List[List] = [q.get(service.app.param.plate) for q in questions]
    except Exception:
        return http_obj.construct_error_msg(result={}, msg='传入参数存在问题：%s' % traceback.format_exc())

    try:
        # 获取问题类别
        start_time = time.time()
        classes = get_business_type(question_lst)
        logger.info('***用户问题类型：%s，执行时间：%ss' % (classes, str(round(time.time() - start_time, 2))))
    except Exception:
        logger.error(traceback.format_exc())
        return http_obj.construct_error_msg(result={}, msg='调用分类模型报错：%s' % traceback.format_exc())

    try:
        # 问题向量化
        start_time = time.time()
        vectors = sbert_encode(question_lst)
        logger.info('***用户问题向量化时间：%ss' % str(round(time.time() - start_time, 2)))
    except Exception:
        logger.error(traceback.format_exc())
        return http_obj.construct_error_msg(result={}, msg='调用问题编码模型报错：%s' % traceback.format_exc())

    # 一组问题的返回答案
    answers = []
    for question, question_id, company_code, sw_industry, zjh_industry, plate, cls, vector in zip(
            question_lst, question_ids, company_codes, sw_industry_names, zjh_industry_names, plates, classes, vectors):

        # 无法回答的问题返回默认回答
        if any([len(question) > QUESTION_MAX_LEN, cls not in AVAILABLE_CLS]):
            answers.append(construct_answers(question, sw_industry, zjh_industry, plate, [DEFAULT_ANSWER]))
            continue

        if not all([
            isinstance(sw_industry, list),
            isinstance(zjh_industry, list),
            isinstance(plate, list)
        ]):
            return http_obj.construct_error_msg(result={}, msg='%s，证监会、申万行业和板块参数必须为列表格式。' % question)

        logger.info('问题类型：%s' % cls)
        logger.info('申万、证监会行业：%s，%s' % (json.dumps(sw_industry, ensure_ascii=False),
                                        json.dumps(zjh_industry, ensure_ascii=False)))
        logger.info('板块：%s' % json.dumps(plate, ensure_ascii=False))

        try:
            # 查询不适用表，剔除用户标注为“不适用”的结果
            pairs = query_unmatched_pairs(question_id, company_code)

            # 获取候选匹配问题
            start_time = time.time()
            candidates = query_candidates(es_client, es.index.intelligent_interact_qa, cls,
                                          sw_industry, zjh_industry, plate, pairs)
            logger.info('***查询智能匹配索引时间：%ss' % str(round(time.time() - start_time, 2)))

            # 计算相似度
            start_time = time.time()
            res = calculate_cosine_similarity(vector, candidates)
            logger.info('***计算语义相似度时间：%ss' % str(round(time.time() - start_time, 2)))

            if res:
                q_ids = [q.get(ai_qa.ID) for q in res[: 2 * CANDIDATE_NUM]]
                start_time = time.time()
                interact_data = query_interact_data(es_client, es.index.interact_qa, q_ids)
                interact_data = refine_res(res[: 2 * CANDIDATE_NUM], interact_data)
                logger.info('***查询互动问答索引时间：%ss' % str(round(time.time() - start_time, 2)))
                answers.append(construct_answers(question, sw_industry, zjh_industry, plate, interact_data))
            else:
                answers.append(construct_answers(question, sw_industry, zjh_industry, plate, []))
        except Exception:
            logger.error(traceback.format_exc())
            continue
    return http_obj.construct_success_msg(answers)


@app.route(url.HEALTH_CHECK_CONTEXT)
def health():
    """ 用于注册中心健康检查
    :return:
    """
    return json.dumps({'status': "UP"})


if __name__ == '__main__':
    app.run(debug=False, port=service.app.port)
