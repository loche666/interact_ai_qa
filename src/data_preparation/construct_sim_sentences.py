# -*- coding: utf-8 -*-
# @Desc   : 生成句向量训练语料
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/10/17
import json
import re
import traceback

import requests
from elasticsearch import Elasticsearch

from src.common_utils.common_utils import get_es_hosts
from src.constants.config_constants import service, es
from src.constants.index_constants import interact_qa


es_cli = Elasticsearch(get_es_hosts(), timeout=5000)
threshold = 0.99

if __name__ == '__main__':
    samples = []
    with open('../model_training/business_type_model/data/dev.txt', 'r') as q_file:
        questions = q_file.readlines()
        # TODO：目前仅支持125三类业务问题回答
        questions = [q.split('\t')[0] for q in questions if re.search('\t[125]\n', q)]

    # 调用互动问答接口，找到与之最匹配的问答对
    for i, q in enumerate(questions):
        # 查询问题证券代码
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match_phrase": {
                                interact_qa.QUESTION: q
                            }
                        }
                    ]
                }
            },
            "_source": [interact_qa.SW_INDUSTRY_NAME, interact_qa.ZJH_INDUSTRY_NAME, interact_qa.PLATE]
        }

        try:
            es_res = es_cli.search(index=es.index.interact_qa, body=query).get('hits').get('hits')
            sw_industry = es_res[0].get('_source').get(interact_qa.SW_INDUSTRY_NAME)
            zjh_industry = es_res[0].get('_source').get(interact_qa.ZJH_INDUSTRY_NAME)
            plate = es_res[0].get('_source').get(interact_qa.PLATE)

            # 调用接口返回相似问
            result = requests.post(
                f'{service.app.url}:{service.app.port}/{service.app.context}/{service.app.get_answers}',
                json={
                    service.app.param.questions: [
                        {
                            service.app.param.question: q,
                            service.app.param.sw_industry_name: [sw_industry],
                            service.app.param.zjh_industry_name: [zjh_industry],
                            service.app.param.plate: [plate],
                        }
                    ]
                },
                headers={'Content-Type': 'application/json'},
                timeout=60)
            interface_res = json.loads(bytes.decode(result.content)).get('result')[0].get('answers')[:20]

            # 如果问题不属于1,2,5，则不考虑
            if isinstance(interface_res, str):
                continue
            sim_sentences = [(e.get(interact_qa.QUESTION), e.get('cosine')) for e in interface_res]
            if sim_sentences[0][0] == q:
                sim_sentences = sim_sentences[1:]

            # 构建正样本
            pos_samples = ['{}\t{}\t1\n'.format(q, re.sub('[\n\t\r]', '', e[0])) for e in sim_sentences if
                           e[1] >= threshold]
            neg_samples = ['{}\t{}\t0\n'.format(q, re.sub('[\n\t\r]', '', e[0])) for e in sim_sentences if
                           e[1] < threshold]
            samples.extend(pos_samples)
            samples.extend(neg_samples)
            print('{}/{}'.format(i + 1, len(questions)))
        except Exception:
            print(q, traceback.format_exc())
            continue

    # 保存文件
    with open('../model_training/question_similarity_model/data/samples.txt', 'w') as q_file:
        q_file.writelines(samples)
