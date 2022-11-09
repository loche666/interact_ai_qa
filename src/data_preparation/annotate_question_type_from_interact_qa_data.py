# -*- coding: utf-8 -*-
# @Desc   : 构建业务领域分类模型数据集
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/6/30
import os
import re

from elasticsearch import Elasticsearch

from src.common_utils.common_utils import get_es_hosts
from src.common_utils.es_utils import query_existing_questions
from src.constants.common_constants import QUESTION_MAX_LEN
from src.constants.config_constants import es
from src.constants.index_constants import interact_qa

# 数据集基础路径
data_set_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  f'model_training/business_type_model/data')

available_cls = ['基本情况', '公司业务及主要产品情况', '历史沿革', '控股参股公司情况', '中介机构',
                 '经营理念与商业模式', '发展战略规划', '产品销售情况', '产品价格及定价策略', '客户情况', '出口及海外业务情况', '采购相关情况', '供应商情况',
                 '采购价格', '生产相关情况', '产能情况', '产品质量及合格率', '知识产权与核心技术', '认证、资质及准入情况', '风险及外部影响因素',
                 '研发及技术储备', '土地使用权',
                 '股东主要情况', '控股股东或实际控制人情况', '解禁', '增持', '减持', '国企改革', '公司股权变更情况', '股权质押情况', '回购及缩股'
                 ]

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
                                "gte": "now-30d",
                                "lte": "now"
                            }
                        }
                    }
                ]
            }
        },
        "_source": [
            interact_qa.QUESTION,
            interact_qa.QUESTION_TYPE
        ]
    }

    # 获取全部原始数据
    es_client = Elasticsearch(get_es_hosts(), timeout=5000)
    query_res = query_existing_questions(es_client, es.index.interact_qa, query_body)
    questions = [e for e in query_res if
                 all([e.get(interact_qa.QUESTION_TYPE), len(e.get(interact_qa.QUESTION)) <= QUESTION_MAX_LEN])]
    questions = [[q.get(interact_qa.QUESTION), ','.join(q.get(interact_qa.QUESTION_TYPE))] for q in questions]
    questions = [e for e in questions if re.search('|'.join(available_cls), e[1])]

    for q in questions:
        if re.search('|'.join(available_cls[:5]), q[1]):
            # q.append(AVAILABLE_CLS[0])
            q.append(1)
        if re.search('|'.join(available_cls[5:22]), q[1]):
            # q.append(AVAILABLE_CLS[1])
            q.append(2)
        if re.search('|'.join(available_cls[22:]), q[1]):
            # q.append(AVAILABLE_CLS[2])
            q.append(5)

    with open(os.path.join(data_set_base_path, '待确认问题.txt'), 'w') as corpus_file:
        output_lines = []
        for q in questions:
            if len(q) == 3:
                # output_lines.append('{},{}\n'.format(re.sub('\n|\r|\t', ' ', q[0]).replace(',', '，'), q[-1]))
                output_lines.append('{}\t{}\n'.format(re.sub('\n|\r|\t', ' ', q[0]), q[-1]))
            else:
                continue
        corpus_file.writelines(output_lines)
