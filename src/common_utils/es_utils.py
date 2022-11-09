# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/6/30
import traceback
from typing import Dict, List

from elasticsearch import Elasticsearch

from src.common_utils.logstash_utils import Logger

logger = Logger(__name__)

CORPUS_SIZE = 2000000


def query_existing_questions(es_cli: Elasticsearch, es_index: str, query_body: Dict, corpus_size=CORPUS_SIZE,
                             scroll_maintain_time='10m', scroll_size=10000) -> List:
    """ 查询所有问答对
    :param scroll_maintain_time: es查询时scroll保持时间
    :param scroll_size: es查询时scroll页面尺寸
    :param corpus_size:
    :param es_index:
    :param query_body:
    :param es_cli:
    :return:
    """

    # 全量问题
    questions = []

    try:
        logger.info('查询%s全量数据中...' % es_index)
        page = es_cli.search(index=es_index, scroll=scroll_maintain_time,
                             size=scroll_size, body=query_body)

        sid = page.get('_scroll_id')
        scroll_size = page.get('hits').get('total').get('value')

        questions.extend([elem.get('_source') for elem in page.get('hits').get('hits')])

        logger.info('索引：%s 共计%d个问答对。' % (es_index, scroll_size))

        while scroll_size > 0:
            page = es_cli.scroll(scroll_id=sid, scroll=scroll_maintain_time)
            questions.extend([elem.get('_source') for elem in page.get('hits').get('hits')])
            sid = page.get('_scroll_id')
            scroll_size = len(page.get('hits').get('hits'))
            # 为了提前退出循环查询
            if len(questions) >= corpus_size:
                break
    except Exception:
        logger.error('查询已有问答对出现异常 %s' % traceback.format_exc())
        return []
    return questions
