import logging
import traceback
from typing import List

import pymysql.cursors
from pymysql import DatabaseError

from src.constants.config_constants import db as db_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

QUERY_SCRIPT = "SELECT not_apply_question_id FROM interact_qa_not_apply_map WHERE question_id = %s AND company_code = %s"


class DB:
    """该类用于连接IPO抽取数据库，并执行数据库相关操作"""

    def __init__(self, db_conf):
        """初始化连接db的方法
        :param db_conf:数据库连接信息
        """
        # 连接数据库
        self.connect = pymysql.Connect(**db_conf)
        # 获取游标
        self.cursor = self.connect.cursor()
        # datetime格式
        self.datetime_format = '%Y-%m-%d %H:%M:%S'

    def close(self):
        """关闭数据库所有连接
        :return:
        """
        self.cursor.close()
        self.connect.close()

    def query_pairs(self, question_id: str, company_code: str) -> List:
        """
        :param question_id:
        :param company_code:
        :return:
        """
        try:
            self.cursor.execute(QUERY_SCRIPT, (question_id, company_code))
            pairs = [t[0] for t in self.cursor.fetchall()]
            return pairs
        except DatabaseError:
            logger.error('%s %s查询数据异常. %s' % (question_id, company_code, traceback.format_exc()))
        return []


def query_unmatched_pairs(question_id: str, company_code: str) -> List:
    """ 查询不适用表
    :param question_id:
    :param company_code:
    :return:
    """
    db_info = {
        'host': db_config.host,
        'port': db_config.port,
        'user': db_config.user,
        'passwd': db_config.passwd,
        'db': db_config.name,
        'charset': db_config.charset
    }

    db = DB(db_info)
    unmatched_pairs = db.query_pairs(question_id, company_code)
    db.close()
    return unmatched_pairs
