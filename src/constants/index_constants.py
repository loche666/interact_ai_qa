# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/8/15


class InteractQADataConstants:
    """已有互动问答索引常量"""

    def __init__(self):
        self.ID = "id"
        self.QUESTION = "interact_qa_data_question_s"
        self.QUESTION_TIME = "interact_qa_data_question_time_dt"
        self.QUESTION_TYPE = "interact_question_class_new_name_txt"
        self.ANSWER_LIST = "interact_qa_data_answer_list"
        self.ANSWER_TIME = "interact_qa_data_answer_time_dt"
        # 申万二级行业代码
        self.SW_INDUSTRY_NAME = "company_industry_code_sw_2021_txt"
        # 证监会二级行业代码
        self.ZJH_INDUSTRY_NAME = "company_industry_code_zjh_txt"
        # 板块代码
        self.PLATE = "company_belongs_plate_code_t"
        self.COMPANY_SORT_NAME = "interact_qa_data_company_sort_name_s"
        self.COMPANY_CODE = "interact_qa_data_company_code_t"
        self.SOURCE = "interact_qa_data_data_source_s"
        self.REPLY_FLAG = "interact_qa_data_reply"


class IntelligentInteractQAConstants:
    """智能互动问答索引常量"""

    def __init__(self):
        self.ID = "id"
        self.QUESTION = "question"
        self.QUESTION_TIME = "question_time"
        self.SW_INDUSTRY_NAME = "sw_industry_code"
        self.ZJH_INDUSTRY_NAME = "zjh_industry_code"
        self.CLS = "cls"
        self.VECTOR = "vector"
        self.PLATE = "plate"
        self.MAPPING = self._set_mapping()

    def _set_mapping(self):
        return {
            "mappings": {
                "properties": {
                    self.CLS: {
                        "type": "keyword"
                    },
                    self.ID: {
                        "type": "keyword"
                    },
                    self.QUESTION: {
                        "type": "text"
                    },
                    self.VECTOR: {
                        "type": "dense_vector"
                    },
                    self.SW_INDUSTRY_NAME: {
                        "type": "keyword"
                    },
                    self.ZJH_INDUSTRY_NAME: {
                        "type": "keyword"
                    },
                    self.PLATE: {
                        "type": "keyword"
                    },
                    self.QUESTION_TIME: {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                    }
                }
            },
            "settings": {
                # 主分片
                "number_of_shards": 1,
                # 副本分片
                "number_of_replicas": 2
            }
        }


interact_qa = InteractQADataConstants()
ai_qa = IntelligentInteractQAConstants()

# 空字段值
EMPTY_COL_VAL = 'None'
