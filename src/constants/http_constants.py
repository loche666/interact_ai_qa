# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/10/21
import json


class HTTPConstants:
    """ 定义http请求用到的常量 """

    def __init__(self):
        """初始化"""

        self.ERROR_CODE_KEY = 'errorCode'
        self.ERROR_MSG_KEY = 'errorMsg'
        self.RESULT = 'result'
        self.SUCCESS_KEY = 'success'

        self.ERROR_CODE_VAL = 500
        self.SUCCESS_VAL = 200

    def construct_error_msg(self, result: object, msg='', status_code=None):
        """构造错误返回信息
        :param status_code: response错误码
        :param result: 调用错误返回结果
        :param msg: 错误信息
        :return:
        """
        return json.dumps({
            self.ERROR_CODE_KEY: status_code if status_code else self.ERROR_CODE_VAL,
            self.ERROR_MSG_KEY: msg,
            self.RESULT: result,
            self.SUCCESS_KEY: False
        }, ensure_ascii=False, indent=4)

    def construct_success_msg(self, result: object, msg=''):
        """构造成功返回信息
        :param result:
        :param msg:
        :return:
        """
        return json.dumps({
            self.ERROR_CODE_KEY: self.SUCCESS_VAL,
            self.ERROR_MSG_KEY: msg,
            self.RESULT: result,
            self.SUCCESS_KEY: True
        }, ensure_ascii=False, indent=4)


http_obj = HTTPConstants()
