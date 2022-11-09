# -*- coding: utf-8 -*-
# @Desc   : Description of File
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/8/12
from src.resource.appSource import config


class ElasticSearchConstants:
    """Elastic search涉及常量"""

    class _Index:
        def __init__(self):
            self.interact_qa = config.get_profile_config('es.index.interact_qa')
            self.intelligent_interact_qa = config.get_profile_config('es.index.intelligent_interact_qa')

    def __init__(self):
        self.index = self._Index()
        self.ip = config.get_profile_config('es.ip')
        self.port = int(config.get_profile_config('es.port'))


class DBConstants:
    """db数据库涉及常量"""

    def __init__(self):
        self.host = config.get_profile_config('db.host')
        self.port = int(config.get_profile_config('db.port'))
        self.user = config.get_profile_config('db.user')
        self.passwd = config.get_profile_config('db.passwd')
        self.name = config.get_profile_config('db.name')
        self.charset = config.get_profile_config('db.charset')


class ServiceConstants:
    """互动问答服务涉及常量"""

    class _AppConstants:
        class _Params:
            def __init__(self):
                self.questions = config.get_profile_config('service.app.param.questions')
                self.question = config.get_profile_config('service.app.param.question')
                self.question_id = config.get_profile_config('service.app.param.question_id')
                self.company_code = config.get_profile_config('service.app.param.company_code')
                self.sw_industry_name = config.get_profile_config('service.app.param.sw_industry_name')
                self.zjh_industry_name = config.get_profile_config('service.app.param.zjh_industry_name')
                self.plate = config.get_profile_config('service.app.param.plate')
                self.answer = config.get_profile_config('service.app.param.answer')

        def __init__(self):
            self.param = self._Params()
            self.context = config.get_profile_config('service.app.context')
            self.get_answers = config.get_profile_config('service.app.get_answers')
            self.url = config.get_profile_config('service.app.url')
            self.port = config.get_profile_config('service.app.port')

    class _ModelConstants:
        class _BusinessTypeConstants:
            def __init__(self):
                self.interface = config.get_profile_config('service.model.business_type.interface')
                self.params = config.get_profile_config('service.model.business_type.params')

        class _QuestionSimilarityConstants:
            def __init__(self):
                self.interface = config.get_profile_config('service.model.sentence_encode.interface')
                self.params = config.get_profile_config('service.model.sentence_encode.params')

        def __init__(self):
            self.context = config.get_profile_config('service.model.context')
            self.url = config.get_profile_config('service.model.url')
            self.port = config.get_profile_config('service.model.port')
            self.business_type = self._BusinessTypeConstants()
            self.sentence_encode = self._QuestionSimilarityConstants()

    def __init__(self):
        self.app = self._AppConstants()
        self.model = self._ModelConstants()


class CudaConstants:
    """Cuda、GPU涉及常量"""

    class _DeviceConstants:
        def __init__(self):
            self.key = config.get_profile_config('cuda.device.key')
            self.val = config.get_profile_config('cuda.device.val')

    def __init__(self):
        self.device = self._DeviceConstants()


class LogstashConstants:
    """logstash涉及常量"""

    def __init__(self):
        self.host = config.get_profile_config('logstash.host')
        self.port = int(config.get_profile_config('logstash.port'))
        self.level = config.get_profile_config('logstash.level')
        self.tag = config.get_profile_config('logstash.tag')
        self.program = config.get_profile_config('logstash.program')


class ConsulConstants:
    """Consul相关常量"""

    def __init__(self):
        self.host = config.get_profile_config('consul.host')
        self.port = config.get_profile_config('consul.port')
        self.token = config.get_profile_config('consul.token')
        self.interval = config.get_profile_config('consul.interval')
        self.service_name = config.get_profile_config('consul.service_name')
        self.local_host = config.get_profile_config('consul.local.host')
        self.local_port = config.get_profile_config('consul.local.port')


class URLConstants:
    """URL相关常量"""

    RETRIEVAL_CONTEXT = '/intelligent_interact_qa/get_answers'
    HEALTH_CHECK_CONTEXT = '/intelligent_interact_qa/health_check'

    ANSWER_RETRIEVAL_URL = 'http://{}:{}' + RETRIEVAL_CONTEXT
    HEALTH_CHECK_URL = 'http://{}:{}' + HEALTH_CHECK_CONTEXT

    CONSUL_CENTER_SET_URL = 'http://{}:{}//v1/catalog/service/{}'
    CONSUL_DATA_CENTER_URL = 'http://{}:{}/v1/health/service/{}?dc={}&token='


# 调用es参数
es = ElasticSearchConstants()
# 调用服务参数
service = ServiceConstants()
# 调用DB参数
db = DBConstants()
# cuda参数
cuda = CudaConstants()
# logstash参数
logstash = LogstashConstants()
# Consul参数
consul = ConsulConstants()
# url参数
url = URLConstants()
