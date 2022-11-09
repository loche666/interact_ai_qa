"""用于定义Consul注册中心操作"""

import json
from random import randint

import consul
import requests

from src.constants.config_constants import url


class ConsulClient:
    """consul 操作类"""

    def __init__(self, host=None, port=None, token=None):
        """ 初始化，指定consul主机，端口，和token
        :param host:
        :param port:
        :param token:
        """
        self.host = host  # consul 主机
        self.port = port  # consul 端口
        self.token = token
        self.consul = consul.Consul(host=host, port=port)

    def register(self, name, service_id, address, port, tags, interval, check):
        """ 注册服务 注册服务的服务名  端口  以及 健康监测端口
        :param name:
        :param service_id:
        :param address:
        :param port:
        :param tags:
        :param interval:
        :param check:
        :return:
        """
        self.consul.agent.service.register(name, service_id=service_id, address=address, port=port, tags=tags,
                                           interval=interval, check=check, token=self.token)

    def deregister(self, service_id):
        """此处有坑，源代码用的get方法是不对的，改成put,两个方法都得改
        :param service_id:
        :return:
        """
        self.consul.agent.service.deregister(service_id)
        self.consul.agent.check.deregister(service_id)

    def getService(self, name):
        """负债均衡获取服务实例
        :param name:
        :return:
        """

        # 获取相应服务下的DataCenter
        data_center_set_url = url.CONSUL_CENTER_SET_URL.format(self.host, self.port, name)
        data_center_resp = requests.get(data_center_set_url)
        if data_center_resp.status_code != 200:
            raise Exception('无法连接到 consul')
        list_data = json.loads(data_center_resp.text)
        dcset = set()  # DataCenter 集合 初始化
        for service in list_data:
            dcset.add(service.get('Datacenter'))
        service_list = []  # 服务列表 初始化
        for dc in dcset:
            dc_url = url.CONSUL_DATA_CENTER_URL.format(self.host, self.port, name, dc)
            # 如果存在token
            if self.token:
                dc_url = '{}{}'.format(dc_url, self.token)
            resp = requests.get(dc_url)
            if resp.status_code != 200:
                raise Exception('无法连接到 consul')
            text = resp.text
            service_list_data = json.loads(text)

            for serv in service_list_data:
                status = serv.get('Checks')[1].get('Status')
                if status == 'passing':  # 选取成功的节点
                    address = serv.get('Service').get('Address')
                    port = serv.get('Service').get('Port')
                    service_list.append({'port': port, 'address': address})
        if len(service_list) == 0:
            raise Exception('没有可用的Consul服务')

        # 随机获取一个可用的服务实例
        service = service_list[randint(0, len(service_list) - 1)]
        return service['address'], int(service['port'])

    def getServices(self):
        """
        :return:
        """
        return self.consul.agent.services()
