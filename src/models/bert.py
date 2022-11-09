# coding: UTF-8
import os

import torch
import torch.nn as nn

from src.model_training.business_type_model.modeling import BertModel
from src.model_training.business_type_model.tokenization import BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'model_training/pretrain_model/bert_pretrain')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class ModelConfig:
    """仅加载模型参数"""

    def __init__(self, model_name):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'model_training')  # model_training层级路径
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.batch_size = 16  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = os.path.join(self.base_path, 'pretrain_model/bert_pretrain')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_classes = 13
        self.save_path = os.path.join(self.base_path, model_name, 'saved_dict/bert.ckpt')


class Model(nn.Module):
    """模型类"""

    def __init__(self, config):
        """初始化模型对象
        :param config:
        """
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前馈传播
        :param x:
        :return:
        """
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

    def predict(self, x):
        """预测
        :param x:
        :return:
        """
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        pred = torch.max(self.sigmoid(self.fc(pooled).data), 1)
        cls, prob = pred[1].cpu().numpy(), pred[0].cpu().numpy()
        return cls, prob
