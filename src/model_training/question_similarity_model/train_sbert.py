"""
训练sentence-bert模型
"""
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.constants.common_constants import QUESTION_MAX_LEN

timestamp = '20221107'
new_dimension = 128


def get_dataset():
    """获取训练集
    :return:
    """
    data_list = []
    dev_list = []
    with open(f'data/finetune{timestamp}.txt', encoding='utf8') as data_file:
        for i in data_file:
            data_list.append(InputExample(texts=[i.split('\t')[0], i.split('\t')[1]], label=int(i.split('\t')[2])))
    with open(f'data/finetune{timestamp}.txt', encoding='utf8') as data_file_dev:
        for i in data_file_dev:
            dev_list.append(InputExample(texts=[i.split('\t')[0], i.split('\t')[1]], label=int(i.split('\t')[2])))
    return data_list, dev_list


if __name__ == '__main__':

    # 加载预训练模型
    word_embedding_model = models.Transformer('../pretrain_model/bert_pretrain', max_seq_length=QUESTION_MAX_LEN)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 获取训练集
    train_examples, dev_examples = get_dataset()
    train_sentences = [elem.texts[0] for elem in train_examples]
    train_sentences.extend([elem.texts[1] for elem in train_examples])
    train_sentences.extend([elem.texts[0] for elem in dev_examples])
    train_sentences.extend([elem.texts[1] for elem in dev_examples])

    # 定义训练集并加载
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    # train_loss = losses.CosineSimilarityLoss(model)
    train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=new_dimension, num_labels=2)

    # Compute PCA on the train embeddings matrix
    train_embeddings = model.encode(train_sentences, convert_to_numpy=True)
    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    dense_model = models.Dense(in_features=model.get_sentence_embedding_dimension(),
                               out_features=new_dimension,
                               bias=False,
                               activation_function=torch.nn.Identity())
    dense_model.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense_model)

    # 增加验证机制，参考 https://www.sbert.net/docs/training/overview.html
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(dev_examples, name='dev')

    # 训练模型
    model_dir = os.path.join(os.curdir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=4, warmup_steps=100,
              output_path='model/sbert', evaluation_steps=100)
