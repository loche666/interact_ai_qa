# -*- coding: utf-8 -*-
# @Desc   : 对检索的所有问题进行业务分类，验证分类情况
# @Licence: (C) Copyright for ValueOnline
# @Author : chen.long
# @Date   : 2022/8/11
import os

from src.common_utils.common_utils import get_business_type

segment_size = 10000
if __name__ == '__main__':
    res = []
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           f'model_training/business_type_model/data/test_corpus.txt'),
              'r') as question_file:
        questions = [q.strip() for q in question_file.readlines()]

    segment_num = int(len(questions) / segment_size)

    for i in range(segment_num):
        predict_lst = questions[i * segment_size: (i + 1) * segment_size]
        res.extend(get_business_type(predict_lst))

    predict_lst = questions[(segment_num - 1) * segment_size:]
    res.extend(get_business_type(predict_lst))

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           f'model_training/business_type_model/data/test_corpus_res.txt'),
              'w') as res_file:
        res_file.writelines(['{}$$${}\n'.format(q, t) for q, t in zip(questions, res)])
