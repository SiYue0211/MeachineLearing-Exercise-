#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/18 9:43
# @Author  : dawn
# @File    : handWriting.py

import numpy as np
import os
from os import listdir

def img2vector(path):
    """
    将所有数据处理成一个向量
    :param path:
    :return:
    """
    data = np.zeros(shape=(1, 1024), dtype=np.int32)
    with open(path) as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                data[0, i*32+j] = int(line[j])
    return data


def getData(path):
    """
    获取数据和标签
    :param path:
    :return:
    """
    dirs = listdir(path)
    # 共有多少数据
    length = len(dirs)

    # 建立一个data, 用来存放数据
    data = np.empty(shape=(length, 1024))
    data.fill(-1)

    # 建立一个labels,用来存放数据的标签
    labels = np.empty(shape=(length, ), dtype=np.int32)
    labels.fill(-1)
    for i in range(length):
        file_name_str = dirs[i];
        file_name = file_name_str.split('.')[0]
        label = int(file_name.split('_')[0])
        labels[i] = label
        data[i, :] = img2vector(path + '/' + file_name_str)
    return data, labels

def test(train_data, train_label, test_datas, test_label, k=3):
    """
    进行测试,KNN不需要训练
    :param train_data:
    :param train_label:
    :param test_datas:
    :param test_label:
    :return:
    """
    assert train_data.shape[0] == train_label.shape[0],"训练集的数据数量要等于训练集的标签数量"
    assert test_datas.shape[0] == test_label.shape[0], "测试集的数据数量要等于测试集的标签数量"
    train_num = train_data.shape[0]
    test_num = test_datas.shape[0]

    correct_num = 0.0

    for i in range(test_num):
        test_data = test_datas[i, :]
        gap = train_data - np.tile(test_data, (train_num, 1))
        gap = gap**2
        distance = np.sum(gap, axis=1)
        distance = distance**0.5
        max_k_index = np.argsort(distance)
        # 对不同的分类进行计数
        cate_count = {}
        for j in range(k):
            vote_label = train_label[max_k_index[j]]
            print(vote_label)
            cate_count[vote_label] = cate_count.get(vote_label, 0) + 1
        predict = sorted(cate_count.items(), key=lambda d:d[1], reverse=True)[0][0]
        if predict == test_label[i]:
            correct_num += 1.0
    accuacy = correct_num / float(test_num)
    print("accuracy : %f" %accuacy)

if __name__ == "__main__":
    k = 3
    train_path = 'digits/trainingDigits'
    test_path = 'digits/testDigits'
    train_data, train_label = getData(train_path)
    test_data, test_label = getData(test_path)
    test(train_data, train_label, test_data, test_label, k)