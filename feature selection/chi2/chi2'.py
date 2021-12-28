#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：feature selection 
@File ：chi2'.py
@Author ：Yang
@Date ：2021/12/28 16:32 
'''

'''
使用chi2卡方检验做特征选择
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#导入IRIS数据集(共有四个特征) ---- 分类数据，target为0，1，2
iris = load_iris()
# 将iris转为csv格式
feature = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)','petal width (cm)']
def iris_to_csv():
    data_np = pd.DataFrame(iris.data,columns=feature)

    label=pd.DataFrame(iris.target,columns=['target'])

    data_np['target'] = label

    data_np.to_csv(".\\iris.csv")



def get_feature_importance():
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    """
    iris_data = pd.read_csv(".\\iris.csv")
    feature_data = iris_data[feature]
    # label_data = iris_data.iloc[:,-1]
    label_data = iris_data['target']
    model = SelectKBest(chi2, k=2)  # 选择k个最佳特征
    X_new = model.fit_transform(feature_data, label_data)
    # feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征

    print("model shape: ", X_new.shape)

    scores = model.scores_
    print('model scores:', scores)  # 得分越高，特征越重要

    p_values = model.pvalues_
    print('model p-values', p_values)  # p-values 越小，置信度越高，特征越重要

    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1]
    k_best_features = list(feature_data.columns.values[indices[0:2]])

    print('k best features are: ', k_best_features)

    return k_best_features

if __name__ == '__main__':
    iris_to_csv()
    get_feature_importance()
