#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：feature selection 
@File ：learning_model_for_feature_rank.py
@Author ：Yang
@Date ：2021/12/28 17:11 
'''
'''
基于学习模型的特征排序 (Model based ranking)

这种方法的思路是直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。
其实Pearson相关系数等价于线性回归里的标准化回归系数。
假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树、随机森林）、
或者扩展的线性模型等。基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。
但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。

参考：https://www.cnblogs.com/hhh5460/p/5186226.html
'''
from sklearn.model_selection  import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

'''
针对每个单独的特征和响应变量建立预测模型
'''
def get_nonlinear_scores():
    # estimator：学习器, max_depth:树的最大深度
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    # X.shape：(506, 13) ，i从0 - 12
    for i in range(X.shape[1]):
         # ShuffleSplit类用于将样本集合随机“打散”后划分为训练集、测试集(n_splits:打乱和划分的次数)
         # cross_val_score(model_name, X,y， cv=k) , 验证某个模型在某个训练集上的稳定性，输出k个预测精度
         score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                  cv=ShuffleSplit(n_splits=len(X), test_size=0.3))
         scores.append((round(np.mean(score), 3), names[i]))
    print(sorted(scores, reverse=True))

if __name__ == '__main__':
    get_nonlinear_scores()