#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：feature selection 
@File ：RF_selecttion_feature.py
@Author ：Yang
@Date ：2021/12/28 18:52 
'''

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

'''
随机森林由多个决策树构成。决策树中的每一个节点都是关于某个特征的条件，为的是将数据集按照不同的响应变量一分为二。
利用不纯度可以确定节点（最优条件），对于分类问题，通常采用基尼不纯度或者信息增益，对于回归问题，通常采用的是方差或者最小二乘拟合。
当训练决策树的时候，可以计算出每个特征减少了多少树的不纯度。对于一个决策树森林来说，可以算出每个特征平均减少了多少不纯度，并把它平均减少的不纯度作为特征选择的值。

下边的例子是sklearn中基于随机森林的特征重要度度量方法：

这里特征得分实际上采用的是Gini Importance。使用基于不纯度的方法的时候，要记住：
1、这种方法存在偏向，对具有更多类别的变量会更有利；2、对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），
并且一旦某个特征被选择之后，其他特征的重要度就会急剧下降，因为不纯度已经被选中的那个特征降下来了，其他的特征就很难再降低那么多不纯度了，
这样一来，只有先被选中的那个特征重要度很高，其他的关联特征重要度往往较低。在理解数据时，这就会造成误解，导致错误的认为先被选中的特征是很重要的，
而其余的特征是不重要的，但实际上这些特征对响应变量的作用确实非常接近的（这跟Lasso是很像的）。
'''
def mean_decrease_impurity():
    #Load boston housing dataset as an example


    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))

'''
另一种常用的特征选择方法就是直接度量每个特征对模型精确率的影响。主要思路是打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。
很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。

在这个例子当中，LSTAT和RM这两个特征对模型的性能有着很大的影响，打乱这两个特征的特征值使得模型的性能下降了73%和57%。
注意，尽管这些我们是在所有特征上进行了训练得到了模型，然后才得到了每个特征的重要性测试，
这并不意味着我们扔掉某个或者某些重要特征后模型的性能就一定会下降很多，因为即便某个特征删掉之后，其关联特征一样可以发挥作用，让模型性能基本上不变。
'''
def mean_decrease_accuracy():
    from sklearn.model_selection import ShuffleSplit, train_test_split
    from sklearn.metrics import r2_score
    from collections import defaultdict


    rf = RandomForestRegressor()
    scores = defaultdict(list)

    # crossvalidate the scores on a number of different random splits of the data
    # for train_idx, test_idx in ShuffleSplit(n_splits=len(X), train_size=100, test_size=0.3):
    #     X_train, X_test = X[train_idx], X[test_idx]
    #     Y_train, Y_test = Y[train_idx], Y[test_idx]
    X_train, X_test , Y_train, Y_test = train_test_split(X,Y,train_size=100,test_size=0.3)
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    # X.shape[1]：遍历特征
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc - shuff_acc) / acc)
    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))

if __name__ == '__main__':
    mean_decrease_impurity()
    # mean_decrease_accuracy()