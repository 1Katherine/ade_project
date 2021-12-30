#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：feature selection 
@File ：linear_model.py
@Author ：Yang
@Date ：2021/12/28 17:51 
'''


import numpy as np

'''
输出特征x以及对应的系数（系数越大，特征越重要）
'''
#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names is None:
        # x = 0，1，2  封装成 X0，X1，X2 格式, name = ['X0', 'X1', 'X2']
        names = ["X%s" % x for x in range(len(coefs))]
    # 将系数和特征名拼成元组数据
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    # 将相关性系数和特征参数拼接成相乘的格式 : 0.984 * X0 , 1.995 * X1 , -0.041 * X2
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

    # coef_是线性回归模型的系数 , Linear model: 0.984 * X0 + 1.995 * X1 + -0.041 * X2

'''
如何用回归模型的系数来选择特征。越是重要的特征在模型中对应的系数就会越大，
而跟输出变量越是无关的特征对应的系数就会越接近于0。在噪音不多的数据上，或者是数据量远远大于特征数的数据上，
如果特征之间相对来说是比较独立的，那么即便是运用最简单的线性回归模型也一样能取得非常好的效果。

*** 使用线性回归拟合线性模型
'''
def LinearModel():
    from sklearn.linear_model import LinearRegression
    np.random.seed(0)
    size = 5000
    # A dataset with 3 features
    # 生成正态分布数值。高斯分布的概率密度函数,概率分布的均值为0，标准差为1（对应分布的宽度），输出的shape（5000*3）5000行3列
    X = np.random.normal(0, 1, (size, 3))
    print(X)
    # Y = X0 + 2*X1 + noise
    # 1 定义线性回归方程（）用x值计算y值
    Y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 2, size)
    # 2 实例化线性回归模型
    lr = LinearRegression()
    # 3 用数据拟合模型
    lr.fit(X, Y)
    print("Linear model:", pretty_print_linear(lr.coef_))


'''
在很多实际的数据当中，往往存在多个互相关联的特征，这时候模型就会变得不稳定，
数据中细微的变化就可能导致模型的巨大变化（模型的变化本质上是系数，或者叫参数，可以理解成W），
这会让模型的预测变得困难，这种现象也称为多重共线性。例如，假设我们有个数据集，它的真实模型应该是Y=X1+X2，
当我们观察的时候，发现Y’=X1+X2+e，e是噪音。如果X1和X2之间存在线性关系，
例如X1约等于X2，这个时候由于噪音e的存在，我们学到的模型可能就不是Y=X1+X2了，有可能是Y=2X1，或者Y=-X1+3X2。

下边这个例子当中，在同一个数据上加入了一些噪音，用随机森林算法进行特征选择。

*** 使用随机森林拟合线性模型
'''
def RandomForest():
    from sklearn.linear_model import LinearRegression

    size = 100
    np.random.seed(seed=5)

    X_seed = np.random.normal(0, 1, size)
    X0 = X_seed + np.random.normal(0, .1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)

    Y = X0 + X1 + X2 + np.random.normal(0, 1, size)
    X = np.array([X0, X1, X2]).T

    lr = LinearRegression()
    lr.fit(X, Y)
    print("Linear model:", pretty_print_linear(lr.coef_))

if __name__ == '__main__':
    LinearModel()

    # RandomForest()