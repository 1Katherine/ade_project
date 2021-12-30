#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：feature selection 
@File ：Regulation.py
@Author ：Yang
@Date ：2021/12/28 18:16 
'''


import linear_model
'''
正则化模型
'''

'''
L1正则化将系数w的l1范数作为惩罚项加到损失函数上，由于正则项非零，这就迫使那些弱的特征所对应的系数变成0。
因此L1正则化往往会使学到的模型很稀疏（系数w经常为0），这个特性使得L1正则化成为一种很好的特征选择方法。

Scikit-learn为线性回归提供了Lasso，为分类提供了L1逻辑回归。
下面的例子在波士顿房价数据上运行了Lasso，其中参数alpha是通过grid search进行优化的。

可以看到，很多特征的系数都是0。如果继续增加alpha的值，得到的模型就会越来越稀疏，即越来越多的特征系数会变成0。
然而，L1正则化像非正则化线性模型一样也是不稳定的，如果特征集合中具有相关联的特征，当数据发生细微变化时也有可能导致很大的模型差异。
'''
def l1_lasso():
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_boston

    boston = load_boston()
    scaler = StandardScaler()
    X = scaler.fit_transform(boston["data"])
    print(X)
    Y = boston["target"]
    print(Y)
    names = boston["feature_names"]

    lasso = Lasso(alpha=.3)
    lasso.fit(X, Y)
    print("Lasso model: ", linear_model.pretty_print_linear(lasso.coef_, names, sort=True))

'''
L2惩罚项中系数是二次方的，这使得L2和L1有着诸多差异，最明显的一点就是，L2正则化会让系数的取值变得平均。对于关联特征，这意味着他们能够获得更相近的对应系数。
还是以Y=X1+X2为例，假设X1和X2具有很强的关联，如果用L1正则化，不论学到的模型是Y=X1+X2还是Y=2X1，惩罚都是一样的，都是2alpha。
但是对于L2来说，第一个模型的惩罚项是2alpha，但第二个模型的是4*alpha。可以看出，系数之和为常数时，各系数相等时惩罚是最小的，所以才有了L2会让各个系数趋于相同的特点。

可以看出，L2正则化对于特征选择来说一种稳定的模型，不像L1正则化那样，系数会因为细微的数据变化而波动。

回过头来看看3个互相关联的特征的例子，分别以10个不同的种子随机初始化运行10次，来观察L1和L2正则化的稳定性。

可以看出，不同的数据上线性回归得到的模型（系数）相差甚远，但对于L2正则化模型来说，结果中的系数非常的稳定，差别较小，都比较接近于1，能够反映出数据的内在结构。
'''
def l2_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LinearRegression
    import numpy as np
    size = 100
    #We run the method 10 times with different random seeds
    for i in range(10):
        print("Random seed %s" % i)
        np.random.seed(seed=i)
        X_seed = np.random.normal(0, 1, size)
        X1 = X_seed + np.random.normal(0, .1, size)
        X2 = X_seed + np.random.normal(0, .1, size)
        X3 = X_seed + np.random.normal(0, .1, size)
        Y = X1 + X2 + X3 + np.random.normal(0, 1, size)
        X = np.array([X1, X2, X3]).T

        # l1正则化回归模型
        lr = LinearRegression()
        lr.fit(X,Y)
        print("Linear model:", linear_model.pretty_print_linear(lr.coef_))

        # l2正则化回归模型
        ridge = Ridge(alpha=10)
        ridge.fit(X,Y)
        print("Ridge model:", linear_model.pretty_print_linear(ridge.coef_))

if __name__ == '__main__':
    l1_lasso()
    # l2_ridge()