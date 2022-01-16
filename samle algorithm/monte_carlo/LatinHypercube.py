#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LatinHypercube.py   
@Author ：Yang 
@CreateTime :   2022/1/15 22:07
@Reference ： https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.MultivariateNormalQMC.html#scipy.stats.qmc.MultivariateNormalQMC
'''
from scipy.stats import qmc

# 从拉丁超立方体发生器中生成样本,范围在[0, 1)
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=5)
print('random sample = \n' + str(sample))

# 使用差异标准计算样本的质量
print('discrepancy of sample = ' + str(qmc.discrepancy(sample)))

# 样本可以按比例划分为界线, 范围在[a, b)
l_bounds = [0, 2]
u_bounds = [10, 5]
bounds_samples = qmc.scale(sample, l_bounds, u_bounds)
print('random sample from bounds scale = \n' + str(bounds_samples))

# engine = qmc.MultinomialQMC(pvals=[0.2, 0.4, 0.4])
# sample = engine.random(10)

