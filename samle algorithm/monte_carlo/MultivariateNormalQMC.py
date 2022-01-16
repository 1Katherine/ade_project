#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MultivariateNormalQMC.py   
@Author ：Yang 
@CreateTime :   2022/1/15 22:35 
'''

import matplotlib.pyplot as plt
from scipy.stats import qmc
engine = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
sample = engine.random(512)
_ = plt.scatter(sample[:, 0], sample[:, 1])
plt.show()
