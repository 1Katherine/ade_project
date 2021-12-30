#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：bayes_opt 
@File ：test.py
@Author ：Yang
@Date ：2021/12/30 16:30 
'''

'''
    将key和bounds拼成pbounds格式：pbounds = {'x': (-5, 5), 'y': (-2, 15)}
'''
keys = ['spark.default.parallelism', 'spark.executor.cores', 'spark.executor.instances']
temp = [[200.0, 500.0], [1.0, 3.0], [4.0, 8.0]]
bounds = []
for i in temp:
    bounds.append(tuple(i))

pbounds = dict(map(lambda x,y:[x,y],keys,bounds))
# print(pbounds)
