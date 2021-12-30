#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：samle algorithm 
@File ：2 wLHS.py
@Author ：Yang
@Date ：2021/12/29 11:42 
'''

'''
加权拉丁超立方抽样

参考文献：AutoConfig: Automatic Configuration Tuning for Distributed Message Systems
github：https://github.com/sselab/autoconfig

'''
import math
import random
import numpy as np

class wLHS():
    def __init__(self,a, K, C, A, B):
        self.a = a
        self.K = K
        self.C = C
        self.A = A
        self.B = B
        self.asample = []


    # 生成0-1之间的随机数
    def getRandom(self):
        return random.random()

    '''
     * @param a 参数权重
     * @param K 样本点采集个数
     * @param C 超参数：重要性采样的积极程度，如果C越大则导致更积极的重要性采样
     * @param A 区间起始
     * @param B 区间末尾
     * Zj: 返回分割点 Zj
    '''
    def funcZi(self, a, C, A, B, j, d, K):
        return -math.log(math.exp(-a * C * A) - (a * C * j )/ (d * K)) / a * C

    '''
     * @param h 缩放参数
     * @param a 参数权重
     * @param d 归一化系数
     * @param C 超参数：重要性采样的积极程度，如果C越大则导致更积极的重要性采样
     * 返回在 【zj，zj1】之间随机生成一个参数值
    '''
    def getPoint(self, zj,  zj1, d, a, C):
        h = d * (math.exp(-a * C * zj) - math.exp(-a * C * zj1)) / a * C
        # return -math.log(math.exp(-a * C * zj) - self.getRandom() * (math.exp(-a * C * zj) - math.exp(-a * C * zj1))) / a * C
        return np.random.uniform(zj, zj1)

    # 返回：某一个参数值的K个采样点
    def getPoints(self, a, K, C, A, B):
        l = []
        # 新建一个空数组
        # zarray = np.empty((1, K+1))
        zarray = [0 for x in range(0, K+1)]
        zarray[0] = A
        zarray[K] = B
        if a == 0:
            a = 0.00001
        d = (a * C) / (math.exp(-a * A * C) - math.exp(-a * B * C))
        # 计算第1 - k         K个采样点 0-1         k-1~K
        for j in range(1,K+1) :
            # 计算第j个分割点
            zarray[j] = self.funcZi(a, C, A, B, j, d, K)
            # 防止分割点越界，产生超出原始区间范围的参数值
            if zarray[j] < A:
                zarray[j] = A
            elif zarray[j] > B:
                zarray[j] = B
            # 根据分割点区间在分割区间内采样一个值
            x = self.getPoint(zarray[j-1], zarray[j], d, a, C)
            # 防止参数越界
            l.append(x); # * (high-low)+low
        print('分割点：' + str(zarray))
        print('采样值:' + str(l) + '\n')
        self.asample.append(l)
        return l

    # 将每一个参数值的采样点shuffle  合并
    def all_samples(self):
        # N个样本
        N = np.shape(self.asample)[1]
        # D维参数（特征）
        D = np.shape(self.asample)[0]
        result = np.empty([N, D])
        temp = np.empty([N])
        # self.asample = self.asample[0]
        # 每一个参数（共D维）生成N个值（样本）
        for i in range(D):
            # print(i)
            # 每一维参数，生成的N个样本值
            for j in range(N):
                # print(j)
                try:
                    temp[j] = self.asample[i][j]
                except:
                    print('第{0}条数据处理失败'.format(j))
            # 将每一维参数的样本值打乱
            np.random.shuffle(temp)
            # 把所有的N个样本存入result中
            # 将每一维参数产生的N个样本值放入该维对应的位置（i)
            for j in range(N):
                # 第j个样本，第i个参数值
                result[j, i] = temp[j]
        # print(result)
        return result

def black_box_function(params):
    x = params
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -1.0 * (-20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e)
    print(params)

def l1_lasso(result, target, para_names):
    from sklearn.linear_model import Lasso
    from Correlation import linear_model

    X = result
    Y = target
    names = para_names

    lasso = Lasso(alpha=.3)
    lasso.fit(X, Y)
    # print(lasso.coef_)
    # print("Lasso model: ", linear_model.pretty_print_linear(lasso.coef_, names, sort=True))
    return lasso.coef_


if __name__ == '__main__':
    # 传入参数x的相关性
    a = 0.0000001
    # a = 0.7
    # 样本点采集个数
    K = 30
    # C 超参数：重要性采样的积极程度
    C = 0.9
    # A：lower 下界
    A = 5
    # B：upper 上界
    B = 20

    w = wLHS(a, K, C, A, B)

    pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    pname = []

    '''
       先用固定的a = 0.0000001生成K个样本
    '''
    # 对每一个参数都生成K个数值
    for para_name in pbounds:
        pname.append(para_name)
        lower = pbounds[para_name][0]
        upper = pbounds[para_name][1]
        l = w.getPoints(a, K, C, lower, upper)

    # 将每个参数的数值打乱后拼接在一起，得到采样总共的30个样本
    result = w.all_samples()
    print(result)

    # temp字典
    temp = {}
    # samples_with_target保存所用样本的参数值和target的值，外层为list，内层为N个样本的字典表达
    samples_with_target = []
    # 专门用一个list保存target，用于lasso相关性分析
    target = []
    # 获取采样样本对应的target
    for r in result:
        temp['parameters'] = r
        temp['target'] = black_box_function(r)
        target.append(black_box_function(r))
        samples_with_target.append(temp)
    # print(samples_with_target)
    '''
       计算样本与target的lasso回归模型，得到每个参数的相关性系数
    '''
    # 计算样本与target的lasso相关性
    corr = l1_lasso(result, target, pname)
    print(corr)
    '''
       使用lasso得到的相关性系数再采样K个样本
    '''

    i = 1
    w.asample = []
    for corr, pname in zip(corr, pbounds):
        a = corr
        lower = pbounds[pname][0]
        upper = pbounds[pname][1]
        l = w.getPoints(corr, K, C, lower, upper)
        print(l)
    result = w.all_samples()
    print(result)
