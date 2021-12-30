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
from sample.LHS_sample import LHSample

class wLHS():
    def __init__(self, K, C):
        self.K = K
        self.C = C
        # 保存所有维度参数对应的采样值，第一维保存参数1的k个采样值，第二维保存参数2的k个采样值
        self.asample = []


    # 生成0-1之间的随机实数
    def getRandom(self):
        return random.random()

    '''
     * @param corr 参数权重
     * @param K 样本点采集个数
     * @param C 超参数：重要性采样的积极程度，如果C越大则导致更积极的重要性采样
     * @param lower 区间起始
     * @param upper 区间末尾
     * Zj: 返回分割点 Zj
    '''
    def funcZi(self, corr, C, lower, upper, j, d, K):
        return -math.log(math.exp(-corr * C * lower) - (corr * C * j )/ (d * K)) / corr * C

    '''
     * @param h 缩放参数
     * @param corr 参数权重
     * @param d 归一化系数
     * @param C 超参数：重要性采样的积极程度，如果C越大则导致更积极的重要性采样
     * 返回在 【zj，zj1】之间随机生成一个参数值
    '''
    def getPoint(self, zj,  zj1, d, corr, C):
        h = d * (math.exp(-corr * C * zj) - math.exp(-corr * C * zj1)) / corr * C
        return -math.log(math.exp(-corr * C * zj) - self.getRandom() * (math.exp(-corr * C * zj) - math.exp(-corr * C * zj1))) / corr * C
        # return np.random.uniform(zj, zj1)

    # 返回：某一个参数值的K个采样点
    def getPoints(self, corr, K, C, lower, upper):
        l = []
        # 新建一个空数组
        # zarray = np.empty((1, K+1))
        zarray = [0 for x in range(0, K+1)]
        zarray[0] = lower
        zarray[K] = upper
        # 如果某一个变量的corr = 0，则对该变量使用标准LHS抽样
        if corr == 0:
            # corr = 0.00001
            print('corr = 0，使用标准lhs抽样')
            bounds = [[lower,upper]]
            std_lhs = LHSample(len(bounds), bounds, K)
            l = std_lhs.lhs()
            self.asample.append(l.ravel().tolist())
        # 只有当某一个变量给的corr不等于0时，对该变量使用wLHS抽样
        else:
            print('corr ！= 0，使用wlhs抽样')
            d = (corr * C) / (math.exp(-corr * lower * C) - math.exp(-corr * upper * C))
            # 计算第1 - k         K个采样点 0-1         k-1~K
            for j in range(1,K+1) :
                # 计算第j个分割点
                zarray[j] = self.funcZi(corr, C, lower, upper, j, d, K)
                # 防止分割点越界，产生超出原始区间范围的参数值
                if zarray[j] < lower:
                    zarray[j] = lower
                elif zarray[j] > upper:
                    zarray[j] = upper
                # 根据分割点区间在分割区间内采样一个值
                x = self.getPoint(zarray[j-1], zarray[j], d, corr, C)
                # 防止参数越界
                l.append(x) # * (high-low)+low
            self.asample.append(l)

    # 将每一个参数值的采样点shuffle后合并,得到所有参数采样值打散后合并的最终的样本
    def all_samples(self):
        # N个样本
        N = np.shape(self.asample)[1]
        # D维参数（特征）
        D = np.shape(self.asample)[0]
        result = np.empty([N, D])
        temp = np.empty([N])
        # 每一个参数（共D维）生成N个值（样本）
        for i in range(D):
            # 每一维参数，生成的N个样本值
            for j in range(N):
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
        return result

    # wlhs入口
    def w_lhs(self, K ,C, corr, pbounds):
        self.asample = []
        for corr, pname in zip(corr, pbounds):
            a = corr
            lower = pbounds[pname][0]
            upper = pbounds[pname][1]
            self.getPoints(corr, K, C, lower, upper)
        result = self.all_samples()
        print('------------lasso计算的corr生成的lhs样本------------')
        print(result)

'''
    初始使用标准lhs，生成初始样本（因为初始没有办法得到变量与target之间的相关性
'''
def std_lhs(pbounds, N):
    bounds = []
    for k,v in pbounds.items():
        bounds.append(list(v))
    # 参数1：变量个数；参数2：bounds= [[0,90],[0,30]]  参数3：需要生成几个初始样本
    std_lhs = LHSample(len(pbounds), bounds, N)
    std_result = std_lhs.lhs()
    print('----------标准lhs生成的样本------------')
    print((std_result))
    return std_result

# 黑盒函数，计算一组参数对应的target
def black_box_function(params):
    x = params
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -1.0 * (-20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e)

# lasso线性回归，计算每一维度参数值与target的相关性系数
def l1_lasso(result, target, para_names):
    from sklearn.linear_model import Lasso
    # from correlation import linear_model

    X = result
    Y = target
    # names = para_names

    lasso = Lasso(alpha=.3)
    lasso.fit(X, Y)
    # print(lasso.coef_)
    # print("Lasso model: ", linear_model.pretty_print_linear(lasso.coef_, names, sort=True))
    return lasso.coef_


if __name__ == '__main__':
    # 样本点采集个数
    K = 30
    # 参数和参数范围
    pbounds = {'x': (-5, 5), 'y': (-2, 15)}
    '''
        使用标准lhs生成初始样本
    '''
    std_result = std_lhs(pbounds, K)

    '''
        获取参数获得对应的target值，生成target列表
    '''
    # temp字典
    temp = {}
    # 专门用一个list保存target，用于lasso相关性分析
    std_target = []
    # 获取采样样本对应的target
    for r in std_result:
        std_target.append(black_box_function(r))
    # print(std_target)

    '''
       计算std_lhs生成的初始样本样本与target的lasso回归模型，得到每个参数的相关性系数corr，每个参数的corr不等，corr可能为0
    '''
    pname = []
    for para_name in pbounds:
        pname.append(para_name)
    # 计算样本与target的lasso相关性
    corr = l1_lasso(std_result, std_target, pname)
    print('\n lasso计算的corr:' + str(corr))

    '''
        使用wlhs产生样本
    '''
    # C 超参数：重要性采样的积极程度
    C = 0.9
    # 生成lhs实例
    w = wLHS(K, C)
    w.w_lhs(K, C, corr, pbounds)

