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

    def getPoints(self, a, K, C, A, B):
        l = []
        # 新建一个空数组
        # zarray = np.empty((1, K+1))
        zarray = [0 for x in range(0, K+1)]
        zarray[0] = A
        zarray[K] = B
        d = (a * C) / (math.exp(-a * A * C) - math.exp(-a * B * C))
        # 计算第1 - k - 1 b        个分割点
        for j in range(1,K) :
            # 计算第j个分割点
            zarray[j] = self.funcZi(a, C, A, B, j, d, K)
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
        return l

if __name__ == '__main__':
    a = 0.0000001
    # a = 0.7
    K = 10
    C = 0.9
    A = 5
    B = 20
    w = wLHS(a, K, C, A, B)

    l = w.getPoints(a, K, C, A, B)
    print(l)