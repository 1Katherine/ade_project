#coding=utf-8
from __future__ import division
__author__ = 'wanghai'
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.pyplot as plt

def RandomSample(D, bounds, N):
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）  bounds = [[0,90],[0,30]]
    :param N:随即搜索产生的样本数量
    :return:样本数据
    '''
    # 结果为：N个样本，每个样本有D个参数（特征）
    result = np.empty([N, D])


    # D = 2， N = 30， i从0-1， j从0-29
    for i in range(D):
        for j in range(N):
            result[j, i] = np.random.uniform(
                low=bounds[i][0], high=bounds[i][1], size = 1)[0]
    return result

def draw_sample(D, bounds, N, kind):
    # 根据lhs间隔，计算范围的间隔距离
    xs = (bounds[0][1] - bounds[0][0])/N  # 参数x的maxbound - minbound/N     = 90 / 3 = 30
    ys = (bounds[1][1] - bounds[1][0])/N  # 参数y的maxbound - minbound/N     = 30 / 3 = 10
    # 不指定ylim和xlim，则坐标范围为默认的0，1
    ax = plt.gca()
    # plt.ylim(ymin, ymax)  指定y轴的坐标范围为ymin, ymax
    plt.ylim(bounds[1][0] - ys,bounds[1][1]+ys)
    # plt.xlim(xmin, xmax)  指定x轴的坐标范围为xmin，xmax
    plt.xlim(bounds[0][0] - xs, bounds[0][1] + xs)
    # 画网格
    plt.grid()
    # 修改坐标轴主刻度的位置：根据参数的范围间隔距离指定坐标轴中网格线位置
    ax.xaxis.set_major_locator( MultipleLocator(xs) )
    ax.yaxis.set_major_locator(MultipleLocator(ys))
    # 使用LHS生成样本(返回的所有样本点嵌套在一个大列表里）
    if kind.lower() == 'random':
        samples = RandomSample(D,bounds,N)
    # 将列表list或元组tuple转换为 ndarray 数组:数组长度维N的数量，指定LHS的N，则生成N个样本
    XY = np.array(samples)
    # 所有样本点的0号元素是对应的参数x的取值
    X = XY[:,0]
    # 所有样本点的1号元素是对应的参数y的取值
    Y = XY[:,1]
    # 画出所有样本点
    plt.scatter(X,Y)
    plt.title(kind.lower(), loc='left', color='b')
    plt.show()

if __name__ =='__main__':
    D = 2 # 两个参数
    N = 30 # LHS层数为30层（将范围划分为30份）
    bounds = [[0,90],[0,30]]  # 参数的边界范围
    kind = 'random'
    draw_sample(D, bounds, N, kind)

