#coding=utf-8
from __future__ import division
__author__ = 'wanghai'
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.pyplot as plt
from Samplespace import SampleSpace

def LHSample(D, bounds, N):
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''
    # 结果为：N个样本，每个样本有D个参数（特征）
    result = np.empty([N, D])
    temp = np.empty([N])
    # 采样距离间隔
    d = 1.0 / N

    # D = 2， N = 30， i从0-1， j从0-29
    # 在【0，1】中间分成30个区域，在每个区域内生成一个实数，共生成30个实数存入temp[30]中
    # 对每一个参数，生成30个位于【0，1】之间有固定间隔的实数值
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]
        # 将序列的所有元素随机排序（打散temp数组）
        np.random.shuffle(temp)

        # 将temp中生成的30个实数赋值给result作为参数1的随机生成值
        for j in range(N):
            # 第j个样本，第i个参数值
            result[j, i] = temp[j]


    #对样本数据进行拉伸
    b = np.array(bounds)
    # 获取所有参数的范围下界 [0 0]
    lower_bounds = b[:,0]
    # 获取所有参数的范围上界 [90 30]
    upper_bounds = b[:,1]
    # 如果下界超过上界的范围，报错
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    # multiply数组和矩阵对应位置相乘：result = 30*2, (upper_bounds - lower_bounds) = 1*2扩展成30*2 ,对应位置相乘 返回30 * 2矩阵
    # add矩阵相加
    print(upper_bounds - lower_bounds)
    # print()
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    print('lhs方差为 ： ' + str(np.var(np.asarray(result))))
    return result

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
    print('random方差为 ： ' + str(np.var(np.asarray(result))))
    return result

# 将第一坐标和第二坐标画图
def draw_sample(D, bounds, keys,  N, kind):
    # 根据lhs间隔，计算范围的间隔距离
    xs = (bounds[0][1] - bounds[0][0])/N  # 参数x的maxbound - minbound/N     = 90 / 3 = 30
    ys = (bounds[1][1] - bounds[1][0])/N  # 参数y的maxbound - minbound/N     = 30 / 3 = 10
    # 不指定ylim和xlim，则坐标范y围为默认的0，1
    ax = plt.gca()
    # plt.ylim(ymin, ymax)  指定y轴的坐标范围为ymin, ymax
    plt.ylim(bounds[1][0] - ys,bounds[1][1]+ys)
    plt.ylabel(keys[1])
    # plt.xlim(xmin, xmax)  指定x轴的坐标范围为xmin，xmax
    plt.xlim(bounds[0][0] - xs, bounds[0][1] + xs)
    plt.xlabel(keys[0])
    # 画网格
    plt.grid()
    # 修改坐标轴主刻度的位置：根据参数的范围间隔距离指定坐标轴中网格线位置
    ax.xaxis.set_major_locator( MultipleLocator(xs) )
    ax.yaxis.set_major_locator(MultipleLocator(ys))
    # 使用LHS生成样本(返回的所有样本点嵌套在一个大列表里）
    if kind.lower() == 'random':
        samples = RandomSample(D,bounds,N)
    elif kind.lower() == 'lhs':
        samples = LHSample(D,bounds,N)
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

# 获得抽样样本
def res(D, bounds, keys, N, kind):
    # 使用LHS生成样本(返回的所有样本点嵌套在一个大列表里）
    if kind.lower() == 'random':
        samples = RandomSample(D,bounds,N)
    elif kind.lower() == 'lhs':
        samples = LHSample(D,bounds,N)
    # 将列表list或元组tuple转换为 ndarray 数组:数组长度维N的数量，指定LHS的N，则生成N个样本
    params = np.array(samples)
    # 将key和参数值拼接在一起
    return  [dict(zip(keys, p)) for p in params]

if __name__ =='__main__':
    # pbounds = {'x': (0,90),
    #             'y': (0, 30)}
    pbounds = {'n_estimators': (10, 250),
     'min_samples_split': (2, 25),
     'max_features': (0.1, 0.999),
     'max_depth': (5, 15)}

    sp = SampleSpace(pbounds)

    D = sp._dim # 两个参数
    N = 15 # LHS层数为30层（将范围划分为30份）
    bounds = sp._bounds  # 参数的边界范围
    keys = sp._keys
    kind = 'random'
    # kind = 'lhs'
    # draw_sample(D, bounds, keys,  N, kind)
    lhs_samples = res(D, bounds, keys, N, 'lhs')
    rd_samples = res(D, bounds, keys, N, 'random')
    # for i in samples:
    #     print(i)
