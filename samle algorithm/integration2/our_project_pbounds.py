#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：samle algorithm 
@File ：our_project_pbounds.py
@Author ：Yang
@Date ：2021/12/27 16:28
'''

import  pandas as pd
from SampleAlgorithm import Samplealgorithm
import os

# 维护的参数-范围表（参数名、范围、min、max、精度（1、0.01）、单位（m、g、M、flag、list、k）
# 使用相对路径 Spark_conf_range_wordcount.xlsx == .\\Spark_conf_range_wordcount.xlsx
conf_range_table = "Spark_conf_range_wordcount.xlsx"
# 参数范围和精度，从参数范围表里面获取
sparkConfRangeDf = pd.read_excel(conf_range_table)
# 将SparkConf列设置为index的方法
sparkConfRangeDf.set_index('SparkConf', inplace=True)
# {'spark.executor.cores': {'Range': '1-4', 'min': 1.0, 'max': 4.0, 'pre': 1.0, 'unit': nan},
# 'spark.executor.instances': {'Range': '2-8', 'min': 2.0, 'max': 8.0, 'pre': 1.0, 'unit': nan}
# 将数据框数据转换为字典形式
confDict = sparkConfRangeDf.to_dict('index')
print(confDict)


# 将我们项目的参数范围文件转换为pbounds格式，传入给采样算法抽样
def rangecsv_to_pbounds():
    pbounds = {}
    # conf是参数名称,将参数名称和范围拼接成字典
    for conf in confDict:
        bounds = []
        bounds.append(confDict[conf]['min'])
        bounds.append(confDict[conf]['max'])
        # key = confDict[conf]
        pbounds[conf] = bounds
    return pbounds

# 格式化参数配置：精度、单位等（在适应度函数中使用，随机生成的参数将其格式化后放入配置文件中实际运行，记录执行时间）
# 遗传算法随机生成的值都是实数值，根据配置对应range表格中的精度，将值处理为配置参数可以运行的值
def formatConf(conf, value):
    res = ''
    # 1. 处理精度
    # s、m、g、M、flag、list设置的精度都是1
    if confDict[conf]['pre'] == 1:
        # round():返回浮点数x的四舍五入值
        res = round(value)
    # 浮点型设置的精度是0.01
    elif confDict[conf]['pre'] == 0.01:
        res = round(value, 2)
    # 2. 添加单位(处理带单位的数据、flag和list数据)
    if not pd.isna(confDict[conf]['unit']):
        # 布尔值 false\true
        if confDict[conf]['unit'] == 'flag':
            res = str(bool(res)).lower()
        # 列表形式的参数（spark.serializer、spark.io.compression.codec等）
        elif confDict[conf]['unit'] == 'list':
            rangeList = confDict[conf]['Range'].split(' ')
            # res = 1就获取列表中第二个值
            res = rangeList[int(res)]
        # 给数字添加单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res

# 传入一个样本,生成对应的配置文件
def gen_conf_file(sample, configNum):
    # # 打开配置文件模板
    # fTemp = open("E:\\Desktop\\temp.config", 'r')
    # # 复制模板，并追加配置，新的配置文件命名为new_1.config/new_2.config....new_99.config
    # 如果new_config下已有配置文件，会自动覆盖旧的配置文件
    try:
        fNew = open("E:\\Desktop\\new_config\\new_" + str(configNum) + ".config", 'a+')
    except FileNotFoundError:
        # 如果new_config文件夹不存在，会自动创建文件夹（makedirs多层创建目录，mkdir只创建一层目录）
        os.makedirs('E:\\Desktop\\new_config\\')
        fNew = open("E:\\Desktop\\new_config\\new_" + str(configNum) + ".config", 'a+')
    # # 将文件从fTemp复制到fNew中，以块的形式复制数据，缓冲区大小为1024（常用）
    # shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        # samples:一个样本（有一组配置参数）
        for key in sample.keys():
            fNew.write(' ')
            # 读取第i个重要参数名称
            fNew.write(key)
            fNew.write('\t')
            # 将遗传算法随机生成的第i个重要参数值转换为可以在配置文件中运行的格式
            fNew.write(formatConf(key, sample[key]))
            fNew.write('\n')
    finally:
        fNew.close()

if __name__ == '__main__':
    # 将我们的csv文件转为pbounds格式 ---- 字典 {{参数1，[l1, up1]}, {参数2，[l2, up2]},....{参数n：[ln, upn]}}
    pbounds = rangecsv_to_pbounds()

    sa = Samplealgorithm(pbounds)

    D = sa._dim # 两个参数
    N = 100 # LHS层数为30层（将范围划分为30份）
    bounds = sa._bounds  # 参数的边界范围
    keys = sa._keys
    kind = 'random'
    # kind = 'lhs'
    # draw_sample(D, bounds, keys,  N, kind)
    lhs_samples = sa.res(D, bounds, keys, N, 'lhs')
    rd_samples = sa.res(D, bounds, keys, N, 'random')
    print(lhs_samples[0].keys())
    # print('-------------lhs samples-----------')
    # sa.output_sanples(lhs_samples)
    # print('-------------rd samples-----------')
    # sa.output_sanples(rd_samples)

    # 对每一个样本生成一个配置文件
    configNum = 0
    for sample in lhs_samples:
        gen_conf_file(sample, configNum)
        configNum += 1
