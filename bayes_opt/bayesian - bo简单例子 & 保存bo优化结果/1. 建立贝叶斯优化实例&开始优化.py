from bo_scode.bayesian_optimization import BayesianOptimization

# 指定要优化的函数 f
def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}
# 实例化一个 BayesianOptimization 对象，该对象指定一个要优化的函数 f 及其相应的边界和边界。
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

''' 开始进行贝叶斯优化
n_iter: 贝叶斯优化迭代次数
init_points：随机探索的步骤
最终输出结果行数为：n_iter + init_points
'''
optimizer.maximize(
    init_points=10,
    n_iter=6,
)

# print(optimizer.max['params']['x'])
# print(optimizer.max['target'])
print('-----默认输出最大值-----')
print(str(optimizer.max) + '\n')

# 如果想要计算min，则给所有的y加上负号，-y的最大值就是最小值
# print(~optimizer.max['target']+1)
print(-optimizer.max['target'])
# 改变边界范围
print('---------------改变边界范围-------------')
optimizer.set_bounds(new_bounds={"x": (-2, 3)})

optimizer.maximize(
    init_points=0,
    n_iter=5,
)

