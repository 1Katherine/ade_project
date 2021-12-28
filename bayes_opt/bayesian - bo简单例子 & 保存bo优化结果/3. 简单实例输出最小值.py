from bo_scode.bayesian_optimization import BayesianOptimization
# （将结果加上负号，原来是贝叶斯优化求最大值对应的参数值，加上负号以后最大值变成最小值。输出结果再把负号加回来）
# 按照最大值优化 ----> 按照最小值进行优化
# 指定要优化的函数 f
def black_box_function(x, y):
    return -(-x ** 2 - (y - 1) ** 2 + 1)

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}
# 实例化一个 BayesianOptimization 对象，该对象指定一个要优化的函数 f 及其相应的边界和边界。
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=50,
)

# print(optimizer.max['params']['x'])
# print(optimizer.max['target'])
print('-----输出最小值-----')
res = {
                'target': -optimizer.space.target.max(),
                'params': dict(
                    zip(optimizer.space.keys, optimizer.space.params[optimizer.space.target.argmax()])
                )
        }
print(str(res) + '\n')

# print('-----输出最小值-----')
# res = {
#                 'target': optimizer.space.target.min(),
#                 'params': dict(
#                     zip(optimizer.space.keys, optimizer.space.params[optimizer.space.target.argmin()])
#                 )
#         }
# print(res)
