from bo_scode.logger import JSONLogger
from bo_scode.event import Events
from bo_scode.bayesian_optimization import BayesianOptimization

'''实例化 BayesianOptimization 对象

1. 指定要优化的函数 f
2. 定义参数的搜索范围 pbounds
3. 实例化一个 BayesianOptimization 对象，该对象指定一个要优化的函数 f 及其相应的边界和边界。

'''
# 1. 指定要优化的函数 f
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

# 2. Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}
# 3. 实例化一个 BayesianOptimization 对象，该对象指定一个要优化的函数 f 及其相应的边界和边界。
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

'''实例化 observer  对象

1. 实例化一个 observer  对象
2. 将观察者对象绑定到优化器触发的特定事件 Events.OPTIMIZATION_STEP
3.优化结果会被保存在 ./logs.json

'''
# 1. 实例化一个 observer  对象
logger = JSONLogger(path="./logs.json")
# 2. 将观察者对象绑定到优化器触发的特定事件。
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# 3.优化结果会被保存在 ./logs.json
optimizer.maximize(
    init_points=2,
    n_iter=3,
)


'''加载优化进度：可以将保存的优化结果加载到新的 BayesianOptimization 对象中

1. 实例化新的 BayesianOptimization 对象，并设置 verbose > 0来跟踪优化的进展
2. 向新的优化器中加载了以前看到的点
3. 新优化器开启进一步优化过程

'''

from bayes_opt.util import load_logs

# 1. 实例化新的 BayesianOptimization 对象，并设置 verbose > 0来跟踪优化的进展
new_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={"x": (-2, 2), "y": (-2, 2)},
    verbose=2,
    random_state=7,
)

# 2. 向新的优化器中加载了以前看到的点
load_logs(new_optimizer, logs=["./logs.json"]);

# 3. 新优化器开启进一步优化过程
new_optimizer.maximize(
    init_points=2,
    n_iter=3,
)