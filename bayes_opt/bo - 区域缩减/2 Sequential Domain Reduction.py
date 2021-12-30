import numpy as np
from bo_scode import BayesianOptimization
from bo_scode import SequentialDomainReductionTransformer
import matplotlib.pyplot as plt


def ackley(**kwargs):
    x = np.fromiter(kwargs.values(), dtype=float)
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -1.0 * (-20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e)


# 贝叶斯优化器，带bound_transformer
def mu_bo(pbounds, btrans):
    # 优化问题，指定 bound_transformer 变量
    mutating_optimizer = BayesianOptimization(
        f=ackley,
        pbounds=pbounds,
        random_state=1,
        bounds_transformer=btrans
    )

    # 开始优化
    mutating_optimizer.maximize(
        init_points=2,
        n_iter=60,
    )

    print(mutating_optimizer.max)
    return mutating_optimizer

# 标准的贝叶斯优化过程，没有参数范围收缩过程
def st_bo(pbounds):
    standard_optimizer = BayesianOptimization(
        f=ackley,
        pbounds=pbounds,
        random_state=1,
    )

    # 开始优化
    standard_optimizer.maximize(
        init_points=2,
        n_iter=60,
    )

    print(standard_optimizer.max)
    return standard_optimizer

# 画出贝叶斯优化过程中每次迭代选择的参数值对应的y值
def draw_mu_target(mu):
    plt.plot(mu.space.target, label='Mutated Optimizer')
    plt.legend()
    plt.show()

def draw_st_target(st):
    plt.plot(st.space.target, label='Standard Optimizer',color='g')
    plt.legend()
    plt.show()

# 画出第一个参数范围的收缩过程
def draw_btrans(mo):
    # example x-bound shrinking
    x_min_bound = [b[0][0] for b in bounds_transformer.bounds]
    x_max_bound = [b[0][1] for b in bounds_transformer.bounds]
    x = [x[0] for x in mo.space.params]

    plt.plot(x_min_bound[1:], label='x lower bound')
    plt.plot(x_max_bound[1:], label='x upper bound')
    plt.plot(x[1:], label='x')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pbounds = {'x': (-5, 5), 'y': (-5, 5)}

    # 定义顺序域降低转换器
    bounds_transformer = SequentialDomainReductionTransformer()

    mo = mu_bo(pbounds, bounds_transformer)
    st = st_bo(pbounds)
    draw_mu_target(mo)
    draw_btrans(mo)

    draw_st_target(st)