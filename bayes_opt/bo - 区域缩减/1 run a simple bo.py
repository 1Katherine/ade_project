import numpy as np
from bo_scode.bayesian_optimization import BayesianOptimization
from bo_scode import SequentialDomainReductionTransformer
# from bayes_opt import SequentialDomainReductionTransformer

def black_box_function(**kwargs):
    x = np.fromiter(kwargs.values(), dtype=float)
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -1.0 * (-20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e)

def black_box_function2(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

# Bounded region of parameter space
pbounds = {'x': (-5, 5), 'y': (-2, 15)}

bounds_transformer = SequentialDomainReductionTransformer()

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
    bounds_transformer=bounds_transformer
)

optimizer.maximize(
    init_points=2,
    n_iter=60,
)

print(optimizer.max)