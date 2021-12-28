import numpy as np
from .target_space import TargetSpace


class DomainTransformer():
    '''The base transformer class'''

    def __init__(self, **kwargs):
        pass

    def initialize(self, target_space: TargetSpace):
        raise NotImplementedError

    def transform(self, target_space: TargetSpace):
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        pass

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters"""
        # 初始的参数范围，第一行为第一个参数的下界上界，第二行为第二个参数的上下界。x*2矩阵
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]
        # previous_optimal ： 按行取均值（bounds一行为一个参数的上下界范围）
        # [参数1的上下界均值，参数2的上下界均值] 1*x矩阵（第一行第一个列为参数1的范围均值，第二列为参数2的范围均值）
        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        # current_optimal ： 按行取均值[参数1的上下界均值，参数2的上下界均值] 1*x矩阵（第一行第一个列为参数1的范围均值，第二列为参数2的范围均值）
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]
        # 1*x矩阵
        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r
        # 1*x矩阵
        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r
        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)
        # 计算r缩减系数：对应每一个参数的范围收缩系数 1*x矩阵（第一行第一个列为参数1的范围收缩系数，第二列为参数2的范围收缩系数）
        self.r = self.contraction_rate * self.r

    # 使用这一代的最佳样本的参数值，计算每个参数的收缩系数
    def _update(self, target_space: TargetSpace) -> None:
        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        # 获取样本空间中最大的target对应的那一行的参数值作为current_optimal
        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]

        # 计算current_d
        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)
        # 计算r收缩系数
        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        # 按行读取new_bounds，每一行代表一个参数值的范围上下界
        # variable为 1*2 矩阵
        for i, variable in enumerate(new_bounds):
            # global_bounds是原始的参数上下界
            # 如果这个参数值的新下界比原始的下界还小（防止新范围越界初始范围）
            if variable[0] < global_bounds[i, 0]:
                variable[0] = global_bounds[i, 0]
            # 如果这个参数值的新上界比原始的上界还大（防止新范围越界初始范围）
            if variable[1] > global_bounds[i, 1]:
                variable[1] = global_bounds[i, 1]

        return new_bounds

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    # 用最佳样本的参数值 +\-参数对应的收缩系数 --> 参数的上下界
    def transform(self, target_space: TargetSpace) -> dict:
        # 根据样本空间的最大值对应的参数，计算参数的收缩系数r
        self._update(target_space)
        # 获取新的边界范围,r是1*x矩阵,current_optimal是1*x矩阵
        # np是2*x的矩阵，转置后为x*2的矩阵，即每个参数对应的新的上下界
        # 参数1的新范围是[最好样本点在参数1的值-0.5r,参数1的值+0.5r]
        # 参数2的新范围是[最好样本点在参数2的值-0.5r,参数2的值+0.5r]
        # 参数n的新范围是[最好样本点在参数n的值-0.5r,参数n的值+0.5r]
        new_bounds = np.array(
            [
                # np 第一行为current_optimal（最好样本点的参数值全部 - 0.5 * self.r）
                self.current_optimal - 0.5 * self.r,
                # np 第二行为current_optimal（最好样本点的参数值全部 + 0.5 * self.r）
                self.current_optimal + 0.5 * self.r
            ]
        ).T
        # print('new_bounds')
        # print(new_bounds)
        self._trim(new_bounds, self.original_bounds)
        # 用于记录参数的范围变化过程
        self.bounds.append(new_bounds)
        # 将new_bounds作为参数搜索过程中的新上下界范围
        return self._create_bounds(target_space.keys, new_bounds)
