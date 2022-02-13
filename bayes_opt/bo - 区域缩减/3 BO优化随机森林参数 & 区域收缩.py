from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bo_scode import BayesianOptimization
from bo_scode import SequentialDomainReductionTransformer
import matplotlib.pyplot as plt

'''
1. 确定 target ： 传入参数 x ，可以得到 y
2. 设置参数 x 的范围 bounds
3. 实例化一个bo优化器：指定target 和 bounds
4. 开始优化 maximize()
'''


class BO:
    # 定义顺序域降低转换器
    bounds_transformer = SequentialDomainReductionTransformer()

    pbounds = {'n_estimators': (10, 250),
               'min_samples_split': (2, 25),
               'max_features': (0.1, 0.999),
               'max_depth': (5, 15)}

    def __init__(
            self,
            init_points: int = 2,
            n_iter: int = 10
    ) -> None:
        self.init_points = init_points
        self.n_iter = n_iter

    # 训练贝叶斯分类器，获获得交叉验证的结果
    # 先要定义一个目标函数。比如此时，函数输入为随机森林的所有参数，输出为模型交叉验证5次的AUC均值，作为我们的目标函数。
    # 因为bayes_opt库只支持最大值，所以最后的输出如果是越小越好，那么需要在前面加上负号，以转为最大值。
    def black_function(self, n_estimators, min_samples_split, max_features, max_depth):
        val = cross_val_score(
            RandomForestClassifier(n_estimators=int(n_estimators),
                                   min_samples_split=int(min_samples_split),
                                   max_features=min(max_features, 0.999),  # float
                                   max_depth=int(max_depth),
                                   random_state=2
                                   ),
            x, y, scoring='roc_auc', cv=6
        ).mean()
        return val

    def simple_bo(self):
        print("------- 开始简单的贝叶斯优化 -------")
        # 建立优化对象 里面的第一个参数是我们的优化目标函数，
        # 第二个参数是我们所需要输入的超参数名称，以及其范围。超参数名称必须和目标函数的输入名称一一对应。
        rf_bo = BayesianOptimization(
            f=self.black_function,
            pbounds=self.pbounds,
            random_state=990,
        )

        # 开始优化
        # rf_bo.maximize()
        # gp_param={'kernel':None}
        rf_bo.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
            acq='ei'
        )
        # 输出最大值
        print(rf_bo.max)
        return rf_bo

    def mutating_bo(self):
        # 使用转换器
        print("------- 使用转化器缩减参数范围 -------")

        # 优化问题，指定 bound_transformer 变量
        mutating_optimizer = BayesianOptimization(
            f=self.black_function,
            pbounds=self.pbounds,
            random_state=222,
            bounds_transformer=self.bounds_transformer
        )

        # 开始优化
        mutating_optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
            acq='ei'
        )
        print(mutating_optimizer.max)
        return mutating_optimizer

    # 画出贝叶斯优化过程中每次迭代选择的参数值对应的y值
    def draw_target(self, bo, mo):
        # 画图

        plt.plot(mo.space.target, label='Mutated Optimizer')
        plt.plot(bo.space.target, label='rf_bo')
        plt.legend()
        plt.show()

    # 画出所有参数范围的收缩过程
    def draw_btrans(self, mo):
        '''
            画出其中一个变量的实际收缩范围
        '''
        # example x-bound shrinking
        for i in range(4):
            x_min_bound = [b[i][0] for b in self.bounds_transformer.bounds]
            x_max_bound = [b[i][1] for b in self.bounds_transformer.bounds]
            x = [x[i] for x in mo.space.params]

            plt.title(mo.space.keys[i])
            # print(mutating_optimizer.space.target[i])
            plt.plot(x_min_bound[1:], label='x lower bound')
            plt.plot(x_max_bound[1:], label='x upper bound')
            plt.plot(x[1:], label='x')
            plt.legend()
            plt.show()


if __name__ == "__main__":
    # 生成 1000个分类样本（10个特征，2个分类）
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

    bo = BO(init_points=2, n_iter=20)
    # 基础的贝叶斯优化器
    rf_bo = bo.simple_bo()
    # 带范围转换器的贝叶斯优化器
    mutat_bo = bo.mutating_bo()
    # 画target
    # bo.draw_target(rf_bo, mutat_bo)
    # 画出参数收缩过程
    bo.draw_btrans(mutat_bo)

