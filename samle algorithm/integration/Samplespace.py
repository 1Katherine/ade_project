import numpy as np

class SampleSpace:
    def __init__(self, pbounds):
        self._pbounds = pbounds
        # 对key进行排序，并返回key值
        self._keys = sorted(pbounds)

        # 创建参数范围数组(按照key排序后的参数范围数组）
        self._bounds = np.array(
            # pbounds.items()为待排序对象，按照待排序对象的第一维(key参数名)进行排序。
            # item为遍历的{参数名：(lower,upper)} item[1]是参数对应的范围上下界
            [item[1] for item in sorted(self._pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        # 参数的个数
        self._dim = len(self._keys)
        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self._dim))

    # 将样本点的参数值字典转为参数数组
    def params_to_array(self, params):
        return np.asarray([params[key] for key in self._keys])

    # 将参数名称和参数值封装成字典形式
    def array_to_params(self, x):
        return dict(zip(self._keys, x))

    # 将参数数组转为一维矩阵
    def _as_array(self, x):
        try:
            # 将数组转成矩阵形式
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)
        # 将x转为一维
        x = x.ravel()
        return x


    @property
    def params(self):
        return self.params

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds


if __name__ == '__main__':
    pbounds = {'n_estimators': (10, 250),
                    'min_samples_split': (2, 25),
                    'max_features': (0.1, 0.999),
                    'max_depth': (5, 15)}
    sp = SampleSpace(pbounds)

    sample = {'n_estimators': 100,
                   'min_samples_split': 15,
                   'max_features': 0.555,
                   'max_depth': 10}

    array = sp.params_to_array(sample)
    x = sp._as_array(array)
    print(type(array))
    print(type(x))