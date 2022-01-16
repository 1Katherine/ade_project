#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   QMCEngine.py
@Author ：Yang 
@CreateTime :   2022/1/15 21:17
@Reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.QMCEngine.html
'''
from scipy.stats import qmc
class RandomEngine(qmc.QMCEngine):
    def __init__(self, d, seed=None):
        super().__init__(d=d, seed=seed)


    def random(self, n=1):
        self.num_generated += n
        return self.rng.random((n, self.d))


    def reset(self):
        super().__init__(d=self.d, seed=self.rng_seed)
        return self


    def fast_forward(self, n):
        self.random(n)
        return self

if __name__ == '__main__':
    engine = RandomEngine(2)
    sample = engine.random(5)
    print(sample)
    # print(sample[:, 0], sample[:, 1])

    _ = engine.reset()
    sample = engine.random(5)
    print(sample)
