import warnings
import time

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    # 获取订阅者：获取_events中的event事件
    def get_subscribers(self, event):
        return self._events[event]

    # 订阅
    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            # 返回对象subscriber的update属性值
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    # 取消订阅
    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    # 分派
    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)

# BayesianOptimization 继承自 Observable类
class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        # 如果在初始化贝叶斯优化对象时，制定了范围收缩方法，则用当前的_space传给_bounds_transformer初始化
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        # 用父类的初始化方法来初始化继承的属性.也就是说,子类继承了父类的所有属性和方法
        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            print('sample为空执行随机生成一个样本点')
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    # 生成初始样本点，放入_queue中
    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            # 最少得有一个初始样本
            init_points = max(init_points, 1)

        '''
            注释代码：随机生成样本代码(随机抽样一次只生成一个样本）
            2021/12/29 19:17
        '''
        # for _ in range(init_points):
        #     # 随机生成初始样本，并放入_queue中
        #     self._queue.add(self._space.random_sample())
        '''
            新增代码：lhs生成样本代码（拉丁超立方根据初始的init_points大小，一次生成所有的init_points样本）
            2021/12/29 19:17
        '''
        lhsample = self._space.lhs_sample(init_points)
        for l in lhsample:
            # print(l.ravel())
            self._queue.add(l.ravel())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    # 开始正式的贝叶斯优化过程（找到最大值）
    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        # 记录搜索算法开始时间
        start_time = time.time()

        # 根据init_points值，随机生成init_points个初始样本点，将初始样本点放入_queue中
        self._prime_queue(init_points)
        # 设置gp的参数
        self.set_gp_params(**gp_params)

        # 实例化aquisition function，用于找到下一个最好的点
        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        # 判断是否达到迭代次数
        iteration = 0
        # 判断算法是否收敛
        cur_target = 0
        # 先输出_queue中的初始随机生成样本，后输出迭代过程中，贝叶斯根据已有样本集合找到的每一次最优可能的样本点
        while not self._queue.empty or iteration < n_iter:
            try:
                # 获取 _queue 中的下一个数 （_queue中存放的是init样本），输出init样本的时候，iteration一直为0
                x_probe = next(self._queue)
            # 直到超出了_queue个数（init样本输出完毕），报错，执行报错后的代码
            except StopIteration:
                # update_params：迭代次数+1， 并更新kappa参数
                util.update_params()
                # suggest：通过aquisition function找到下一个最有可能的点（init样本生成完成后，根据已有样本集合找到最有可能的样本点）
                x_probe = self.suggest(util)
                iteration += 1
            # 获取x_probe 对应的target
            self.probe(x_probe, lazy=False)

            # 如果设置了_bounds_transformer，调用transform收缩每个参数的范围
            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

            '''
               注释代码：以前自己加的判断迭代收敛的方法
               2021/12/29 19:17
            '''
            # # 判断是否达到收敛标准
            # pre_target = cur_target
            # # 获取x_probe 对应的target，_space.probe会返回target值
            # cur_target = self._space.probe(x_probe)
            # if abs(cur_target - pre_target) / abs(cur_target) < 0.0001:
            #     print('达到收敛要求,不再继续搜索新的样本。最后两次迭代找到的样本点target差值为：' + str(abs(cur_target - pre_target) / abs(cur_target)))
            #     break

        # 记录搜索算法结束时间
        end_time = time.time()
        print(str(int(end_time - start_time)) + 's')  # 秒级时间戳
        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)
