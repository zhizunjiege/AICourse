import math
import time
from typing import Callable, Iterable

from scipy.optimize import minimize


class Solution:
    '''求解所得信息的包装类.'''
    def __init__(self, success: bool, theta: Iterable, time: float, iter: int, msg: str) -> None:
        self.success = success
        self.theta = theta
        self.time = time
        self.iter = iter
        self.msg = msg


class EM:
    '''EM算法类，只有一个静态方法可以调用.'''
    def __init__(self) -> None:
        pass

    @staticmethod
    def solve(
        distri_func: Callable[[Iterable, Iterable, Iterable], float],
        dataset: Iterable[Iterable],
        latent_space: Iterable,
        init_params: Iterable,
        iteration: int,
        tolerance: float = 1e-8,
        **kargs,
    ) -> Solution:
        '''使用EM算法估计参数.

        Args:
            distri_func: 在一组参数下，求观测数据和隐变量的联合分布.
            dataset: 原始数据.
            latent_space: 隐变量的样本空间.
            init_params: 初始参数值.
            iteration: 最大迭代次数.
            tolerance: 浮点数容差.
            kargs: scipy.optimize.minimize所需的其他参数.

        Returns:
            求解所得信息.
        '''

        u = init_params

        def objective_func(v):
            ret = 0
            for x in dataset:
                sum_qi = 0
                sum_qipi = 0
                for z in latent_space:
                    qi = distri_func(x, z, u)
                    sum_qi += qi
                    pi = distri_func(x, z, v)
                    sum_qipi += qi * math.log(pi)
                ret += sum_qipi / sum_qi
            return -ret

        t1 = time.time()
        for j in range(iteration):
            res = minimize(objective_func, u, **kargs)
            if (not res.success or max([abs(thetai - ui) for thetai, ui in zip(res.x, u)]) < tolerance):
                break
            u = res.x
        t2 = time.time()
        return Solution(res.success, res.x, t2 - t1, j + 1, res.message)
