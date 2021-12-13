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

    def print(self) -> None:
        print(f'求解{"成功" if self.success else "失败"}！', )
        print(f'目标参数：{self.theta}')
        print(f'求解耗时：{self.time:.2f} s')
        print(f'迭代次数：{self.iter}')
        print(f'附加信息：{self.msg}')


class EM:
    '''EM算法类，只有一个静态方法可以调用.'''
    def __init__(self) -> None:
        ...

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
            distri_func: 在一组参数下，观测数据和隐变量的联合分布函数.
            dataset: 原始数据集.
            latent_space: 隐变量的样本空间，只支持离散量.
            init_params: 初始参数值.
            iteration: 最大迭代次数.
            tolerance: 参数优化容差，两次迭代间参数绝对变化量小于该容差时停止迭代.
            kargs: scipy.optimize.minimize所需的其他参数.

        Returns:
            求解所得信息.
        '''

        u = init_params

        def objective_func(v):
            ret = 0
            for x in dataset:  # 对样本求期望
                sum_qi = 0
                sum_qipi = 0
                for z in latent_space:  # 对隐变量求期望
                    qi = distri_func(x, z, u)
                    pi = distri_func(x, z, v)
                    sum_qi += qi
                    sum_qipi += qi * math.log(pi)
                ret += sum_qipi / sum_qi
            return -ret  # 优化目标由极大值转极小值

        t1 = time.time()
        for j in range(iteration):
            res = minimize(objective_func, u, **kargs)  # 局部极小算法
            if (not res.success or max([abs(thetai - ui) for thetai, ui in zip(res.x, u)]) < tolerance):  # 提前停止条件
                break
            u = res.x
        t2 = time.time()
        return Solution(res.success, res.x, t2 - t1, j + 1, res.message)  # 返回解信息
