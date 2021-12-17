# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:48:15 2021

@author: admim
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pylab as plt


# EM算法
def EM(u1, sig1, u2, sig2, pi, x):
    # E-step
    # 权重*后验概率
    p1 = pi * stats.norm.pdf(x, u1, sig1)
    p2 = (1 - pi) * stats.norm.pdf(x, u2, sig2)
    w1 = p1 / (p1 + p2)
    w2 = p2 / (p1 + p2)

    # M-step
    pi = np.sum(w1) / len(x)
    u1 = np.sum(w1 * x) / np.sum(w1)
    u2 = np.sum(w2 * x) / np.sum(w2)
    sig1 = np.sqrt(np.sum(w1 * np.square(x - u1)) / np.sum(w1))
    sig2 = np.sqrt(np.sum(w2 * np.square(x - u2)) / np.sum(w2))

    return u1, sig1, u2, sig2, pi


def main():
    # 1. 生成身高数据
    np.random.seed(100)
    # （1）给定二元高斯混合模型的参数
    female = np.random.normal(162, np.sqrt(10), 500)
    male = np.random.normal(175, np.sqrt(10), 1000)

    # （2）男女生身高分布直方图
    female_h = pd.Series(female).hist(bins=int(5 * (np.max(female) - np.min(female))), color='r')
    male_h = pd.Series(male).hist(bins=int(5 * (np.max(male) - np.min(male))), color='b')
    female_h.plot()
    male_h.plot()
    plt.title('Random Heights Distribution Histogram of 500 females and 1000 males')
    plt.xlabel('height/cm')
    plt.ylabel('number')
    plt.show()

    # （3）混合男女身高数据
    mix = list(female)
    mix.extend(male)
    mix = np.array(mix)
    mix_p = pd.Series(mix).hist(bins=int(5 * (np.max(mix) - np.min(mix))), color='g')
    mix_p.plot()
    plt.title('Random Heights Distribution Histogram of 1500 people')
    plt.xlabel('height/cm')
    plt.ylabel('number')
    plt.show()

    # 2. 利用EM算法估计参数
    # （1）初始化参数
    u1 = 165
    sig1 = 10
    u2 = 180
    sig2 = 10
    pi = 0.2

    # （2）EM算法
    for i in range(100):
        u1, sig1, u2, sig2, pi = EM(u1, sig1, u2, sig2, pi, mix)
    print(u1, sig1, u2, sig2, pi)

    # （3）按预测模型绘图
    x = np.linspace(140, 210, 500)
    female_pre = stats.norm.pdf(x, u1, sig1)
    male_pre = stats.norm.pdf(x, u2, sig2)
    mix_pre = pi * female_pre + (1 - pi) * male_pre
    plt.plot(x, female_pre, color='r')
    plt.plot(x, male_pre, color='b')
    plt.plot(x, mix_pre, color='k')
    plt.title('Probability density curve')
    plt.legend(["female", "male", "mixed"], loc='upper right')
    plt.xlabel('height/cm')
    plt.ylabel('Probability')  # 坐标轴设置
    plt.show()

    # 3. 读入真实身高数据
    # （1）读入数据
    INPUT_FILE = 'tests/height.csv'
    height_statistic = pd.read_csv(INPUT_FILE, encoding='utf-8')

    real_mix_num = height_statistic.shape[0]
    height_np = height_statistic.to_numpy()

    real_female = []
    real_male = []
    real_mix = []

    for i in range(real_mix_num):
        if height_np[i, 0] == '女':
            real_female.append(height_np[i, 1])
        else:
            real_male.append(height_np[i, 1])
        real_mix.append(height_np[i, 1])

    real_female = np.array(real_female)
    real_male = np.array(real_male)
    real_mix = np.array(real_mix)

    # （2）绘制直方图
    real_female_h = pd.Series(real_female).hist(bins=int(1 * (np.max(real_female) - np.min(real_female))), color='r')
    real_male_h = pd.Series(real_male).hist(bins=int(1 * (np.max(real_male) - np.min(real_male))), color='b')
    real_female_h.plot()
    real_male_h.plot()
    plt.title('Histogram of Real Height Distribution')
    plt.xlabel('height/cm')
    plt.ylabel('number')
    plt.show()

    real_mix_h = pd.Series(real_mix).hist(bins=int(1 * (np.max(real_mix) - np.min(real_mix))), color='g')
    real_mix_h.plot()
    plt.title('Histogram of Real Mixed Height Distribution')
    plt.xlabel('height/cm')
    plt.ylabel('number')
    plt.show()

    # 4. 利用EM算法估计参数
    # （1）EM算法
    real_u1, real_sig1, real_u2, real_sig2, real_pi = 162, 3.2, 175, 3.2, 0.33
    for i in range(2000):
        real_u1, real_sig1, real_u2, real_sig2, real_pi = EM(real_u1, real_sig1, real_u2, real_sig2, real_pi, real_mix)
    print(real_u1, real_sig1, real_u2, real_sig2, real_pi)

    # （2）按预测模型绘图
    real_female_pre = stats.norm.pdf(x, real_u1, real_sig1)
    real_male_pre = stats.norm.pdf(x, real_u2, real_sig2)
    real_mix_pre = real_pi * real_female_pre + (1 - real_pi) * real_male_pre
    plt.plot(x, real_female_pre, color='r')
    plt.plot(x, real_male_pre, color='b')
    plt.plot(x, real_mix_pre, color='k')
    plt.title('Probability density curve')
    plt.legend(["female", "male", "mixed"], loc='upper right')
    plt.xlabel('height/cm')
    plt.ylabel('Probability')  # 坐标轴设置
    plt.show()


if __name__ == "__main__":
    main()
