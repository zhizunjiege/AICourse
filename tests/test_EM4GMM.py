from src.em import EM4GMM

if __name__ == "__main__":
    import numpy as np

    samples = 10000  # 数据量
    p, m1, m2, s1, s2 = 0.7, 175, 165, 5, 5  # 真实参数

    np.random.seed(0)  # 设定随机数种子

    male = np.random.normal(m1, s1, int(samples * p))
    female = np.random.normal(m2, s2, int(samples * (1 - p)))
    dataset = np.hstack([male, female]).reshape((-1, 1))  # 随机数据集
    init_pro = [0.5, 0.5]
    init_mean = [[180], [160]]  # 先验：男生比女生高
    init_cov = [[[36]], [[36]]]
    iteration = 1000
    tolerance = 1e-8

    print('求解中...')

    solution = EM4GMM.solve(
        dataset,
        init_pro,
        init_mean,
        init_cov,
        iteration,
        tolerance,
    )

    print('求解完毕.')

    solution.print()
