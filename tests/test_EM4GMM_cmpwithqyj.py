from src.em import EM4GMM

if __name__ == "__main__":
    import numpy as np

    samples = 1500  # 数据量
    p, m1, m2, s1, s2 = 0.6667, 175, 162, np.sqrt(10), np.sqrt(10)  # 真实参数

    np.random.seed(100)  # 设定随机数种子

    female = np.random.normal(m2, s2, int(samples * (1 - p)))
    male = np.random.normal(m1, s1, int(samples * p))
    dataset = np.hstack([male, female]).reshape((-1, 1))  # 随机数据集
    init_pro = [0.8, 0.2]
    init_mean = [[180], [165]]  # 先验：男生比女生高
    init_cov = [[[10]], [[10]]]
    iteration = 100
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
