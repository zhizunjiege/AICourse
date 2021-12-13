from src.em import EM

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(0)  # 设定随机数种子

    data = pd.read_csv("tests/groundtruth.csv")  # 读取数据

    def distri_func(x, z, u):
        if z[0] == 'm':
            p, m, s = u[0], u[1], u[3]
        else:
            p, m, s = 1 - u[0], u[2], u[4]
        return float(p * 1 / np.sqrt(2 * np.pi) / s * np.exp(-(x[0] - m)**2 / 2 / s**2))  # 先验：已知联合分布为高斯分布

    dataset = np.array(data['height']).reshape(-1, 1)  # 同学身高数据集
    latent_space = [['m'], ['f']]  # 先验：分男女两类
    init_params = np.array([0.7, 175, 165, 5, 5])  # 先验：男生比女生高
    iteration = 1000
    tolerance = 1e-4
    bounds = [(0.01, 0.99)] + [(0.01, None)] * 4  # 参数必须有一个非负下界，不然计算log时会出现负数报错

    print('求解中...')

    solution = EM.solve(
        distri_func,
        dataset,
        latent_space,
        init_params,
        iteration,
        tolerance,
        bounds=bounds,
    )

    print('求解完毕.')

    solution.print()

    # 根据条件概率计算某个样本属于哪一类
    theta = solution.theta
    genders = ['male' if distri_func(x, 'm', theta) >= distri_func(x, 'f', theta) else 'female' for x in dataset]

    pd.DataFrame({'gender': genders}).to_csv('tests/estimation.csv', index=None)

    print('性别判断结果：', genders)
