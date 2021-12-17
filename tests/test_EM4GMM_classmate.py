from src.em import EM4GMM

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy.stats import multivariate_normal as mul_norm

    np.random.seed(100)  # 设定随机数种子

    data = pd.read_csv("tests/height.csv")  # 读取数据

    dataset = np.array(data['身高']).reshape(-1, 1)  # 同学身高数据集
    init_pro = [0.6667, 0.3334]
    init_mean = [[175], [162]]  # 先验：男生比女生高
    init_cov = [[[10]], [[10]]]
    iteration = 2000
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

    # 根据条件概率计算某个样本属于哪一类
    theta = solution.theta
    pro_male = mul_norm.pdf(dataset, mean=theta['mean'][0], cov=theta['cov'][0])
    pro_female = mul_norm.pdf(dataset, mean=theta['mean'][1], cov=theta['cov'][1])
    genders = ['男' if x >= y else '女' for x, y in zip(pro_male, pro_female)]
    data['预测性别'] = genders
    data.to_csv('tests/height.csv', index=None)

    print('性别判断结果：', genders)
