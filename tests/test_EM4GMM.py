from src.em import EM4GMM

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.stats import norm

    samples = 2000  # 数据量
    p, m1, m2, s1, s2 = 0.7, 175, 165, 3, 3  # 真实参数

    np.random.seed(0)  # 设定随机数种子

    male = np.random.normal(m1, s1, int(samples * p))
    female = np.random.normal(m2, s2, int(samples * (1 - p)))
    dataset = np.hstack([male, female]).reshape((-1, 1))  # 随机数据集
    init_pro = [0.5, 0.5]
    init_mean = [[180], [160]]  # 先验：男生比女生高
    init_cov = [[[36]], [[36]]]
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

    theta = solution.theta
    plot_x = np.linspace(150, 190, 100)
    plot_ytm = norm.pdf(plot_x, loc=m1, scale=s1)
    plot_ytf = norm.pdf(plot_x, loc=m2, scale=s2)
    plot_yt = p * plot_ytm + (1 - p) * plot_ytf
    plot_ypm = norm.pdf(plot_x, loc=theta["mean"][0][0], scale=np.sqrt(theta["cov"][0][0, 0]))
    plot_ypf = norm.pdf(plot_x, loc=theta["mean"][1][0], scale=np.sqrt(theta["cov"][1][0, 0]))
    plot_yp = theta["pro"][0] * plot_ypm + theta["pro"][1] * plot_ypf

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.subplot(2, 1, 1)
    plt.title('随机数据-真实概率密度函数')
    plt.ylabel('概率密度')
    plt.xlabel('身高/厘米')
    plt.plot(plot_x, plot_ytm)
    plt.plot(plot_x, plot_ytf)
    plt.plot(plot_x, plot_yt)
    plt.legend(['男生', '女生', '混合'])
    plt.subplot(2, 1, 2)
    plt.title('随机数据-预测概率密度函数')
    plt.ylabel('概率密度')
    plt.xlabel('身高/厘米')
    plt.plot(plot_x, plot_ypm)
    plt.plot(plot_x, plot_ypf)
    plt.plot(plot_x, plot_yp)
    plt.legend(['男生', '女生', '混合'])
    plt.show()
