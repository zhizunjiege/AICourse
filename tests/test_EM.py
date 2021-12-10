from src.em import EM

if __name__ == "__main__":
    import numpy as np

    samples = 1000
    p, m1, m2, s1, s2 = 0.7, 175, 165, 5, 5

    np.random.seed(0)

    c = 1 / np.sqrt(2 * np.pi)

    def distri_func(x, z, u):
        if z[0] == 'm':
            p, m, s = u[0], u[1], u[3]
        else:
            p, m, s = 1 - u[0], u[2], u[4]
        return float(p * c / s * np.exp(-(x[0] - m)**2 / 2 / s**2))

    male = np.random.normal(m1, s1, int(samples * p))
    female = np.random.normal(m2, s2, int(samples * (1 - p)))
    dataset = np.hstack([male, female]).reshape((-1, 1))
    latent_space = [['m'], ['f']]
    init_params = np.array([0.5, 180, 160, 10, 10])
    iteration = 100
    tolerance = 1e-4
    bounds = [(0.01, 0.99)] + [(0.01, None)] * 4  # 参数必须有一个非负下界，不然计算log时会出现负数报错

    solution = EM.solve(
        distri_func,
        dataset,
        latent_space,
        init_params,
        iteration,
        tolerance,
        bounds=bounds,
    )

    print(solution.success)
    print(solution.theta)
    print(solution.time)
    print(solution.iter)
    print(solution.msg)
