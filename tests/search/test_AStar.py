from src.search.a_star import StateBase, AStar

if __name__ == "__main__":
    import random

    import numpy as np

    # 定义状态类
    class State(StateBase):
        def __init__(self, matrix, coordinate):
            self.matrix = np.array(matrix)
            self.coordinate = coordinate

        def operate(self):
            children = []
            i, j = self.coordinate
            if i > 0:
                m = np.array(self.matrix)
                m[i, j], m[i - 1, j] = m[i - 1, j], m[i, j]
                children.append(State(m, (i - 1, j)))
            if i < 3:
                m = np.array(self.matrix)
                m[i, j], m[i + 1, j] = m[i + 1, j], m[i, j]
                children.append(State(m, (i + 1, j)))
            if j > 0:
                m = np.array(self.matrix)
                m[i, j], m[i, j - 1] = m[i, j - 1], m[i, j]
                children.append(State(m, (i, j - 1)))
            if j < 3:
                m = np.array(self.matrix)
                m[i, j], m[i, j + 1] = m[i, j + 1], m[i, j]
                children.append(State(m, (i, j + 1)))
            return children

        def evaluation(self, target, depth=0):
            h = 0
            for i in range(4):
                for j in range(4):
                    num = self.matrix[i, j]
                    if num == 16:
                        continue
                    q, r = num // 4, num % 4
                    if r == 0:
                        q -= 1
                        r = 4
                    r -= 1
                    h += abs(q - i) + abs(r - j)
            return depth + h

        def __str__(self) -> str:
            return str(self.matrix)

        def rand_gen(self, iter_times):
            '''在此状态基础上随机生成一个状态.'''
            random_state = self
            for _ in range(iter_times):
                children = random_state.operate()
                random_state = random.choice(children)
            return random_state

    # 初始状态
    init_state = State([
        [11, 9, 4, 15],
        [1, 3, 16, 12],
        [7, 5, 8, 6],
        [13, 2, 10, 14],
    ], (1, 2))
    # 目标状态
    target_state = State(np.arange(1, 17).reshape((4, 4)), (3, 3))
    # 求解
    solution = AStar.solve(init_state, target_state)
    # 输出结果
    print(f'求解{"成功" if solution.success else "失败"}！耗时 {solution.time_cost:.6f} 秒，共搜索 {solution.search_nodes} 个节点.')
    if solution.success:
        print(f'操作 {solution.steps} 步，解为：')
        for state in solution.solution_path:
            print(state)
            print()

    # 随机生成状态并求解
    # for i in range(10):
    #     random_state = target_state.rand_gen(100)
    #     solution = AStar.solve(random_state, target_state)

    #     print(f'求解{"成功" if solution.success else "失败"}！耗时 {solution.time_cost:.6f} 秒，共搜索 {solution.search_nodes} 个节点.')
    #     if solution.success:
    #         print(f'操作 {solution.steps} 步，解为：')
    #         for state in solution.solution_path:
    #             print(state)
    #             print()
