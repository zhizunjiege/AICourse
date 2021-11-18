from abc import ABC, abstractmethod
import heapq
import time
from typing import List, Optional


class StateBase(ABC):
    '''状态的抽象基类.'''
    def __init__(self) -> None:
        ...

    @abstractmethod
    def operate(self) -> List["StateBase"]:
        '''由当前状态进行操作，得到子状态列表.

        Returns:
            子状态的列表.
        '''
        ...

    @abstractmethod
    def evaluation(self, target: "StateBase", depth: int) -> float:
        '''对当前状态进行评估，得到估价函数值.

        Args:
            target: 目标状态.
            depth: 当前状态对应节点的搜索深度.

        Returns:
            估价函数值.
        '''
        ...

    @abstractmethod
    def __str__(self) -> str:
        '''返回状态对象的唯一标识字符串表示，用于计算哈希值.

        Returns:
            唯一标识字符串.
        '''
        ...


class Node:
    '''对状态类进行简单包装的节点类.'''
    def __init__(self, state: StateBase, value: float, depth: int, parent: Optional["Node"]) -> None:
        self.state = state
        self.value = value
        self.depth = depth
        self.parent = parent

        self.hash = hash(str(state))
        self.removed = False

    def __lt__(self, other: "Node") -> bool:
        return self.value < other.value

    def __eq__(self, other: "Node") -> bool:
        return self.hash == other.hash


class Solution:
    '''求解所得信息的包装类.'''
    def __init__(self, success: bool, time_cost: float, search_nodes: int, target_node: Optional[Node]) -> None:
        self.success = success
        self.time_cost = time_cost
        self.search_nodes = search_nodes

        if success:
            solution_path = []
            node = target_node
            while node is not None:
                solution_path.append(node.state)
                node = node.parent
            self.solution_path = [n for n in reversed(solution_path)]
        else:
            self.solution_path = []
        self.steps = len(self.solution_path) - 1


class AStar:
    '''A*算法类，只有一个静态方法可以调用.'''
    def __init__(self) -> None:
        pass

    @staticmethod
    def solve(init_state: StateBase, target_state: StateBase) -> Solution:
        '''使用A*算法进行求解.

        Args:
            init_state: 初始状态.
            target_state: 目标状态.

        Returns:
            求解所得信息.
        '''
        # 生成初始节点和目标节点
        root = Node(init_state, init_state.evaluation(target_state, depth=0), 0, None)
        target = Node(target_state, 0, 0, None)
        # 如果初始状态与目标状态相同，直接返回
        if root == target:
            return Solution(True, 0, 0, target)
        # 生成所需数据结果
        open_table = [root]
        open_dict, closed_dict = {root.hash: root}, {}
        heapq.heapify(open_table)
        # 开始搜索解
        t1 = time.time()
        search_nodes = 1
        while len(open_table):
            cur_node = heapq.heappop(open_table)
            # 如果节点已被标记为removed，则跳过处理阶段
            # 这是一种基于heapq实现删除操作的方法，详见 https://docs.python.org/zh-cn/3.8/library/heapq.html
            if cur_node.removed:
                continue
            else:
                # 将节点从open表移除，并加入closed表
                del open_dict[cur_node.hash]
                closed_dict[cur_node.hash] = cur_node
            # 扩展生成子节点
            children = cur_node.state.operate()
            for child in children:
                depth = cur_node.depth + 1
                value = child.evaluation(target_state, depth=depth)
                child_node = Node(child, value, depth, cur_node)
                if child_node == target:
                    # 如果子节点状态等于目标状态，则找到解
                    t2 = time.time()
                    return Solution(True, t2 - t1, search_nodes + 1, child_node)
                elif cur_node.parent is not None and child_node == cur_node.parent:
                    # 如果子节点状态等于当前节点的父节点，则忽略
                    continue
                else:
                    hash = child_node.hash
                    if hash in open_dict:
                        # 如果子节点在open表中，比较其估价函数值以决定是否取代已有节点
                        exist_node = open_dict[hash]
                        if child_node.value < exist_node.value:
                            exist_node.removed = True
                            heapq.heappush(open_table, child_node)
                            open_dict[hash] = child_node
                    elif hash in closed_dict:
                        # 如果子节点在closed表中，比较其估价函数值以决定是否删除已有节点，并重新加入open表中
                        exist_node = closed_dict[hash]
                        if child_node.value < exist_node.value:
                            del closed_dict[hash]
                            heapq.heappush(open_table, child_node)
                            open_dict[hash] = child_node
                    else:
                        # 如果子节点即不在open表也不在closed表中，则直接加入open表
                        heapq.heappush(open_table, child_node)
                        open_dict[hash] = child_node
                        search_nodes += 1
        # 如果节点均已扩展仍没有找到解，则失败退出
        t2 = time.time()
        return Solution(False, t2 - t1, search_nodes, None)
