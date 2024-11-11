from math import floor
import numpy as np


class GenaTSP(object):
    def __init__(self, data, max_gen=200, size_pop=200, cross_prob=0.9, mut_prob=0.01, select_prob=0.8):
        self.data = data[0]  # 城市的左边数据
        self.labels = data[1]
        self.num = len(self.data)  # 城市个数 对应染色体长度

        self.max_gen = max_gen  # 最大迭代次数
        self.size_pop = size_pop  # 群体个数
        self.cross_prob = cross_prob  # 交叉概率
        self.mut_prob = mut_prob  # 变异概率
        self.select_prob = select_prob  # 选择概率

        self.matrix_distance = self.matrix_dis()
        # 距离矩阵n*n, 第[i,j]个元素表示城市i到j距离matrix_dis函数见下文
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        # 通过选择概率确定子代的选择个数
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop, self.num)
        self.sub_sel = np.array([0] * self.select_num * self.num).reshape(self.select_num, self.num)
        # 父代和子代群体的初始化（不直接用np.zeros是为了保证单个染色体的编码为整数，np.zeros对应的数据类型为浮点型）
        self.fitness = np.zeros(self.size_pop)
        # 存储群体中每个染色体的路径总长度，对应单个染色体的适应度就是其倒数
        self.best_fit = []
        self.best_path = []
        # 保存每一步的群体的最优路径和距离

    def matrix_dis(self):
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i + 1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
        return res

    def rand_chrom(self):
        rand_ch = np.array(range(self.num))
        for i in range(self.size_pop):
            np.random.shuffle(rand_ch)
            self.chrom[i, :] = rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

    def comp_fit(self, one_path):
        res = 0
        check_list = {}
        for i in range(self.num - 1):
            p1 = one_path[i]
            number, status = self.labels[p1]
            if i in [0, 1] and number != 0:
                res = 1000
                break
            if status == 'D':
                if number not in check_list.keys():
                    res = 1000
                    break
            else:
                check_list[number] = 0
            res += self.matrix_distance[p1, one_path[i + 1]]
        return res

    def select_sub(self):
        fit = 1. / self.fitness  # 适应度函数
        sum_fit = np.cumsum(fit)
        pick = sum_fit[-1] / self.select_num * (np.random.rand() + np.array(range(self.select_num)))
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if sum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]

    def cross_sub(self):
        if self.select_num % 2 == 0:
            num = range(0, self.select_num, 2)
        else:
            num = range(0, self.select_num - 1, 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.inter_cross(self.sub_sel[i, :],
                                                                              self.sub_sel[i + 1, :])

    def inter_cross(self, ind_a, ind_b):
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        left, right = min(r1, r2), max(r1, r2)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b

    def mutation_sub(self):
        for i in range(self.select_num):
            if np.random.rand() <= self.mut_prob:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num)
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]

    def reverse_sub(self):
        for i in range(self.select_num):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r2 == r1:
                r2 = np.random.randint(self.num)
            left, right = min(r1, r2), max(r1, r2)
            sel = self.sub_sel[i, :].copy()

            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):
                self.sub_sel[i, :] = sel

    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num], :] = self.sub_sel


def run(data):
    path_short = GenaTSP(data)  # 根据位置坐标，生成一个遗传算法类
    path_short.rand_chrom()  # 初始化父类
    # 循环迭代遗传过程
    sequence, dist = [], 0
    for i in range(path_short.max_gen):
        path_short.select_sub()  # 选择子代
        path_short.cross_sub()  # 交叉
        path_short.mutation_sub()  # 变异
        path_short.reverse_sub()  # 进化逆转
        path_short.reins()  # 子代插入
        # 重新计算新群体的距离值
        for j in range(path_short.size_pop):
            path_short.fitness[j] = path_short.comp_fit(path_short.chrom[j, :])
        # 每隔三十步显示当前群体的最优路径
        index = path_short.fitness.argmin()
        # 存储每一步的最优路径及距离
        dist = path_short.fitness[index]
        sequence = path_short.chrom[index, :]
        path_short.best_fit.append(dist)
        path_short.best_path.append(sequence)
    return sequence, dist  # 返回遗传算法结果类
