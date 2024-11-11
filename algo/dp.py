"""动态规划法实现TSP"""
import math

import numpy as np


class TSP:
    def __init__(self, v):
        self.v = v
        self.n = n = v.shape[0]
        self.metrix = np.zeros((n, n))  # 城市间距离矩阵
        for i in range(n):
            for j in range(n):
                self.metrix[i, j] = math.sqrt(np.sum((v[i, :] - v[j, :]) ** 2))
        self.path = np.ones((2 ** (n + 1), n))
        self.dp = np.ones((2 ** (n + 1), n)) * -1
        self.init_point = 0
        self.s = 0
        for i in range(1, n + 1):
            self.s = self.s | (1 << i)

    def run(self):
        distance = self.tsp(
            self.s, self.init_point, 0,
            self.n, self.dp, self.metrix, self.path
        )

        s = 0b11111110
        init = 0
        num = 0
        # print(distance)
        sequence = []
        while True:
            sequence.append(init)
            init = int(self.path[s][init])
            s = s & (~(1 << init))
            num += 1
            if num > self.n-1:
                break
        sequence.append(0)
        # self.plot(sequence)
        return sequence, distance

    def plot(self, sequence):
        import matplotlib.pyplot as plt

        print(sequence)
        fig = plt.figure()
        for i, seq in enumerate(sequence[:-1]):
            x1 = self.v[seq]
            x2 = self.v[sequence[i + 1]]
            plt.plot([x1[0], x2[0]], [x1[1], x2[1]], label='{}'.format(i))
            plt.text(x1[0], x1[1], '{}'.format(seq))
        plt.legend()
        plt.show()

    def tsp(self, s, init, num, n, dp, dist, path):
        if dp[s][init] != -1:
            return dp[s][init]

        if s == (1 << n):
            return dist[0][init]

        sum_path = 1000000000
        for i in range(n):
            if s & (1 << i):
                m = self.tsp(s & (~(1 << i)), i, num + 1, n, dp, dist, path) + dist[i][init]
                if m < sum_path:
                    sum_path = m
                    path[s][init] = i
        dp[s][init] = sum_path
        return dp[s][init]
