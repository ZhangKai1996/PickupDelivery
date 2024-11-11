import math

import numpy as np
import matplotlib.pyplot as plt


def coord2state(coord, size):
    return coord[0] * size + coord[1]


def state2coord(state, size):
    return [state // size, state % size]


def dense_reward(coord, target):
    return -math.sqrt(
        math.pow(coord[0] - target[0], 2) +
        math.pow(coord[1] - target[1], 2)
    )


def sparse_reward(coord, target):
    if coord[0] == target[0] and coord[1] == target[1]:
        return 1.0
    return 0.0


def plot(fig, size, reward, row=0, gamma=.95):
    ax1 = fig.add_subplot(2, 4, 1+row*4)
    ax2 = fig.add_subplot(2, 4, 2+row*4, projection='3d')
    ax3 = fig.add_subplot(2, 4, 3+row*4, projection='3d')
    ax4 = fig.add_subplot(2, 4, 4+row*4, projection='3d')

    number = size
    target = [15.0, 15.0]
    size_ = size * size
    x = np.linspace(0, size_ - 1, size_)
    r = []
    for i in range(size_):
        coord = state2coord(x[i], size)
        r.append(reward(coord, target))
    ax1.plot(x, r)
    r = sorted(r)
    ax1.plot(x, r)

    x = np.linspace(0, size - 1, number)
    y = np.linspace(0, size - 1, number)
    x_grid, y_grid = np.meshgrid(x, y)
    r = np.zeros((number, number))
    for i in range(number):
        for j in range(number):
            r[i, j] = reward([x[i], y[j]], target)
    ax2.plot_surface(x_grid, y_grid, r, cmap='rainbow')

    number = size_
    s1 = np.linspace(0, size_ - 1, number)
    s2 = np.linspace(0, size_ - 1, number)
    s1_grid, s2_grid = np.meshgrid(s1, s2)
    v = np.zeros((number, number))
    for i in range(number):
        coord1 = state2coord(s1[i], size)
        r1 = reward(coord1, target)
        for j in range(number):
            coord2 = state2coord(s2[j], size)
            r2 = reward(coord2, target)
            v[i, j] = r1 + gamma * r2
    ax3.plot_surface(s1_grid, s2_grid, v, cmap='rainbow')
    v = np.sort(v, axis=0)
    v = np.sort(v, axis=1)
    ax4.plot_surface(s1_grid, s2_grid, v, cmap='rainbow')


def main():
    size = 30
    gamma = 0.95
    fig = plt.figure()  # 创建一个3D图形对象

    plot(fig, size, dense_reward, row=0, gamma=gamma)
    plot(fig, size, sparse_reward, row=1, gamma=gamma)
    # ax1 = fig.add_subplot(2, 4, 1)
    # ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    # ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    # ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    #
    # number = size
    # target = [5.0, 5.0]
    # size_ = size * size
    # x = np.linspace(0, size_ - 1, size_)
    # r = []
    # for i in range(size_):
    #     coord = state2coord(x[i], size)
    #     r.append(reward(coord, target))
    # ax1.plot(x, r)
    # r = sorted(r)
    # ax1.plot(x, r)
    #
    # x = np.linspace(0, size - 1, number)
    # y = np.linspace(0, size - 1, number)
    # x_grid, y_grid = np.meshgrid(x, y)
    # r = np.zeros((number, number))
    # for i in range(number):
    #     for j in range(number):
    #         r[i, j] = reward([x[i], y[j]], target)
    # ax2.plot_surface(x_grid, y_grid, r, cmap='rainbow')
    #
    # number = size_
    # s1 = np.linspace(0, size_ - 1, number)
    # s2 = np.linspace(0, size_ - 1, number)
    # s1_grid, s2_grid = np.meshgrid(s1, s2)
    # v = np.zeros((number, number))
    # for i in range(number):
    #     coord1 = state2coord(s1[i], size)
    #     r1 = reward(coord1, target)
    #     for j in range(number):
    #         coord2 = state2coord(s2[j], size)
    #         r2 = reward(coord2, target)
    #         v[i, j] = r1 + gamma * r2
    # ax3.plot_surface(s1_grid, s2_grid, v, cmap='rainbow')
    # v = np.sort(v, axis=0)
    # v = np.sort(v, axis=1)
    # ax4.plot_surface(s1_grid, s2_grid, v, cmap='rainbow')

    # s1 = np.linspace(0, size_ - 1, number)
    # s2 = np.linspace(0, size_ - 1, number)
    # s3 = np.linspace(0, size_ - 1, number)
    # s1_grid, s2_grid = np.meshgrid(s1, s2)
    # v = np.zeros((number, number, number))
    # for i in range(number):
    #     r1 = reward(state2coord(s1[i], size), target)
    #     for j in range(number):
    #         r2 = reward(state2coord(s2[j], size), target)
    #         for k in range(number):
    #             r3 = reward(state2coord(s3[k], size), target)
    #             v[i, j, k] = r1 + gamma * r2 + gamma * gamma * r3

    # for column in range(4):
    #     ax = fig.add_subplot(2, 4, 4+column+1, projection='3d')
    #     v_ = v[:, :, column]
    #     v_ = np.sort(v_, axis=0)
    #     v_ = np.sort(v_, axis=1)
    #     ax.plot_surface(s1_grid, s2_grid, v_, cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    main()
