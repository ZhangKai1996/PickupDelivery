import math
import numpy as np


def one_hot(vector, num=6):
    vector_ = np.zeros(num)
    vector_[vector - 1] = 1.0
    return vector_


def which_figure(min_v=1, max_v=4, p=0.8):
    prob = np.random.uniform(0.0, 1.0)  # random
    k = 1
    for k in range(min_v, max_v):
        prob_ = math.pow((1 - p), k - 1) * p
        if prob < prob_:
            break
        prob -= prob_
    return k


def state2coord(state, size, reg=False):
    x = state // size
    y = state % size
    if reg:
        return [float(x)/size, float(y)/size]
    return [x, y]


def coord2state(coord, size):
    return coord[0] * size + coord[1]


def distance(p1, p2, size=None):
    if size is not None:
        p1 = state2coord(p1, size)
        p2 = state2coord(p2, size)
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def border_func(x, min_v=0.0, max_v=1.0, d_type=float):
    """
    与最大值取较小者，与最小值取较大者，返回值的类型取决于d_type。
    """
    # print(x, min_v, max_v, d_type)
    return d_type(min(max(x, min_v), max_v))


def bbox(pos: tuple, delta=(0.0, 0.0)):
    """
    :return: (min_x, min_y, max_x, max_y)
    """
    return (pos[0] - delta[0], pos[1] - delta[1],
            pos[0] + delta[0], pos[1] + delta[1])


def is_overlap(pos1: tuple, pos2: tuple, delta=(0.0, 0.0)):
    box1 = bbox(pos1, delta=delta)
    box2 = bbox(pos2, delta=delta)
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    if box1[1] > box2[3] or box2[1] > box1[3]:
        return False
    return True

