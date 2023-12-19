import math
import numpy as np


def one_hot(vector):
    vector_ = np.zeros(6)
    vector_[vector - 1] = 1.0
    return np.expand_dims(vector_, axis=0)


def which_figure(min_v=1, max_v=4, p=0.8):
    prob = np.random.uniform(0.0, 1.0)  # random
    k = 1
    for k in range(min_v, max_v):
        prob_ = math.pow((1 - p), k - 1) * p
        if prob < prob_:
            break
        prob -= prob_
    return k


def distance(p1, p2):
    delta_x = math.pow(p2[0] - p1[0], 2)
    delta_y = math.pow(p2[1] - p1[0], 2)
    return math.sqrt(delta_x + delta_y)


def is_collision(obj1, obj2):
    return distance(obj1.state.p_pos, obj2.state.p_pos) < (obj1.size + obj2.size)


def border_func(x, min_v=0.0, max_v=1.0, d_type=float):
    """
    与最大值取较小者，与最小值取较大者，返回值的类型取决于d_type。
    """
    print(x, min_v, max_v, d_type)
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


def region_segmentation(kwargs: dict, size: int, radius: float):
    pos_dict, check_list = {}, []
    for key, num in kwargs.items():
        poses = set()
        while True:
            pos = tuple(np.random.uniform(0, size, size=(2,)))
            for pos1 in check_list:
                if is_overlap(pos, pos1, delta=(radius, radius)):
                    break
            else:
                poses.add(pos)
                check_list.append(pos)

            if len(poses) >= num:
                break
        pos_dict[key] = poses
    return pos_dict
