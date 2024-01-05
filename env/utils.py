import math
import numpy as np


def one_hot(vector):
    vector_ = np.zeros(6)
    vector_[vector - 1] = 1.0
    return np.expand_dims(vector_, axis=0)


def distance(p1, p2):
    delta_x = math.pow(p2[0] - p1[0], 2)
    delta_y = math.pow(p2[1] - p1[1], 2)
    return math.sqrt(delta_x + delta_y)


def bbox(pos: tuple, delta=(0.0, 0.0)):
    """
    :return: (min_x, min_y, max_x, max_y)
    """
    return (
        pos[0] - delta[0],
        pos[1] - delta[1],
        pos[0] + delta[0],
        pos[1] + delta[1]
    )


def is_overlap(pos1: tuple, pos2: tuple, delta1=(0.0, 0.0), delta2=(0.0, 0.0)):
    box1 = bbox(pos1, delta=delta1)
    box2 = bbox(pos2, delta=delta2)
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    if box1[1] > box2[3] or box2[1] > box1[3]:
        return False
    return True
