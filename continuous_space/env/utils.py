import math
import numpy as np


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


def bbox(pos, delta_x=1.0, delta_y=1.0):
    """
    :param pos:
    :param delta_x:
    :param delta_y:
    :return: (min_x, min_y, max_x, max_y)
    """
    return (pos[0] - delta_x, pos[1] - delta_y,
            pos[0] + delta_x, pos[1] + delta_y)


def is_overlap(pos1, pos2, delta_x=1.0, delta_y=1.0):
    box1 = bbox(pos1, delta_x=delta_x, delta_y=delta_y)
    box2 = bbox(pos2, delta_x=delta_x, delta_y=delta_y)
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False

    if box1[1] > box2[3] or box2[1] > box1[3]:
        return False

    return True


def region_segmentation(kwargs, size, radius=0.5):
    pos_dict = {}
    check_list = []
    for key, num in kwargs.items():
        if isinstance(num, float):
            num = int(size * size * num)

        poses = set()
        while True:
            pos = tuple(np.random.uniform(0, size, size=(2, )))

            ok = True
            for pos1 in check_list:
                if is_overlap(pos, pos1, delta_x=radius, delta_y=radius):
                    ok = False
            if ok:
                poses.add(pos)
                check_list.append(pos)
            if len(poses) >= num:
                break

        pos_dict[key] = poses
    return pos_dict
