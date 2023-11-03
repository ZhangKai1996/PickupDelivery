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


def transform_tuple2digit(a: tuple, row: int, column: int) -> int:
    assert a[0] < row and a[1] < column
    return a[0] * column + a[1]


def transform_digit2tuple(a: int, row: int, column: int) -> tuple:
    return int(a // column), a % row
