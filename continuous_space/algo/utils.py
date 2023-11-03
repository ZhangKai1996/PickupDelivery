"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.flag = "VALID"

    def to_string(self):
        return '({}, {})'.format(self.x, self.y)


class Edge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = "VALID"


class Utils:
    def __init__(self, env):
        self.env = env
        self.obs_circle = env.get_obs(shape='circle')
        self.obs_rectangle = env.get_obs(shape='rectangle')
        self.boundary = env.get_obs(shape='boundary')

    def update_obs(self, obs_cir=None, bound=None, obs_rec=None):
        if obs_cir is not None:
            self.obs_circle += obs_cir
        if bound is not None:
            self.boundary += bound
        if obs_rec is not None:
            self.obs_rectangle += obs_rec

    def get_obs_vertex(self, inputs=None):
        if inputs is None:
            inputs = self.obs_rectangle

        obs_list = []
        for (min_x, min_y, max_x, max_y) in inputs:
            vertex_list = [[min_x, min_y],
                           [min_x, max_y],
                           [max_x, max_y],
                           [max_x, min_y]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False

    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r:
                return True

        return False

    def is_collision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False

    def is_inside_obs(self, node, delta=0.5):
        nx, ny = node.x, node.y

        for (x, y, r) in self.obs_circle:
            if math.hypot(nx - x, ny - y) <= r:
                return True

        min_nx, max_nx = nx-delta, nx+delta
        min_ny, max_ny = ny-delta, ny+delta
        for (min_x, min_y, max_x, max_y) in self.obs_rectangle:
            if min_nx > max_x or min_x > max_nx:
                continue
            if min_ny > max_y or min_y > max_ny:
                continue
            return True

        [min_x, min_y, max_x, max_y] = self.boundary
        if nx < min_x or nx >= max_x or ny < min_y or ny >= max_y:
            return True

        return False

    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)
