"""
Collision avoidance using Velocity-obstacle method

author: Ashwin Bose (atb033@github.com)
"""
from math import cos, sin, atan2, asin, pi, sqrt
import numpy as np

from common.utils import distance


class VOPlanner:
    def __init__(self, env):
        self.env = env
        self.obstacles = env.walls
        self.max_v = 1.0

    def simulate(self, num_iter=100, step=1.0):
        i = 0
        while True:
            v_des = self.compute_desired_velocity()
            v_actions = self.compute_velocity(v_des)
            self.env.step(v_actions, duration=step)
            self.env.render(show=False)

            i += 1
            if i >= num_iter:
                break

    def compute_desired_velocity(self):
        merchants = list(self.env.merchants)

        vel_desired = []
        for i, drone in enumerate(self.env.drones):
            start = drone.position
            goal = merchants[i]

            dif_x = (goal[0] - start[0], goal[1] - start[1])
            norm = distance(dif_x, [0, 0])
            norm_dif_x = [dif_x[k] * self.max_v / norm for k in range(2)]
            vel_d = (.0, .0) if distance(start, goal) < 0.1 else norm_dif_x[:]
            vel_desired.append(vel_d)
        return vel_desired

    def compute_velocity(self, v_des, over_approx_c2s=1.5):
        """
        compute the best velocity given the desired velocity, current velocity and workspace model
        """
        radius = self.env.radius
        drones = self.env.drones

        v_opt = []
        for i, drone in enumerate(drones):
            v_a = drone.velocity
            p_a = drone.position

            rvo_ba_all = []
            for j, drone1 in enumerate(drones):
                if i == j:
                    continue

                v_b = drone1.velocity
                p_b = drone1.position
                transl_v_ba = [p_a[0] + 0.5 * (v_b[0] + v_a[0]), p_a[1] + 0.5 * (v_b[1] + v_a[1])]
                theta_ba = atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
                dist_ba = max(distance(p_a, p_b), 2 * radius)
                theta_ba_ort = asin(2 * radius / dist_ba)
                theta_ort_left = theta_ba + theta_ba_ort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                theta_ort_right = theta_ba - theta_ba_ort
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]

                rvo_ba_all.append([
                    transl_v_ba, bound_left, bound_right, dist_ba, 2 * radius
                ])

            for p_b in self.obstacles:
                transl_ba = p_b[:]
                rad = sqrt(2) * radius * over_approx_c2s
                dist_ba = max(distance(p_a, p_b), rad + radius)
                theta_ba = atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])

                # over-approximation of square to circular
                theta_ba_ort = asin((rad + radius) / dist_ba)
                theta_ort_left = theta_ba + theta_ba_ort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                theta_ort_right = theta_ba - theta_ba_ort
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                rvo_ba_all.append([transl_ba, bound_left, bound_right, dist_ba, rad + radius])
            v_opt.append(self.intersect(p_a, v_des[i], rvo_ba_all)[:])
        return v_opt

    def intersect(self, p_a, v_a, rvo_ba_all):
        norm_v = distance(v_a, [0, 0])

        suitable_v = []
        unsuitable_v = []
        for theta in np.arange(0, 2 * pi, 0.1):
            for rad in np.arange(0.02, norm_v + 0.02, norm_v / 5.0):
                new_v = [rad * cos(theta), rad * sin(theta)]
                suit = True
                for RVO_BA in rvo_ba_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dif = [new_v[0] + p_a[0] - p_0[0], new_v[1] + p_a[1] - p_0[1]]
                    theta_dif = atan2(dif[1], dif[0])
                    theta_right = atan2(right[1], right[0])
                    theta_left = atan2(left[1], left[0])
                    if self.in_between(theta_right, theta_dif, theta_left):
                        suit = False
                        break
                if suit:
                    suitable_v.append(new_v)
                else:
                    unsuitable_v.append(new_v)
        new_v = v_a[:]
        suit = True
        for RVO_BA in rvo_ba_all:
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0] + p_a[0] - p_0[0], new_v[1] + p_a[1] - p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
            if self.in_between(theta_right, theta_dif, theta_left):
                suit = False
                break
        if suit:
            suitable_v.append(new_v)
        else:
            unsuitable_v.append(new_v)

        # ----------------------
        if suitable_v:
            # print('Suitable found')
            v_a_post = min(suitable_v, key=lambda v: distance(v, v_a))
            new_v = v_a_post[:]
            for RVO_BA in rvo_ba_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0] + p_a[0] - p_0[0], new_v[1] + p_a[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
        else:
            # print('Suitable not found')
            tc_V = dict()
            for unsuit_v in unsuitable_v:
                tc_V[tuple(unsuit_v)] = 0
                tc = []
                for RVO_BA in rvo_ba_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dist = RVO_BA[3]
                    rad = RVO_BA[4]
                    dif = [unsuit_v[0] + p_a[0] - p_0[0], unsuit_v[1] + p_a[1] - p_0[1]]
                    theta_dif = atan2(dif[1], dif[0])
                    theta_right = atan2(right[1], right[0])
                    theta_left = atan2(left[1], left[0])
                    if self.in_between(theta_right, theta_dif, theta_left):
                        small_theta = abs(theta_dif - 0.5 * (theta_left + theta_right))
                        if abs(dist * sin(small_theta)) >= rad:
                            rad = abs(dist * sin(small_theta))
                        big_theta = asin(abs(dist * sin(small_theta)) / rad)
                        dist_tg = abs(dist * cos(small_theta)) - abs(rad * cos(big_theta))
                        dist_tg = max(dist_tg, 0)
                        tc_v = dist_tg / (distance(dif, [0, 0])+1e-4)
                        tc.append(tc_v)
                tc_V[tuple(unsuit_v)] = min(tc) + 0.001
            WT = 0.2
            v_a_post = min(unsuitable_v, key=lambda v: ((WT / tc_V[tuple(v)]) + distance(v, v_a)))
        return v_a_post

    def in_between(self, theta_right, theta_dif, theta_left):
        if abs(theta_right - theta_left) <= pi:
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        else:
            if (theta_left < 0) and (theta_right > 0):
                theta_left += 2 * pi
                if theta_dif < 0:
                    theta_dif += 2 * pi
                if theta_right <= theta_dif <= theta_left:
                    return True
                return False

            if (theta_left > 0) and (theta_right < 0):
                theta_right += 2 * pi
                if theta_dif < 0:
                    theta_dif += 2 * pi

                if theta_left <= theta_dif <= theta_right:
                    return True
                return False
