"""
RRT_2D
@author: huiming zhou
"""
import copy
import math

from tqdm import tqdm
import numpy as np

from .utils import Node, Edge, Utils


class RRT:
    def __init__(self, s_start, s_goal, env):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)

        self.vertex = [self.s_start]

        self.env = env
        self.x_range = [0, env.size]
        self.y_range = [0, env.size]
        self.utils = Utils(env=self.env)

    def planning(self, sample_rate=0.05, iter_max=10000, step_len=0.5):
        for _ in tqdm(range(iter_max), desc='RRT Planning:'):
            node_rand = self.generate_random_node(sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand, step_len=step_len)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= step_len and not self.utils.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal, step_len=step_len)
                    return self.extract_path(node_new)
        return None

    def generate_random_node(self, sample_rate):
        if np.random.random() > sample_rate:
            return Node(
                (np.random.uniform(self.x_range[0], self.x_range[1]),
                 np.random.uniform(self.y_range[0], self.y_range[1]))
            )
        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end, step_len):
        dist, theta = self.get_distance_and_angle(node_start, node_end)
        dist = min(step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))
        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


class RRTConnect:
    def __init__(self, s_start, s_goal, env):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.env = env
        self.x_range = [0, env.size]
        self.y_range = [0, env.size]
        self.utils = Utils(env=self.env)

    def planning(self, sample_rate=0.05, iter_max=10000, step_len=0.5):
        for _ in tqdm(range(iter_max), desc='RRT Connect Planning:'):
            node_rand = self.generate_random_node(self.s_goal, sample_rate)
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand, step_len)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new, step_len)

                if node_new_prim and not self.utils.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new, step_len)
                        if node_new_prim2 and not self.utils.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim
        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if node_new_prim.x == node_new.x and node_new_prim.y == node_new.y:
            return True
        return False

    def generate_random_node(self, sample_goal, sample_rate):
        if np.random.random() > sample_rate:
            return Node((np.random.uniform(self.x_range[0], self.x_range[1]),
                         np.random.uniform(self.y_range[0], self.y_range[1])))
        return sample_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end, step_len):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


class DynamicRRT:
    def __init__(self, s_start, s_goal, env):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.vertex = [self.s_start]
        self.vertex_old = []
        self.vertex_new = []
        self.edges = []

        self.env = env
        self.x_range = [0, env.size]
        self.y_range = [0, env.size]
        self.utils = Utils(env=self.env)
        self.obs_add = [0, 0, 0]

        self.path = []
        self.waypoint = []

    def planning(self, sample_rate=0.05, iter_max=10000, step_len=0.5):
        for _ in tqdm(range(iter_max), desc='Dynamic RRT Planning:'):
            node_rand = self.generate_random_node(sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand, step_len)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.edges.append(Edge(node_near, node_new))
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= step_len:
                    self.new_state(node_new, self.s_goal, step_len)
                    self.path = self.extract_path(node_new)
                    self.waypoint = self.extract_waypoint(node_new)
                    return self.path
        return None

    def on_press(self, pos, radius=0.5, start=None):
        self.s_start = Node(start)

        x, y = pos[0], pos[1]
        print("Add circle obstacle at:", (x, y))
        self.obs_add = [x, y, radius]
        self.utils.update_obs(obs_cir=[[x, y, radius], ])
        self.InvalidateNodes()

        if self.is_path_invalid():
            print("Path is Replanning ...")
            self.path, self.waypoint = self.re_planning()

            print("len_vertex: ", len(self.vertex))
            print("len_vertex_old: ", len(self.vertex_old))
            print("len_vertex_new: ", len(self.vertex_new))

            self.vertex_new = []
        else:
            print("Trimming Invalid Nodes ...")
            self.TrimRRT()

    def InvalidateNodes(self):
        for edge in self.edges:
            if self.is_collision_obs_add(edge.parent, edge.child):
                edge.child.flag = "INVALID"

    def is_path_invalid(self):
        for node in self.waypoint:
            if node.flag == "INVALID":
                return True

    def is_collision_obs_add(self, start, end):
        obs_add = self.obs_add
        if math.hypot(start.x - obs_add[0], start.y - obs_add[1]) <= obs_add[2]:
            return True

        if math.hypot(end.x - obs_add[0], end.y - obs_add[1]) <= obs_add[2]:
            return True

        o, d = self.utils.get_ray(start, end)
        if self.utils.is_intersect_circle(o, d, [obs_add[0], obs_add[1]], obs_add[2]):
            return True

        return False

    def re_planning(self, sample_rate=0.05, iter_max=10000, step_len=0.5, wpt_sample_rate=.6):
        self.TrimRRT()

        for i in range(iter_max):
            node_rand = self.generate_random_node_replanning(sample_rate, wpt_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand, step_len)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.vertex_new.append(node_new)
                self.edges.append(Edge(node_near, node_new))
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= step_len:
                    self.new_state(node_new, self.s_goal, step_len)
                    path = self.extract_path(node_new)
                    waypoint = self.extract_waypoint(node_new)
                    print("path: ", len(path))
                    print("waypoint: ", len(waypoint))
                    return path, waypoint

        return None

    def TrimRRT(self):
        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node_p = node.parent
            if node_p.flag == "INVALID":
                node.flag = "INVALID"

        self.vertex = [node for node in self.vertex if node.flag == "VALID"]
        self.vertex_old = copy.deepcopy(self.vertex)
        self.edges = [Edge(node.parent, node) for node in self.vertex[1:len(self.vertex)]]

    def generate_random_node(self, goal_sample_rate):
        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0], self.x_range[1]),
                         np.random.uniform(self.y_range[0], self.y_range[1])))
        return self.s_goal

    def generate_random_node_replanning(self, sample_rate, waypoint_sample_rate):
        p = np.random.random()

        if p < sample_rate:
            return self.s_goal
        elif sample_rate < p < sample_rate + waypoint_sample_rate:
            return self.waypoint[np.random.randint(0, len(self.waypoint) - 1)]
        else:
            return Node((np.random.uniform(self.x_range[0], self.x_range[1]),
                         np.random.uniform(self.y_range[0], self.y_range[1])))

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end, step_len):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))
        return path

    def extract_waypoint(self, node_end):
        waypoint = [self.s_goal]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            waypoint.append(node_now)
        return waypoint

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
