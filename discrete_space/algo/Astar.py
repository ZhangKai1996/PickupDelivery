"""
A_star 2D
@author: huiming zhou
"""

import math
import heapq


class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, obs, size):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.obs = obs
        self.size = size

        self.OPEN = []           # priority queue / OPEN set
        self.CLOSED = []         # CLOSED set / VISITED order
        self.PARENT = dict()     # recorded parent
        self.g = dict()          # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(
            self.OPEN,
            (self.f_value(self.s_start), self.s_start)
        )

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(
            OPEN,
            (g[s_start] + e * self.heuristic(s_start), s_start)
        )

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        motions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        nei_list = set()
        for u in motions:
            s_next = (s[0] + u[0], s[1] + u[1])
            if self.size > s_next[0] >= 0 and self.size > s_next[1] >= 0:
                nei_list.add(s_next)

        return nei_list

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


class BidirectionalAStar:
    def __init__(self, s_start, s_goal, heuristic_type, obs):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        self.obs = obs

        self.OPEN_fore = []  # OPEN set for forward searching
        self.OPEN_back = []  # OPEN set for backward searching
        self.CLOSED_fore = []  # CLOSED set for forward
        self.CLOSED_back = []  # CLOSED set for backward
        self.PARENT_fore = dict()  # recorded parent for forward
        self.PARENT_back = dict()  # recorded parent for backward
        self.g_fore = dict()  # cost to come for forward
        self.g_back = dict()  # cost to come for backward

    def init(self):
        """
        initialize parameters
        """

        self.g_fore[self.s_start] = 0.0
        self.g_fore[self.s_goal] = math.inf
        self.g_back[self.s_goal] = 0.0
        self.g_back[self.s_start] = math.inf
        self.PARENT_fore[self.s_start] = self.s_start
        self.PARENT_back[self.s_goal] = self.s_goal
        heapq.heappush(self.OPEN_fore,
                       (self.f_value_fore(self.s_start), self.s_start))
        heapq.heappush(self.OPEN_back,
                       (self.f_value_back(self.s_goal), self.s_goal))

    def searching(self):
        """
        Bidirectional A*
        :return: connected path, visited order of forward, visited order of backward
        """

        self.init()
        s_meet = self.s_start

        while self.OPEN_fore and self.OPEN_back:
            # solve foreward-search
            _, s_fore = heapq.heappop(self.OPEN_fore)

            if s_fore in self.PARENT_back:
                s_meet = s_fore
                break

            self.CLOSED_fore.append(s_fore)

            for s_n in self.get_neighbor(s_fore):
                new_cost = self.g_fore[s_fore] + self.cost(s_fore, s_n)

                if s_n not in self.g_fore:
                    self.g_fore[s_n] = math.inf

                if new_cost < self.g_fore[s_n]:
                    self.g_fore[s_n] = new_cost
                    self.PARENT_fore[s_n] = s_fore
                    heapq.heappush(self.OPEN_fore,
                                   (self.f_value_fore(s_n), s_n))

            # solve backward-search
            _, s_back = heapq.heappop(self.OPEN_back)

            if s_back in self.PARENT_fore:
                s_meet = s_back
                break

            self.CLOSED_back.append(s_back)

            for s_n in self.get_neighbor(s_back):
                new_cost = self.g_back[s_back] + self.cost(s_back, s_n)

                if s_n not in self.g_back:
                    self.g_back[s_n] = math.inf

                if new_cost < self.g_back[s_n]:
                    self.g_back[s_n] = new_cost
                    self.PARENT_back[s_n] = s_back
                    heapq.heappush(self.OPEN_back,
                                   (self.f_value_back(s_n), s_n))

        return self.extract_path(s_meet), self.CLOSED_fore, self.CLOSED_back

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """
        motions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        return [(s[0] + u[0], s[1] + u[1]) for u in motions]

    def extract_path(self, s_meet):
        """
        extract path from start and goal
        :param s_meet: meet point of bi-direction a*
        :return: path
        """

        # extract path for foreward part
        path_fore = [s_meet]
        s = s_meet

        while True:
            s = self.PARENT_fore[s]
            path_fore.append(s)
            if s == self.s_start:
                break

        # extract path for backward part
        path_back = []
        s = s_meet

        while True:
            s = self.PARENT_back[s]
            path_back.append(s)
            if s == self.s_goal:
                break

        return list(reversed(path_fore)) + list(path_back)

    def f_value_fore(self, s):
        """
        forward searching: f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g_fore[s] + self.h(s, self.s_goal)

    def f_value_back(self, s):
        """
        backward searching: f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g_back[s] + self.h(s, self.s_start)

    def h(self, s, goal):
        """
        Calculate heuristic value.
        :param s: current node (state)
        :param goal: goal node (state)
        :return: heuristic value
        """

        heuristic_type = self.heuristic_type

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False
