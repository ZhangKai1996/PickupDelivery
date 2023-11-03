from math import fabs
from bisect import bisect


class State(object):
    def __init__(self, position=(-1, -1), t=0, interval=(0, float('inf'))):
        self.position = tuple(position)
        self.time = t
        self.interval = interval


class SIPPGrid(object):
    def __init__(self):
        # self.position = ()
        self.interval_list = [(0, float('inf'))]
        self.f = float('inf')
        self.g = float('inf')
        self.parent_state = State()

    def split_interval(self, t, last_t=False):
        """
        Function to generate safe-intervals
        """
        for interval in self.interval_list:
            if last_t:
                if t <= interval[0]:
                    self.interval_list.remove(interval)
                elif t > interval[1]:
                    continue
                else:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t - 1))
            else:
                if t == interval[0]:
                    self.interval_list.remove(interval)
                    if t + 1 <= interval[1]:
                        self.interval_list.append((t + 1, interval[1]))
                elif t == interval[1]:
                    self.interval_list.remove(interval)
                    if t - 1 <= interval[0]:
                        self.interval_list.append((interval[0], t - 1))
                elif bisect(interval, t) == 1:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t - 1))
                    self.interval_list.append((t + 1, interval[1]))
            self.interval_list.sort()


class SIPPGraph(object):
    def __init__(self, env):
        self.dimensions = [env.size, env.size]

        self.obstacles = env.walls
        self.dyn_obstacles = {}

        self.graph = {}
        self.init_graph()
        self.init_intervals()

    def init_graph(self):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                grid_dict = {(i, j): SIPPGrid()}
                self.graph.update(grid_dict)

    def init_intervals(self):
        if not self.dyn_obstacles:
            return

        for schedule in self.dyn_obstacles.values():
            for i in range(len(schedule)):
                location = schedule[i]
                last_t = i == len(schedule) - 1
                position = (location["x"], location["y"])
                t = location["t"]
                self.graph[position].split_interval(t, last_t)

    def is_valid_position(self, position):
        dim_check = position[0] in range(self.dimensions[0]) and position[1] in range(self.dimensions[1])
        obs_check = position not in self.obstacles
        return dim_check and obs_check

    def get_valid_neighbours(self, position):
        neighbour_list = []

        up = (position[0], position[1] + 1)
        if self.is_valid_position(up): neighbour_list.append(up)

        down = (position[0], position[1] - 1)
        if self.is_valid_position(down): neighbour_list.append(down)

        left = (position[0] - 1, position[1])
        if self.is_valid_position(left): neighbour_list.append(left)

        right = (position[0] + 1, position[1])
        if self.is_valid_position(right): neighbour_list.append(right)

        return neighbour_list


class SIPPPlanner(SIPPGraph):
    def __init__(self, name, start, goal, env):
        SIPPGraph.__init__(self, env)
        self.start = start
        self.goal = goal
        self.name = name
        self.open = []
        self.plan = []

    def get_successors(self, state):
        successors = []
        m_time = 1
        neighbour_list = self.get_valid_neighbours(state.position)

        for neighbour in neighbour_list:
            start_t = state.time + m_time
            end_t = state.interval[1] + m_time
            for i in self.graph[neighbour].interval_list:
                if i[0] > end_t or i[1] < start_t:
                    continue
                time = max(start_t, i[0])
                s = State(neighbour, time, i)
                successors.append(s)
        return successors

    def get_heuristic(self, position):
        return fabs(position[0] - self.goal[0]) + fabs(position[1] - self.goal[1])

    def compute_plan(self):
        self.open = []
        goal_reached = False
        cost = 1

        s_start = State(self.start, 0)
        self.graph[self.start].g = 0.
        f_start = self.get_heuristic(self.start)
        self.graph[self.start].f = f_start
        self.open.append((f_start, s_start))

        # successor
        suc = None
        while not goal_reached:
            if self.open == {}:
                # Plan not found
                return 0
            s = self.open.pop(0)[1]
            for suc in self.get_successors(s):
                if self.graph[suc.position].g > self.graph[s.position].g + cost:
                    self.graph[suc.position].g = self.graph[s.position].g + cost
                    self.graph[suc.position].parent_state = s

                    if suc.position == self.goal:
                        print("Plan successfully calculated!!")
                        goal_reached = True
                        break

                    self.graph[suc.position].f = self.graph[suc.position].g + self.get_heuristic(
                        suc.position)
                    self.open.append((self.graph[suc.position].f, suc))

        # Tracking back
        start_reached = False
        self.plan = []
        current = suc
        while not start_reached:
            self.plan.insert(0, current)
            if current.position == self.start:
                start_reached = True
            current = self.graph[current.position].parent_state
        return 1

    def get_plan(self):
        path_list = []

        # first setpoint
        setpoint = self.plan[0]
        temp_dict = {"x": setpoint.position[0], "y": setpoint.position[1], "t": setpoint.time}
        path_list.append(temp_dict)

        for i in range(len(self.plan) - 1):
            for j in range(self.plan[i + 1].time - self.plan[i].time - 1):
                x = self.plan[i].position[0]
                y = self.plan[i].position[1]
                t = self.plan[i].time
                setpoint = self.plan[i]
                temp_dict = {"x": x, "y": y, "t": t + j + 1}
                path_list.append(temp_dict)

            setpoint = self.plan[i + 1]
            temp_dict = {"x": setpoint.position[0], "y": setpoint.position[1], "t": setpoint.time}
            path_list.append(temp_dict)

        data = {self.name: path_list}
        return data
