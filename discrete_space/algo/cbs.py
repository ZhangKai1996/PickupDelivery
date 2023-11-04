"""

Python implementation of Conflict-based search

author: Ashwin Bose (@atb033)

"""
import math
from itertools import combinations
from copy import deepcopy


class AStar:
    def __init__(self, env, constraints):
        self.size = env.size
        self.constraints = constraints
        self.obstacles = env.walls

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def state_valid(self, state):
        x, y = state.location
        return 0 <= x < self.size and 0 <= y < self.size \
            and VertexConstraint(state.time, (x, y)) not in self.constraints.vertex_constraints \
            and (x, y) not in self.obstacles

    def transition_valid(self, state_1, state_2, empty=False):
        if empty:
            return True
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def get_neighbors(self, state):
        motions = [
            (-1, 1), (0, 1), (1, 1),
            (-1, 0), (0, 0), (1, 0),
            (-1, -1), (0, -1), (1, -1)
        ]

        neighbors = []
        x, y = state.location
        for [delta_x, delta_y] in motions:
            n = State(state.time + 1, (x+delta_x, y+delta_y))
            empty = delta_x == 0 and delta_y == 0
            if self.state_valid(n) and self.transition_valid(state, n, empty=empty):
                neighbors.append(n)
        return neighbors

    def cost(self, s_start, s_goal):
        s_start = s_start.location
        s_goal = s_goal.location
        if self.is_collision(s_start, s_goal):
            return math.inf

        # return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
        return 1

    def is_collision(self, s_start, s_end):
        if s_start in self.obstacles or s_end in self.obstacles:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obstacles or s2 in self.obstacles:
                return True
        return False

    def search(self, start, goal):
        """
        low level search
        """
        closed_set = set()
        open_set = {start}
        came_from = {}

        g_score = {start: 0}
        f_score = {start: start.admissible_heuristic(goal)}

        while open_set:
            temp_dict = {open_item: f_score.setdefault(open_item, float("inf"))
                         for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)
            if current.is_equal_except_time(goal):
                return self.reconstruct_path(came_from, current)

            open_set -= {current}
            closed_set |= {current}
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                collision_cost = self.cost(current, neighbor)
                tentative_g_score = g_score.setdefault(current, float("inf")) + collision_cost
                # tentative_g_score = g_score.setdefault(current, float("inf")) + 1
                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + neighbor.admissible_heuristic(goal)
        return False


class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        if self.location[0] != other.location[0]:
            return False
        if self.location[1] != other.location[1]:
            return False
        return self.time == other.time

    def __hash__(self):
        return hash(str(self.time) + str(self.location[0]) + str(self.location[0]))

    def is_equal_except_time(self, state):
        return self.location == state.location

    def admissible_heuristic(self, goal):
        return math.fabs(self.location[0] - goal.location[0]) + math.fabs(self.location[1] - goal.location[1])

    def __str__(self):
        return str((self.time, self.location[0], self.location[0]))


class Conflict(object):
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = (-1, -1)
        self.location_2 = (-1, -1)

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
            ', ' + str(self.location_1) + ', ' + str(self.location_2) + ')'


class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash(str(self.time) + str(self.location))

    def __str__(self):
        return '(' + str(self.time) + ', ' + str(self.location) + ')'


class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2

    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))

    def __str__(self):
        return '(' + str(self.time) + ', ' + str(self.location_1) + ', ' + str(self.location_2) + ')'


class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints]) + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])


class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost


class CBS(object):
    def __init__(self, env):
        self.env = env
        self.open_set = set()
        self.closed_set = set()
        self.constraints = Constraints()
        self.constraint_dict = {}
        self.a_star = AStar(env=env, constraints=self.constraints)
        self.__make_agent_dict(env)

    def __make_agent_dict(self, env):
        merchants = list(env.merchants)
        self.od_dict = {}
        for i, drone in enumerate(env.drones):
            start_state = State(0, drone.position)
            goal_state = State(0, merchants[i])
            self.od_dict.update({
                drone.name: [start_state, goal_state]
            })

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

    def compute_solution(self):
        solution = {}
        for key, [start, goal] in self.od_dict.items():
            self.constraints = self.constraint_dict.setdefault(key, Constraints())
            local_solution = self.a_star.search(start, goal)
            if not local_solution:
                return False
            solution.update({key: local_solution})
        return solution

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t + 1)
                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t + 1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint
        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()
            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)
            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}
            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2
        return constraint_dict

    def search(self):
        start = HighLevelNode()
        for key in self.od_dict.keys():
            start.constraint_dict[key] = Constraints()

        start.solution = self.compute_solution()
        if not start.solution:
            return {}
        start.cost = self.compute_solution_cost(start.solution)

        self.open_set |= {start}
        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            self.constraint_dict = P.constraint_dict
            conflict_dict = self.get_first_conflict(P.solution)
            if not conflict_dict:
                print("solution found")
                return self.generate_plan(P.solution)

            constraint_dict = self.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.constraint_dict = new_node.constraint_dict
                new_node.solution = self.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.compute_solution_cost(new_node.solution)

                # TODO: ending condition
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}
        return {}

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [{'t': state.time, 'x': state.location[0], 'y': state.location[1]} for state in path]
            plan[agent] = path_dict_list
        return plan

