"""
Best-First Searching
@author: huiming zhou
"""

import math
import heapq

from algo import Astar


class BestFirst(Astar.AStar):
    """BestFirst set the heuristics as the priority
    """
    def searching(self):
        """
        Breadth-first Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(
            self.OPEN,
            (self.heuristic(self.s_start), self.s_start)
        )

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s

                    # best first set the heuristics as the priority
                    heapq.heappush(self.OPEN, (self.heuristic(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

