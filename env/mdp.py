import random

from env.utils import one_hot


class StochasticEnv:
    def __init__(self):
        self.visited_six = False
        self.current_state = 2
        # number of actions (left: 0, right: 1)
        self.n_action = 2
        # number of states
        self.n_state = 6

    def reset(self):
        self.visited_six = False
        self.current_state = 2
        return one_hot(self.current_state)

    def step(self, action):
        if self.current_state != 1:
            # If "right" selected
            if action == 1:
                if random.random() < 0.5 and self.current_state < 6:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
            if self.current_state == 6:
                self.visited_six = True

        if self.current_state == 1:
            if self.visited_six:
                return one_hot(self.current_state), 1.00, True, {}
            return one_hot(self.current_state), 1.00 / 100.00, True, {}
        return one_hot(self.current_state), 0.0, False, {}
