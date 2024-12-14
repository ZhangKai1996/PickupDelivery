from copy import deepcopy

from env.utils import distance


# properties and state of physical world entity
class Entity(object):
    def __init__(self, name='', movable=False):
        self.name = name
        self.movable = movable

        self.state = None
        self.end_state = None
        self.last_state = None

    def set_state(self, state):
        self.state = deepcopy(state)

    def set_last_state(self, state):
        self.last_state = deepcopy(state)

    def set_end_state(self, state):
        self.end_state = deepcopy(state)

    def clear(self):
        self.state = None
        self.end_state = None
        self.last_state = None


# properties of landmark entities
class Destination(Entity):
    def __init__(self, **kwargs):
        super(Destination, self).__init__(**kwargs)
        self.movable = False
        self.occupied = None
        self.arrived = False

    def update(self, clock):
        self.occupied = clock

    def is_occupied(self):
        return self.occupied is not None

    def clear(self):
        super().clear()
        self.occupied = None
        self.arrived = False


class Stone(Entity):
    def __init__(self, **kwargs):
        super(Stone, self).__init__(**kwargs)
        self.movable = False

    def clear(self):
        super().clear()


# properties of agent entities
class Agent(Entity):
    def __init__(self, **kwargs):
        super(Agent, self).__init__(**kwargs)
        self.movable = True

        self.dist = 0.0
        self.orders = []
        self.sequence = []
        self.cur_idx = 0

    def cur_state(self):
        if self.cur_idx >= len(self.sequence): return self.end_state
        return self.sequence[self.cur_idx].state

    def set_sequence(self, seq=None):
        self.sequence = []
        for order in self.orders:
            self.sequence.append(order.merchant)
            self.sequence.append(order.buyer)

        # print([p.name for p in self.sequence], end='-->')
        if seq is not None:
            ret = {v: i for i, v in enumerate(seq)}
            self.sequence = [self.sequence[ret[k]]
                             for k in sorted(ret.keys(), reverse=True)]

    def update(self):
        if self.last_state is not None:
            self.dist += distance(self.last_state, self.state)
        if self.is_sequence_over(): return False
        if self.cur_idx >= len(self.sequence):
            is_arrived = distance(self.state, self.end_state) <= 0
        else:
            point = self.sequence[self.cur_idx]
            is_arrived = distance(self.state, point.state) <= 0
            point.arrived = is_arrived

        if is_arrived: self.cur_idx += 1
        return is_arrived

    def is_empty(self):
        if len(self.orders) <= 0: return True
        return all([task.is_finished() for task in self.orders])

    def is_sequence_over(self):
        return self.cur_idx >= len(self.sequence) + 1

    @property
    def mass(self):
        mass = 0
        for order in self.orders:
            if order.is_finished(): continue
            if order.is_picked(): mass += order.mass
        return mass

    def clear(self):
        super().clear()
        self.dist = 0.0
        self.orders = []
        self.sequence = []
        self.cur_idx = 0


class Order:
    def __init__(self, name='', buyer=None, merchant=None, clock=0):
        self.name = name
        self.clock = clock
        self.mass = 1.0

        self.merchant = merchant
        self.buyer = buyer
        self.agent = None

    def pick_time(self):
        if self.merchant.occupied is None: return None
        return self.merchant.occupied - self.clock

    def delivery_time(self):
        if self.buyer.occupied is None: return None
        return self.buyer.occupied - self.clock

    def is_assigned(self):
        return self.agent is not None

    def is_picked(self):
        return self.is_assigned() and self.merchant.occupied is not None

    def is_finished(self):
        if not self.is_picked() or self.buyer.occupied is None:
            return False
        return self.merchant.occupied < self.buyer.occupied

    def clear(self):
        self.agent = None
        self.merchant.clear()
        self.buyer.clear()


