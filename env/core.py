from copy import deepcopy

from env.utils import distance


# properties and state of physical world entity
class Entity(object):
    def __init__(self, name='', movable=False):
        self.name = name
        self.movable = movable

        self.state = None
        self.last_state = None

    def set_state(self, state):
        self.state = deepcopy(state)

    def set_last_state(self, state):
        self.last_state = deepcopy(state)

    def clear(self):
        self.state = None
        self.last_state = None


# properties of landmark entities
class Destination(Entity):
    def __init__(self, **kwargs):
        super(Destination, self).__init__(**kwargs)
        self.movable = False
        self.occupied = None

    def update(self, clock):
        self.occupied = clock

    def is_occupied(self):
        return self.occupied is not None

    def clear(self):
        super().clear()
        self.occupied = None


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
        self.orders = []
        self.dist = 0.0

    def status(self):
        return '>>>{}: ({},{:>6.2f})'.format(self.name, self.mass, self.dist)

    def update(self):
        if self.last_state is None:
            return
        self.dist += distance(self.last_state, self.state)

    def is_empty(self):
        if len(self.orders) <= 0:
            return True
        return all([task.is_finished() for task in self.orders])

    @property
    def mass(self):
        mass = 0
        for order in self.orders:
            if order.is_finished(): continue
            if order.is_picked(): mass += 1
        return mass

    def clear(self):
        super().clear()
        self.dist = 0.0
        self.orders = []


class Order:
    def __init__(self, name='', buyer=None, merchant=None, clock=0):
        self.name = name
        self.clock = clock
        self.mass = 1.0
        self.merchant = merchant
        self.buyer = buyer
        self.agent = None

    def status(self):
        status = [self.name, ]
        if self.is_finished():
            status += ['Finished', str(self.pick_time()), str(self.delivery_time())]
        elif self.is_picked():
            status += ['Picked', str(self.pick_time()), '-1']
        elif self.is_assigned():
            status += ['Assigned', '-1', '-1']
        else:
            status += ['Unassigned', '-1', '-1']
        return ','.join(status)

    def pick_time(self):
        m = self.merchant
        if m.occupied is None:
            return None
        return m.occupied - self.clock

    def delivery_time(self):
        b = self.buyer
        if b.occupied is None:
            return None
        return b.occupied - self.clock

    def is_assigned(self):
        return self.agent is not None

    def is_picked(self):
        return self.is_assigned() and self.merchant.occupied is not None

    def is_finished(self):
        return self.is_picked() and self.buyer.occupied is not None

    def clear(self):
        self.agent = None
        self.merchant.clear()
        self.buyer.clear()


