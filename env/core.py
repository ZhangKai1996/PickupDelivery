from copy import deepcopy

from env.utils import distance


# physical/external base state of all entities
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

    def set(self, other):
        self.p_pos = deepcopy(other.p_pos)
        self.p_vel = deepcopy(other.p_vel)


# properties and state of physical world entity
class Entity(object):
    def __init__(self, name='', size=0.05, movable=False, color=(0.0, 0.0, 0.0)):
        # name
        self.name = name
        # properties:
        self.size = size
        # entity can move / be pushed
        self.movable = movable
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = color
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        self.last_state = EntityState()


# properties of landmark entities
class Buyer(Entity):
    def __init__(self, **kwargs):
        super(Buyer, self).__init__(**kwargs)
        self.occupied = False
        self.occ_time = None

    def update(self, clock):
        self.occupied = True
        self.occ_time = clock


class Merchant(Entity):
    def __init__(self, **kwargs):
        super(Merchant, self).__init__(**kwargs)
        self.occupied = False
        self.occ_time = None

    def update(self, clock):
        self.occupied = True
        self.occ_time = clock


# properties of agent entities
class Agent(Entity):
    def __init__(self, **kwargs):
        super(Agent, self).__init__(**kwargs)
        # physical motor noise amount
        self.u_noise = None
        # control range
        self.u_range = 1.0
        # action
        self.action = None
        # tasks
        self.tasks = []
        # endurance
        self.dist = 0.0
        # mass
        self.initial_mass = 1.0

    def status(self):
        return '>>>{}: ({},{}), ({:>6.3f})'.format(
            self.name, self.initial_mass, self.mass, self.dist
        )

    def pre_update(self):
        self.last_state.set(other=self.state)

    def update(self):
        self.dist += distance(self.last_state.p_pos, self.state.p_pos)

    def empty(self):
        if len(self.tasks) <= 0:
            return True
        return all([task.is_finished() for task in self.tasks])

    @property
    def mass(self):
        m = self.initial_mass
        # for task in self.tasks:
        #     if task.is_finished():
        #         continue
        #     if task.is_picked():
        #         m += task.mass
        return m

    def clear(self):
        self.dist = 0.0
        self.tasks = []
        self.action = None


class Task:
    def __init__(self, name='', buyer=None, merchant=None, clock=0):
        self.name = name
        self.clock = clock
        self.mass = 1.0
        self.merchant = buyer
        self.buyer = merchant
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
        if not m.occupied:
            return None
        return m.occ_time - self.clock

    def delivery_time(self):
        b = self.buyer
        if not b.occupied:
            return None
        return b.occ_time - self.clock

    def is_assigned(self):
        return self.agent is not None

    def is_picked(self):
        return self.is_assigned() and self.merchant.occupied

    def is_finished(self):
        return self.is_picked() and self.buyer.occupied
