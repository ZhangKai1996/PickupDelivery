from copy import deepcopy

from .utils import is_collision


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
    def __init__(self, name='', size=0.05, movable=False, collide=True, color=(0.0, 0.0, 0.0)):
        # name
        self.name = name
        # properties:
        self.size = size
        # entity can move / be pushed
        self.movable = movable
        # entity collides with others
        self.collide = collide
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
        # mass
        self.initial_mass = 1.0
        self.mass = 1.0

    def update(self):
        self.last_state.set(other=self.state)


# properties of landmark entities
class Buyer(Entity):
    def __init__(self, **kwargs):
        super(Buyer, self).__init__(**kwargs)
        self.collide = False
        self.movable = False
        self.occupied = False
        self.size = 0.20


class Merchant(Entity):
    def __init__(self, **kwargs):
        super(Merchant, self).__init__(**kwargs)
        self.collide = False
        self.movable = False
        self.occupied = False
        self.size = 0.20


class Barrier(Entity):
    def __init__(self, **kwargs):
        super(Barrier, self).__init__(**kwargs)
        self.collide = True
        self.movable = False


# properties of agent entities
class Agent(Entity):
    def __init__(self, **kwargs):
        super(Agent, self).__init__(**kwargs)
        self.collide = True
        self.movable = True
        # cannot observe the world
        self.o_range = 0.1
        # physical motor noise amount
        self.u_noise = None
        # control range
        self.u_range = 1.0
        # action
        self.action = None
        # tasks
        self.tasks = []
        self.payload = 10.0

    @property
    def is_overload(self):
        return self.mass - self.initial_mass > self.payload

    def is_collision(self, entity):
        return is_collision(self, entity)

    def update_mass(self):
        task_mass = sum([t.mass for t in self.tasks if t.is_picked])
        self.mass = self.initial_mass + task_mass


class Task:
    def __init__(self, name='', buyer=None, merchant=None, clock=0):
        self.name = name
        self.merchant = buyer
        self.buyer = merchant
        self.clock = clock
        self.agent = None
        self.status = 'Unassigned'

    @property
    def mass(self):
        return 1.0

    @property
    def pick_pos(self):
        return self.merchant.state.p_pos

    @property
    def delivery_pos(self):
        return self.buyer.state.p_pos

    @property
    def is_assigned(self):
        return self.status == 'Assigned'

    @property
    def is_picked(self):
        return self.status == 'Pickup'

    @property
    def is_finished(self):
        return self.status == 'Finished'

    def assign_to(self, agent):
        self.agent = agent
        agent.tasks.append(self)
        self.update_task_status()

    def check_occupied(self, agent):
        if is_collision(self.merchant, agent):
            self.merchant.occupied = True
            self.update_task_status()
            return True
        return False

    def update_task_status(self):
        if self.buyer.occupied:
            self.status = 'Finished'
            return
        if self.merchant.occupied:
            self.status = 'Pickup'
            return
        if self.agent is not None:
            self.status = 'Assigned'
            return
        self.status = 'Unassigned'
