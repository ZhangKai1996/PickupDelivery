from typing import Dict, Tuple, List
from copy import deepcopy


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
    def __init__(self, name='', size=0.05, movable=False, collide=True,
                 color=(0.0, 0.0, 0.0)):
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
        self.last_state= EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Buyer(Entity):
     def __init__(self, **kwargs):
        super(Buyer, self).__init__(**kwargs)
        self.occupied = False


class Merchant(Entity):
    def __init__(self, **kwargs):
        super(Merchant, self).__init__(**kwargs)
        self.occupied = False


# properties of agent entities
class Agent(Entity):
    def __init__(self, **kwargs):
        super(Agent, self).__init__(**kwargs)
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

    def update(self):
        self.last_state.set(other=self.state)


class Rider(Entity):
    name: str
    radius: float
    state: str = 'Empty'  # Empty, Pickup, Delivery, Collision_O, Collision_A
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_pos: Tuple[float, float]
    distance: int = 0
    endurance: List[int] = [60, 60]
    tasks: Dict[str, int] = {}
    is_collision: bool = False

    def __init__(self, name='Drone'):
        super(Rider, self).__init__()
        self.name = name

    def __str__(self):
        return '{},{},{}, Loc: ({:>6.3f},{:>6.3f}), Vel: ({:>+5.3f},{:>+5.3f})'.format(
            self.name, self.state,
            int(self.is_collision),
            self.position[0], self.position[1],
            self.velocity[0], self.velocity[1]
        )


class Task:
    def __init__(self, name='', buyer=None, merchant=None, clock=0):
        self.name = name
        self.merchant = buyer
        self.buyer = merchant
        self.clock = clock
        self.agent = None

    def is_finished(self):
        return (
            self.agent is not None
            and self.merchant.occupied
            and self.buyer.occupied
        )