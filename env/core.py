from typing import Dict, Tuple, List


# physical/external base state of all entities
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self, name='', size=0.050, movable=False, collide=True, color=(0.0, 0.0, 0.0)):
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

    def execute_action(self, action: list, size: int, duration=1.0):
        if self.is_collision:
            return

        old_pos = self.position[:]
        new_pos = (old_pos[0]+action[0]*duration, old_pos[1]+action[1]*duration)
        self.velocity = tuple(action)
        # print(old_pos, action, new_pos)
        if size > new_pos[0] >= 0 and size > new_pos[1] >= 0:
            self.position = new_pos
            self.distance += 1

        self.endurance[1] -= 1
        self.last_pos = old_pos

    def detect(self, obj):
        if isinstance(obj, Drone):
            dist = distance(self.position, obj.position)
            if dist <= self.radius+obj.radius:
                self.is_collision = True
                self.state = 'Collision_A'
                obj.is_collision = True
                obj.state = 'Collision_A'
            return

        if isinstance(obj, set) or isinstance(obj, list):
            for wall in obj:
                if is_overlap(self.position, wall, delta=(self.radius, self.radius)):
                    self.is_collision = True
                    self.state = 'Collision_O'
                    break


class Task:
    def __init__(self, name='', buyer=None, merchant=None, clock=0):
        self.name = name
        self.merchant = buyer
        self.buyer = merchant
        self.clock = clock
        self.agent = None

    def is_finished(self):
        return self.merchant.occupied and self.buyer.occupied