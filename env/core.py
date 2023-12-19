from typing import Dict, Tuple, List


class Entity:
    def __init__(self):
        pass


class Buyer:
    name: str
    address: Tuple[float, float]
    orders: Dict[int, int]

    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.orders = {}

    def buy(self, merchant, t: int):
        self.orders[merchant] = t

    def get_merchants(self):
        return list(self.orders.keys())

    def update_orders(self, clock):
        orders = {}
        for merchant, t in self.orders.items():
            if clock - t >= 100:  # 如果服务时间超过100s，则消除该订单
                continue
            orders[merchant] = t
        self.orders = orders
        return len(orders) <= 0


class Drone(Entity):
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

    def __init__(self, pos, radius, name='Drone'):
        super().__init__()
        self.position = pos
        self.last_pos = None
        self.radius = radius
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
    pass
