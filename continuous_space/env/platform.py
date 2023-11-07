from typing import Dict, Tuple, List

import numpy as np

from common.utils import distance, region_segmentation, is_overlap, bbox
from .visual import CVRender


class Drone:
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


class Platform:
    def __init__(self, size=10, radius=.5, time_flow=True, **kwargs):
        """
        Drone/Merchant/Buyer都用圆形来表示，半径均为radius；而Wall用正方形表示，边长为2*radius；
        """
        print('---------------Pickup-Delivery environment--------------------')
        print(kwargs)
        self.size: int = size
        self.radius: float = radius
        self.clock: int = 0
        self.time_flow = time_flow

        pos_dict = region_segmentation(kwargs, size, radius)
        self.buyers_stack = list(pos_dict['buyers'])
        self.buyers = []
        self.merchants = pos_dict['merchants']
        self.walls = pos_dict['walls']
        self.drones = [Drone(name='Drone_{}'.format(i + 1), pos=pos, radius=radius)
                       for i, pos in enumerate(pos_dict['drones'])]

        self.cv_render = None

    def global_info(self):
        return 'Clock: {}, Drone: {}, Buyer:{}, Merchant:{}'.format(
            self.clock, len(self.drones), len(self.buyers), len(self.merchants)
        )

    def get_obs(self, shape='circle'):
        width = height = self.radius
        if shape == 'circle':
            return []
        if shape == 'rectangle':
            return [bbox(wall, delta=(width, height)) for wall in self.walls]
        if shape == 'boundary':
            return [width, height, self.size-width, self.size-height]
        raise NotImplementedError

    def __update_buyers(self, ):
        buyers = []
        for buyer in self.buyers:
            if not buyer.update_orders(self.clock):
                buyers.append(buyer)

        if self.time_flow:
            buyers += self.__generate_buyers(max_buyers=5)
        else:
            buyers += self.__generate_buyers(max_buyers=len(self.buyers_stack), random=False)
        self.buyers = buyers

    def __generate_buyers(self, max_buyers=3, random=True):
        if len(self.buyers_stack) <= 0:
            return []

        clock = self.clock
        num_new_buyer = np.random.randint(0, 5) if random else max_buyers
        print('T:{:>4d}, New: {}'.format(clock, num_new_buyer))

        new_buyers = []
        for i in range(num_new_buyer):
            address = self.buyers_stack.pop(0)
            buyer = Buyer(name='Buyer_{}_{}'.format(clock, i + 1), address=address)
            new_buyers.append(buyer)
            m_id = np.random.randint(0, len(self.merchants))
            buyer.buy(m_id, clock)
            print('\t>>>', buyer.address, m_id, buyer.orders)

        print('\tTotal: {:>3d}'.format(len(new_buyers)))
        return new_buyers

    def __detect_collision(self):
        num_drone = len(self.drones)
        for i in range(num_drone):
            drone = self.drones[i]
            if drone.is_collision:
                continue

            # Collision between drones
            for j in range(i + 1, num_drone):
                drone.detect(self.drones[j])

            if drone.is_collision:
                continue
            # Collision between drones and walls
            drone.detect(self.walls)

    def step(self, actions=None, duration=1.0):
        self.clock += 1
        self.__update_buyers()
        if actions is None:
            return

        for i, (drone, action) in enumerate(zip(self.drones, actions)):
            drone.execute_action(action, self.size, duration=duration)
        self.__detect_collision()

        print(self.clock)
        for i, drone in enumerate(self.drones):
            action = actions[i]
            print('\t>>>', drone, 'Delta: ({:>+5.3f},{:>+5.3f})'.format(action[0], action[1]))

    def render(self, show=False, **kwargs):
        if self.cv_render is None:
            self.cv_render = CVRender(self)
            self.cv_render.draw_dynamic(**kwargs)
        self.cv_render.draw(show=show)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
