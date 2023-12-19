from typing import Dict, Tuple, List

from .utils import *
from .rendering import CVRender


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
