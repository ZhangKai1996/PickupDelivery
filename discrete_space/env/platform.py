from typing import Dict, Tuple, List

import numpy as np

from .visual import CVRender


class Drone:
    name: str
    state: str = 'Empty'
    position: Tuple[int, int]
    last_pos: Tuple[int, int]
    distance: int = 0
    endurance: List[int] = [60, 60]
    orders: Dict[str, int] = {}
    is_collision: bool = False

    def __init__(self, pos, name='Drone'):
        self.position = pos
        self.last_pos = None
        self.name = name

    def update_state(self):
        if self.state == 'Collision':
            return

        if self.is_collision:
            self.state = 'Collision'
        elif len(self.orders) <= 0:
            self.state = 'Empty'
        else:
            self.state = 'Pickup'

    def execute_action(self, action: int, size: int, walls: list):
        """
        king's moves
        :param walls:
        :param size:
        :param action: [0,1,2,3,4,5,6,7,8]
        """
        old_pos = self.position
        motions = [
            (-1, 1), (0, 1), (1, 1),
            (-1, 0), (0, 0), (1, 0),
            (-1, -1), (0, -1), (1, -1)
        ]
        print(action, motions[action], old_pos, end=', ')
        [delta_x, delta_y] = motions[action]
        new_pos = (new_x, new_y) = (old_pos[0]+delta_x, old_pos[1]+delta_y)
        print(new_pos, end=', ')
        if size > new_x >= 0 and size > new_y >= 0:
            self.position = new_pos
            self.distance += 1
            self.is_collision = new_pos in walls
            self.update_state()

        print(self.position)
        self.endurance[1] -= 1
        self.last_pos = old_pos
        return new_pos

    def detect(self, drone):
        if self.is_collision or drone.is_collision:
            return

        if self.position == drone.position:
            self.is_collision = True
            drone.is_collision = True


class Buyer:
    name: str
    address: Tuple[int, int]
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
            if clock - t >= 60:  # 如果服务时间超过60s，则消除该订单
                continue
            orders[merchant] = t
        self.orders = orders
        return len(orders) <= 0


class Platform:
    def __init__(self, size=10, **kwargs):
        print('---------------Snake environment--------------------')
        print(kwargs)

        self.size: int = size
        self.clock: int = 0
        self.kwargs = kwargs

        num_states = size * size
        states = [(i, j) for j in range(size) for i in range(size)]
        np.random.shuffle(states)
        state_dict = {}
        for key, num in kwargs.items():
            if isinstance(num, float):
                num = int(num_states * num)

            state_dict[key] = states[:num]
            states = states[num:]

        self.empty_grids = states
        self.buyers = None
        self.merchants = state_dict['merchants']
        self.walls = state_dict['walls']
        self.drones = [
            Drone(name='Drone_{}'.format(i + 1), pos=state)
            for i, state in enumerate(state_dict['drones'])
        ]
        self.cv_render = None
        self.dynamic_obs = {}

    def global_info(self):
        return 'Clock: {}, Drone: {}, Buyer:{}, Merchant:{}'.format(
            self.clock, len(self.drones), len(self.buyers), len(self.merchants)
        )

    def __update_buyers(self, ):
        time_flow = self.kwargs['time_flow']

        if self.buyers is None:
            self.buyers = self.__random_generate_buyers(
                max_buyers=3 if time_flow else self.kwargs['buyers'],
                random=time_flow
            )
            return

        buyers = []
        for buyer in self.buyers:
            if not buyer.update_orders(self.clock):
                buyers.append(buyer)
            else:
                self.empty_grids.append(buyer.address)

        if time_flow:
            buyers += self.__random_generate_buyers()
        self.buyers = buyers

    def __random_generate_buyers(self, max_buyers=3, random=True):
        clock = self.clock
        empty_grids = self.empty_grids

        num_new_buyers = np.random.randint(0, max_buyers+1) if random else max_buyers
        addresses = empty_grids[:num_new_buyers]
        self.empty_grids = empty_grids[num_new_buyers:]
        print('T:{:>4d}, New: {}'.format(clock, num_new_buyers))

        buyers = []
        for i in range(num_new_buyers):
            address = addresses[i]
            buyer = Buyer(name=str(address), address=address)
            buyers.append(buyer)

            m_id = np.random.randint(0, len(self.merchants))
            buyer.buy(self.merchants[m_id], clock)
            print('\t>>>', buyer.address, self.merchants[m_id], buyer.orders)

        print('\tTotal: {:>3d}'.format(len(buyers)))
        return buyers

    def add_obstacles(self, poses=None, num=1):
        if poses is not None:
            for pos in poses:
                if pos in self.empty_grids:
                    self.walls.append(pos)
                    self.empty_grids.remove(pos)
            return

        for _ in range(num):
            pos = self.empty_grids.pop(0)
            self.walls.append(pos)

    def __detect_collision(self):
        num_drone = len(self.drones)
        for i in range(num_drone):
            drone = self.drones[i]
            for j in range(i+1, num_drone):
                drone.detect(self.drones[j])

    def step(self, actions=None):
        self.clock += 1
        self.__update_buyers()

        if actions is None:
            return

        for i, (drone, action) in enumerate(zip(self.drones, actions)):
            if drone.is_collision:
                continue
            drone.execute_action(action, self.size, self.walls)
        self.__detect_collision()

    def render(self, show=False, **kwargs):
        if self.cv_render is None:
            self.cv_render = CVRender(self)
            self.cv_render.draw_visited(**kwargs)

        self.cv_render.draw(show=show)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
