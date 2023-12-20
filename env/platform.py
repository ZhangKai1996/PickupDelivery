from .utils import *


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
