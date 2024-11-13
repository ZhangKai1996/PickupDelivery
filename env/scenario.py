from env.core import Agent, Order, Destination, Stone
from env.utils import *


class Scenario:
    def __init__(self, args):
        self.agents = [Agent(name='agent%d' % i) for i in range(args.num_agents)]
        self.stones = [Stone(name='stone%d' % i) for i in range(args.num_stones)]
        self.orders = [Order(name='order%d' % i,
                             buyer=Destination(name='buyer %d' % i),
                             merchant=Destination(name='merchant %d' % i))
                       for i in range(args.num_orders)]
        self.others = None

        self.dim_p = 2
        self.args = args
        self.size = args.size
        self.collaborative = True
        self.clock = 0

    def reset(self):
        self.clock = 0

        positions = list(range(self.size * self.size))
        np.random.shuffle(positions)
        poses = positions[:self.args.num_agents]  # random properties for agents
        positions = positions[self.args.num_agents:]
        for i, agent in enumerate(self.agents):
            agent.clear()
            coord = state2coord(poses[i], self.size)
            agent.set_state(state=np.array(coord))

        poses = positions[:self.args.num_stones]  # random properties for stones
        positions = positions[self.args.num_stones:]
        for i, stone in enumerate(self.stones):
            coord = state2coord(poses[i], self.size)
            stone.set_state(state=np.array(coord))

        poses = positions[:self.args.num_orders*2]  # random properties for orders
        self.others = positions[self.args.num_orders*2:]
        for i, order in enumerate(self.orders):
            order.clear()
            coord = state2coord(poses[i], self.size)
            order.buyer.set_state(state=np.array(coord))
            coord = state2coord(poses[i+1], self.size)
            order.merchant.set_state(state=np.array(coord))


        return np.array([self.observation(agent) for agent in self.agents])

    def task_assignment(self, scheme, **kwargs):
        scheme = np.argmax(scheme, axis=1)
        for i, order in zip(scheme, self.orders):
            agent = self.agents[i]
            order.agent = agent
            agent.orders.append(order)

    def reward(self, agent):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        done, dists = [], []
        rew = 0.0
        for order in agent.orders:
            is_finished = order.is_finished()
            done.append(is_finished)
            if is_finished: continue
            p = order.buyer if order.is_picked() else order.merchant
            if p.occupied is None:
                dist = distance(p.state, agent.state)
                if dist <= 0:
                    p.update(clock=self.clock)
                    rew = 100.0
                else:
                    dists.append(dist)
        if rew <= 0.0 and len(dists) > 0: rew = -min(dists)
        return rew, all(done)

    def observation_meta(self):
        obs_n = []
        for order in self.orders:
            p_pos_m = order.merchant.state
            p_pos_b = order.buyer.state
            entity_pos = []
            for agent in self.agents:
                entity_pos.append(p_pos_m - agent.state)
                entity_pos.append(p_pos_b - agent.state)
            obs = np.concatenate([p_pos_m, p_pos_b] + entity_pos)
            obs_n.append(obs)
        return np.array(obs_n)

    def observation(self, agent):
        pos = agent.state
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for order in self.orders:  # world.entities:
            if order not in agent.orders:
                entity_pos.append(np.zeros(self.dim_p))
                entity_pos.append(np.zeros(self.dim_p))
                continue
            # merchants' relative position
            m = order.merchant
            if m.occupied is not None:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(m.state - pos)
            # buyers' relative position
            b = order.buyer
            if b.occupied is not None:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(b.state - pos)
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other == agent: continue
            other_pos.append(other.state - pos)
        stone_pos = []
        for stone in self.stones:
            stone_pos.append(stone.state - pos)
        return np.concatenate(
            [pos, ] + entity_pos + other_pos + stone_pos
        )

    # update state of the world
    def step(self, action_n, **kwargs):
        self.clock += 1
        # set action for each agent
        for i, agent in enumerate(self.agents):
            if agent.is_empty():
                continue
            action = np.argmax(action_n[i])
            # print(agent.name, action, agent.last_state, agent.state, end=' ')
            self.__execute_action(agent, action)
            # print(agent.last_state, agent.state)

        obs_n, reward_n, done_n = [], [], []
        for agent in self.agents:
            obs_n.append(self.observation(agent))
            reward, done = self.reward(agent)
            reward_n.append(reward)
            done_n.append(done)
        # reward_n = [sum(reward_n) for _ in self.agents]
        return np.array(obs_n), reward_n, done_n, {}

    def __execute_action(self, agent, action):
        if not agent.movable: return

        motions = np.array([[+0, +1], [-1, +0], [+1, +0], [+0, -1]])
        new_state = agent.state + motions[action]
        if (new_state[0] < 0 or new_state[0] >= self.size or
                new_state[1] < 0 or new_state[1] >= self.size):
            return
        agent.set_last_state(last_state=agent.state)
        agent.set_state(state=new_state)

