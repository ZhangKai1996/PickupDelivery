import gym
import numpy as np
from gym import spaces

from env.core import Agent, Order, Destination, Stone
from env.utils import *
from env.rendering import EnvRender


class CityEnv(gym.Env):
    def __init__(self, args):
        self.agents = [Agent(name='agent%d' % i) for i in range(args.num_agents)]
        self.stones = [Stone(name='stone%d' % i) for i in range(args.num_stones)]
        self.orders = [Order(name='order%d' % i,
                             buyer=Destination(name='b%d' % i),
                             merchant=Destination(name='m%d' % i))
                       for i in range(args.num_orders)]
        self.others = None

        self.dim_p = 2
        self.args = args
        self.size = args.size
        self.shared_reward = True
        self.clock = 0

        # configure spaces
        self.act_space_meta = spaces.Discrete(args.num_agents)
        self.obs_space_meta = spaces.Box(low=-1, high=+1, shape=(12,), dtype=np.float32)
        self.act_space_ctrl = spaces.Discrete(4)
        self.obs_space_ctrl = spaces.Box(low=-1, high=+1, shape=(12,), dtype=np.float32)

        self.cv_render = None

    def reset(self, render=False, **kwargs):
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

        poses = positions[:self.args.num_orders * 2]  # random properties for orders
        self.others = positions[self.args.num_orders * 2:]
        for i, order in enumerate(self.orders):
            order.clear()
            coord = state2coord(poses[i*2], self.size)
            order.buyer.set_state(state=np.array(coord))
            coord = state2coord(poses[i*2+1], self.size)
            order.merchant.set_state(state=np.array(coord))

        if render:
            if self.cv_render is None:
                self.cv_render = EnvRender(self)
            self.cv_render.initialize()
        return True

    def step(self, action_n, **kwargs):
        self.clock += 1
        for i, agent in enumerate(self.agents):
            if not agent.is_empty():
                action = np.argmax(action_n[i])
                self.__execute_action(agent, action)

        next_obs_n, reward_n, done_n, terminated_n = [], [], [], []
        for agent in self.agents:
            reward, done, terminated = self.__reward(agent)
            next_obs_n.append(self.__observation(agent))
            reward_n.append(reward)
            done_n.append(done)
            terminated_n.append(terminated)
        # reward_n = [sum(reward_n) for _ in self.agents]
        return np.array(next_obs_n), reward_n, done_n, any(terminated_n)

    def __execute_action(self, agent, action):
        if not agent.movable: return False

        motions = np.array([[+0, +1], [-1, +0], [+1, +0], [+0, -1]])
        new_state = agent.state + motions[action]
        agent.set_last_state(state=agent.state)
        if (new_state[0] < 0 or new_state[0] >= self.size or
                new_state[1] < 0 or new_state[1] >= self.size):
            return True
        agent.set_state(state=new_state)
        agent.update()
        return False

    def __reward(self, agent):
        # Agent are penalized for collisions by stones
        for stone in self.stones:
            if distance(stone.state, agent.state) <= 0:
                return -100.0, False, True
        # Agent are penalized for collisions between agents
        for other in self.agents:
            if other == agent:
                continue
            if distance(other.state, agent.state) <= 0:
                return -100.0, False, True

        # Agent are rewarded based on minimum agent distance to each landmark
        done, rew, dists = [], 0.0, []
        for order in agent.orders:
            if order.is_finished():
                done.append(True)
                continue

            if not order.is_picked():
                dist = distance(order.merchant.state, agent.state)
                if dist <= 0:
                    order.merchant.update(clock=self.clock)
                    rew = 1.0
                dists.append(dist)
            else:
                dist = distance(order.buyer.state, agent.state)
                if dist <= 0:
                    order.buyer.update(clock=self.clock)
                    rew = 1.0
                dists.append(dist)
            done.append(order.is_finished())

        if all(done): rew = +100.0
        if rew <= 0.0 and len(dists) > 0: rew -= min(dists) * 0.1
        return rew, all(done), False

    def observation_meta(self):
        entity_pos = [agent.state / self.size for agent in self.agents]

        obs_n = []
        for order in self.orders:
            obs = np.concatenate([order.merchant.state, order.buyer.state] + entity_pos)
            obs_n.append(obs)
        return np.array(obs_n)

    def observation(self):
        return np.array([self.__observation(agent) for agent in self.agents])

    def __observation(self, agent):
        pos = agent.state
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for order in agent.orders:
            if order.is_finished():
                entity_pos.append(np.zeros((self.dim_p + 1, )))
                entity_pos.append(np.zeros((self.dim_p + 1,)))
                continue

            if not order.is_picked():
                entity_pos.append(list(order.merchant.state - pos) + [float(order.merchant.occupied is not None), ])
                entity_pos.append(np.zeros((self.dim_p + 1,)))
            else:
                entity_pos.append(np.zeros((self.dim_p + 1,)))
                entity_pos.append(list(order.buyer.state - pos) + [float(order.buyer.occupied is not None), ])

        # communication of all other agents
        other_pos = [other.state - pos for other in self.agents if other != agent]
        stone_pos = [stone.state - pos for stone in self.stones]
        return np.concatenate(entity_pos + other_pos + stone_pos)

    def task_assignment(self, scheme, **kwargs):
        scheme = np.argmax(scheme, axis=1)
        for i, order in zip(scheme, self.orders):
            agent = self.agents[i]
            order.agent = agent
            agent.orders.append(order)

    def render(self, **kwargs):
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
