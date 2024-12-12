import gym
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
        self.clock = 0

        # configure spaces
        self.act_space_ta = spaces.Discrete(args.num_agents)
        self.obs_space_ta = spaces.Box(low=-args.size, high=+args.size,
                                       shape=(2 * args.num_agents,),
                                       dtype=np.float32)
        self.obs_space_pf = spaces.Box(low=-args.size, high=+args.size,
                                       shape=(4,),
                                       dtype=np.float32)
        self.act_space_tp = spaces.Discrete(4)
        self.obs_space_tp = spaces.Box(low=-args.size, high=+args.size,
                                       shape=(2*args.num_agents,),
                                       dtype=np.float32)
        self.cv_render = None

    def dot(self):
        return [sum([int(order.is_finished()) for order in agent.orders])
                for agent in self.agents]

    def reset(self, render=False, **kwargs):
        self.clock = 0

        positions = list(range(self.size * self.size))
        np.random.shuffle(positions)

        poses = positions[:self.args.num_agents*2]  # random properties for agents
        positions = positions[self.args.num_agents*2:]
        for i, agent in enumerate(self.agents):
            agent.clear()
            coord = state2coord(poses[i*2], self.size)
            agent.set_state(state=np.array(coord))
            coord = state2coord(poses[i*2+1], self.size)
            agent.set_end_state(state=np.array(coord))

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

    def __execute_action(self, agent, action):
        # if agent.is_empty(): return False
        if not agent.movable: return False

        motions = np.array([[+0, +1], [-1, +0], [+1, +0], [+0, -1]])
        new_state = agent.state + motions[action]
        agent.set_last_state(state=agent.state)
        if (new_state[0] < 0 or new_state[0] >= self.size or
                new_state[1] < 0 or new_state[1] >= self.size):
            return False
        agent.set_state(state=new_state)
        return agent.update()

    def step(self, action_n, verbose=False, **kwargs):
        self.clock += 1

        obs_n, arr_n = [], []
        for i, agent in enumerate(self.agents):
            obs_n.append(self.__observation_tp(agent))
            action = np.argmax(action_n[i])
            arr_n.append(self.__execute_action(agent, action))

        status = []
        next_obs_n, reward_n, done_n, terminated_n = [], [], [], []
        for i, (agent, obs_tp) in enumerate(zip(self.agents, obs_n)):
            done, done_tp = self.__update_orders(agent)
            next_obs_tp = self.__observation_tp(agent)
            reward, terminated = self.__reward(obs_tp, next_obs_tp, agent, done_tp, arr_n[i])
            if verbose:
                print('\t', obs_tp, next_obs_tp, reward, arr_n[i], done_tp, done)
            status.append(done)
            done_n.append(done_tp)
            reward_n.append(reward)
            next_obs_n.append(next_obs_tp)
            terminated_n.append(terminated)
        return np.array(next_obs_n), reward_n, done_n, any(terminated_n), all(status)

    def __update_orders(self, agent):
        done = []
        for order in agent.orders:
            if order.is_finished():
                done.append(True)
                continue

            if not order.is_picked():
                if distance(order.merchant.state, agent.state) <= 0:
                    order.merchant.update(clock=self.clock)
            elif distance(order.buyer.state, agent.state) <= 0:
                    order.buyer.update(clock=self.clock)
            done.append(order.is_finished())
        done = all(done) and distance(agent.end_state, agent.state) <= 0
        return done, agent.is_sequence_over()

    def __reward(self, obs, next_obs, agent, done_tp, is_arrived):
        # Agent are rewarded for arriving the goal state
        if done_tp: return +10.0, False
        if is_arrived: return +0.0, False
        # Agent are penalized for collisions by stones
        for stone in self.stones:
            if distance(stone.state, agent.state) <= 0:
                return -100.0, True
        # Agent are penalized for collisions between agents
        for other in self.agents:
            if other == agent: continue
            if distance(other.state, agent.state) <= 0:
                return -100.0, True
        # Agent are rewarded based on minimum agent distance to each landmark
        dists1 = []
        for i, x in enumerate(obs):
            if i % 2 == 1:
                if x == 0 and obs[i-1] == 0:
                    continue
                dists1.append(abs(x) + abs(obs[i-1]))
        dists2 = []
        for i, x in enumerate(next_obs):
            if i % 2 == 1:
                if x == 0 and obs[i-1] == 0:
                    continue
                dists2.append(abs(x) + abs(next_obs[i-1]))
        rew = -0.5 if min(dists1) > min(dists2) else -1.0
        return rew, False

    def __observation_ta(self):
        obs_n = []
        for order in self.orders:
            obs = []
            for agent in self.agents:
                obs.append(order.merchant.state - agent.state)
                obs.append(order.buyer.state - agent.state)
            obs_n.append(np.concatenate(obs))
        return np.array(obs_n)

    def __observation_pf(self, agent):
        obs = []
        for order in agent.orders:
            if order.is_finished():
                obs.append(np.zeros((self.dim_p*2, )))
                obs.append(np.zeros((self.dim_p*2, )))
                continue

            if not order.is_picked():
                obs.append(np.concatenate([
                    order.merchant.state - agent.state,
                    order.merchant.state - agent.end_state
                ]))
            else:
                obs.append(np.zeros((self.dim_p*2, )))
            obs.append(np.concatenate([
                order.buyer.state - agent.state,
                order.buyer.state - agent.end_state
            ]))
        return np.array(obs)

    def __observation_tp(self, agent):
        pos = agent.state
        # get positions of all entities in this agent's reference frame
        entity_pos = [agent.cur_state() - pos, ]
        # communication of all other agents
        other_pos = [other.state - pos for other in self.agents if other != agent]
        stone_pos = [stone.state - pos for stone in self.stones]
        return np.concatenate(entity_pos + other_pos + stone_pos)

    def observation(self, label='ta'):
        if label == 'ta':
            return self.__observation_ta()
        if label == 'pf':
            return np.array([self.__observation_pf(agent) for agent in self.agents])
        if label == 'tp':
            return np.array([self.__observation_tp(agent) for agent in self.agents])
        raise NotImplementedError

    def path_planning(self, sequences=None):
        for i, agent in enumerate(self.agents):
            seq = None if sequences is None else sequences[i]
            agent.set_sequence(seq)

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
