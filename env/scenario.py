import numpy as np

from env.core import Agent, Task, Buyer, Merchant
from env.utils import distance


class Scenario:
    def __init__(self, args):
        # add agents
        self.agents = [Agent() for _ in range(args.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.name = 'agent %d' % i
            agent.movable = True
            agent.size = 0.10
            agent.color = (255, 0, 255)
        # add tasks
        self.tasks = [Task(buyer=Buyer(), merchant=Merchant())
                      for _ in range(args.num_tasks)]
        for i, task in enumerate(self.tasks):
            task.name = '%d' % i
            task.buyer.name = 'buyer %d' % i
            task.buyer.movable = False
            task.buyer.color = (255, 204, 153)
            task.merchant.name = 'merchant %d' % i
            task.merchant.movable = False
            task.merchant.color = (153, 153, 255)
        # position dimensionality
        self.dim_p = 2
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # global property
        self.collaborative = True
        self.range_p = (-10.0, +10.0)
        self.clock = 0
        # make initial conditions
        self.reset()

    @property
    def entities(self):
        return self.agents

    def reset(self):
        self.clock = 0
        # random properties for agents
        for i, agent in enumerate(self.agents):
            agent.clear()
            agent.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            agent.state.p_vel = np.zeros(self.dim_p)
            agent.pre_update()
            agent.tasks = []
        # random properties for landmarks
        for i, task in enumerate(self.tasks):
            task.agent = None
            buyer = task.buyer
            buyer.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            buyer.state.p_vel = np.zeros(self.dim_p)
            buyer.occupied = False
            merchant = task.merchant
            merchant.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            merchant.state.p_vel = np.zeros(self.dim_p)
            merchant.occupied = False
        return np.array([self.observation(agent) for agent in self.agents])

    def task_assignment(self, scheme):
        # print(scheme)
        scheme = np.argmax(scheme, axis=1)
        # print('num:', sum(scheme), scheme)
        for i, task in zip(scheme, self.tasks):
            agent = self.agents[i]
            task.agent = agent
            agent.tasks.append(task)

    def reward(self, agent):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        pos = agent.state.p_pos
        done, dists = [], []
        rew = 0.0
        for task in agent.tasks:
            if task.is_finished():
                done.append(True)
                continue

            p = task.buyer if task.is_picked() else task.merchant
            d = p.occupied
            if not d:
                dist = distance(p.state.p_pos, pos)
                if dist < (p.size + agent.size):
                    p.update(clock=self.clock)
                    rew = 10.0
                    d = True
                else:
                    dists.append(dist)
            done.append(d)
        if rew <= 0.0 and len(dists) > 0:
            rew = -min(dists)

        for a in self.agents:
            if a == agent: continue
            if distance(a.state.p_pos, pos) < (a.size + agent.size):
                rew -= 1.0
        return rew, all(done)

    def observation_meta(self):
        obs_n = []
        for task in self.tasks:
            p_pos_m = task.merchant.state.p_pos
            p_pos_b = task.buyer.state.p_pos
            entity_pos = []
            for agent in self.agents:
                entity_pos.append(p_pos_m - agent.state.p_pos)
                entity_pos.append(p_pos_b - agent.state.p_pos)
            obs = np.concatenate([p_pos_m, p_pos_b] + entity_pos)
            obs_n.append(obs)
        return np.array(obs_n)

    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for task in self.tasks:  # world.entities:
            if task not in agent.tasks:
                entity_pos.append(np.zeros(self.dim_p))
                entity_pos.append(np.zeros(self.dim_p))
                continue
            # merchants' relative position
            m = task.merchant
            if m.occupied:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(m.state.p_pos - agent.state.p_pos)
            # buyers' relative position
            b = task.buyer
            if b.occupied:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(b.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    # update state of the world
    def step(self, action_n, **kwargs):
        self.clock += 1
        # set action for each agent
        obs_n, reward_n, done_n = [], [], []
        for i, agent in enumerate(self.agents):
            if agent.empty():
                continue
            agent.pre_update()
            self.__set_action(action_n[i], agent, **kwargs)
            self.__apply_action(agent)
            agent.update()

        for agent in self.agents:
            obs_n.append(self.observation(agent))
            reward, done = self.reward(agent)
            reward_n.append(reward)
            done_n.append(done)
        # reward_n = [sum(reward_n) for _ in self.agents]
        return np.array(obs_n), reward_n, done_n, {}

    def __set_action(self, action, agent, force_discrete_action=False, discrete_action_space=False):
        agent.action = np.zeros(self.dim_p)
        if agent.movable:
            # physical action
            if force_discrete_action:
                d = np.argmax(action)
                action[:] = 0.0
                action[d] = 1.0
            if discrete_action_space:
                agent.action[0] += action[1] - action[2]
                agent.action[1] += action[3] - action[4]
            else:
                agent.action = action
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action *= sensitivity

    # gather agent action forces
    def __apply_action(self, agent):
        # set applied forces
        noise = np.random.randn(*agent.action.shape) * agent.u_noise if agent.u_noise else 0.0
        p_force = agent.action + noise

        agent.state.p_vel = agent.state.p_vel * (1 - self.damping)
        agent.state.p_vel += (p_force / agent.mass) * self.dt
        if agent.max_speed is not None:
            speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
            if speed > agent.max_speed:
                agent.state.p_vel = agent.state.p_vel / speed * agent.max_speed
        agent.state.p_pos += agent.state.p_vel * self.dt
