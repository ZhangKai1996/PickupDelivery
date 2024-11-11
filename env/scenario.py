import numpy as np

from algo.ga import run
from env.core import Agent, Task, Person, Stone
from env.utils import distance, random_generator


class Scenario:
    def __init__(self, args):
        # add agents
        self.agents = [Agent() for _ in range(args.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.name = 'agent %d' % i
            agent.size = 0.3
            agent.color = (255, 0, 255)
        # add tasks
        self.tasks = [Task(buyer=Person(), merchant=Person()) for _ in range(args.num_tasks)]
        for i, task in enumerate(self.tasks):
            task.name = '%d' % i
            task.buyer.name = 'buyer %d' % i
            task.buyer.size = 0.2
            task.buyer.color = (255, 204, 153)
            task.merchant.name = 'merchant %d' % i
            task.merchant.size = 0.2
            task.merchant.color = (153, 153, 255)
        # add barriers
        self.barriers = [Stone() for _ in range(args.num_barriers)]
        for i, barrier in enumerate(self.barriers):
            barrier.name = '%d' % i
            barrier.size = 0.2
            barrier.color = (0, 0, 0)
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
        self.sequence = {i: [] for i in range(args.num_agents)}
        # make initial conditions
        self.reset()

    def reset(self):
        self.clock = 0
        poses = []
        # random properties for agents
        for i, agent in enumerate(self.agents):
            agent.clear()
            pos = np.random.uniform(*self.range_p, self.dim_p)
            agent.set_state(pos=pos, vel=np.zeros(self.dim_p))
            agent.pre_update()
            poses.append(pos)
        # random properties for tasks
        for i, task in enumerate(self.tasks):
            task.clear()
            pos = np.random.uniform(*self.range_p, self.dim_p)
            task.buyer.set_state(pos=pos)
            poses.append(pos)
            pos = np.random.uniform(*self.range_p, self.dim_p)
            task.merchant.set_state(pos=pos)
            poses.append(pos)
        # random properties for barriers
        for i, barrier in enumerate(self.barriers):
            pos = random_generator(poses, self.range_p, self.dim_p, size=0.6)
            barrier.set_state(pos=pos)

        return np.array([self.observation(agent) for agent in self.agents])

    def task_assignment(self, scheme, test=False):
        # print(scheme)
        scheme = np.argmax(scheme, axis=1)
        # print('num:', sum(scheme), scheme)
        for i, task in zip(scheme, self.tasks):
            agent = self.agents[i]
            task.agent = agent
            agent.tasks.append(task)
        if test:
            self.shortest_sequence()

    # def reward(self, agent, is_collision):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     if is_collision:
    #         return -10.0, True
    #
    #     rew, done = -1.0, []
    #     for task in agent.tasks:
    #         is_finished = task.is_finished()
    #         done.append(is_finished)
    #         if is_finished: continue
    #         p = task.buyer if task.is_picked() else task.merchant
    #         if p.occupied is None:
    #             if distance(p.state.p_pos, agent.state.p_pos) <= (p.size + agent.size):
    #                 p.update(clock=self.clock)
    #                 rew = 100.0
    #     return rew, all(done)

    def reward(self, agent, is_collision):
        if is_collision:
            return -100.0, True

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        pos = agent.state.p_pos
        done, dists = [], []
        rew = 0.0
        for task in agent.tasks:
            is_finished = task.is_finished()
            done.append(is_finished)
            if is_finished: continue
            p = task.buyer if task.is_picked() else task.merchant
            if p.occupied is None:
                dist = distance(p.state.p_pos, pos)
                if dist <= (p.size + agent.size):
                    p.update(clock=self.clock)
                    rew = 100.0
                else:
                    dists.append(dist)
        if rew <= 0.0 and len(dists) > 0: rew = -min(dists)
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
        pos = agent.state.p_pos
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for task in self.tasks:  # world.entities:
            if task not in agent.tasks:
                entity_pos.append(np.zeros(self.dim_p))
                entity_pos.append(np.zeros(self.dim_p))
                continue
            # merchants' relative position
            m = task.merchant
            if m.occupied is not None:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(m.state.p_pos - pos)
            # buyers' relative position
            b = task.buyer
            if b.occupied is not None:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(b.state.p_pos - pos)
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - pos)
        barrier_pos = []
        for barrier in self.barriers:
            barrier_pos.append(barrier.state.p_pos - pos)
        return np.concatenate(
            [agent.state.p_vel, pos] +
            entity_pos + other_pos + barrier_pos
        )

    def shortest_sequence(self):
        self.sequence = {}
        for i, agent in enumerate(self.agents):
            pos = agent.state.p_pos
            poses = [pos, pos]
            labels = [(0, 'P'), (0, 'D')]
            for j, task in enumerate(agent.tasks):
                poses.append(task.merchant.state.p_pos)
                poses.append(task.buyer.state.p_pos)
                labels.append((j+1, 'P'))
                labels.append((j+1, 'D'))
            sequence, dist = run(data=[np.array(poses), labels])
            self.sequence[i] = [poses[seq] for seq in sequence]
            agent.shortest_dist = dist

    def repair_tasks(self, test=False):
        if not all([task.is_finished() for task in self.tasks]):
            return
        poses = [barrier.state.p_pos for barrier in self.barriers]
        for task in self.tasks:
            if not task.is_finished():
                continue
            agent = task.agent
            task.clear()
            pos = random_generator(poses, self.range_p, self.dim_p, size=0.6)
            task.buyer.set_state(pos=pos)
            pos = random_generator(poses, self.range_p, self.dim_p, size=0.6)
            task.merchant.set_state(pos=pos)
            task.agent = agent
        if test:
            self.shortest_sequence()

    # update state of the world
    def step(self, action_n, test=False, **kwargs):
        self.clock += 1
        self.repair_tasks(test=test)
        # set action for each agent
        lst = []
        for i, agent in enumerate(self.agents):
            if agent.is_empty():
                lst.append(False)
                continue
            agent.pre_update()
            self.__set_action(action_n[i], agent, **kwargs)
            lst.append(self.__apply_action(agent))
            agent.update()
        obs_n, reward_n, done_n = [], [], []
        for agent, is_collision in zip(self.agents, lst):
            obs_n.append(self.observation(agent))
            reward, done = self.reward(agent, is_collision)
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
        p_force = agent.action * agent.u_range + noise
        # p_force = agent.action + noise

        p_vel = agent.state.p_vel * (1 - self.damping)
        p_vel += (p_force / agent.mass) * self.dt
        if agent.max_speed is not None:
            speed = np.sqrt(np.square(p_vel[0]) + np.square(p_vel[1]))
            if speed > agent.max_speed:
                p_vel = p_vel / speed * agent.max_speed
        p_pos = agent.state.p_pos + p_vel * self.dt
        agent.state.p_vel = p_vel[:]
        agent.state.p_pos = p_pos[:]
        if (not (self.range_p[0] <= p_pos[0] <= self.range_p[1]) or
                not (self.range_p[0] <= p_pos[1] <= self.range_p[1])):
            # agent.state.p_vel = np.zeros_like(p_vel)
            return True
        for entity in self.barriers + self.agents:
            if entity == agent:
                continue
            if distance(p_pos, entity.state.p_pos) <= (agent.size + entity.size):
                # agent.state.p_vel = np.zeros_like(p_vel)
                return True
        return False

    # def __apply_action(self, agent):
    #     # set applied forces
    #     noise = np.random.randn(*agent.action.shape) * agent.u_noise if agent.u_noise else 0.0
    #     p_force = agent.action + noise
    #
    #     agent.state.p_vel = agent.state.p_vel * (1 - self.damping)
    #     agent.state.p_vel += (p_force / agent.mass) * self.dt
    #     if agent.max_speed is not None:
    #         speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
    #         if speed > agent.max_speed:
    #             agent.state.p_vel = agent.state.p_vel / speed * agent.max_speed
    #     agent.state.p_pos += agent.state.p_vel * self.dt
