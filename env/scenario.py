import numpy as np
from rtree.index import Index, Property

from env.core import *
from env.utils import distance, bbox


class Scenario:
    def __init__(self, args):
        # add agents
        self.agents = [Agent() for _ in range(args.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.name = 'agent %d' % i
            agent.color = (255, 0, 255)
            agent.size = 0.40
        # add tasks
        self.tasks = [Task(buyer=Buyer(), merchant=Merchant())
                      for _ in range(args.num_tasks)]
        for i, task in enumerate(self.tasks):
            task.name = '%d' % i
            task.buyer.name = 'buyer %d' % i
            task.buyer.color = (255, 204, 153)
            task.merchant.name = 'merchant %d' % i
            task.merchant.color = (153, 153, 255)
        # add barriers
        self.barriers = [Barrier() for _ in range(32)]
        for i, barrier in enumerate(self.barriers):
            barrier.name = 'barrier %d'
            barrier.size = 0.40
            barrier.color = (0, 0, 0)
        # position dimensionality
        self.dim_p = 2
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # global property
        self.collaborative = True
        self.range_p = (-10.0, +10.0)
        # the size of scenario (for rendering)
        self.size = (1200, 1200)
        # make initial conditions
        self.reset()

    @property
    def entities(self):
        return self.agents + self.barriers

    def reset(self):
        # random properties for agents
        for agent in self.agents:
            agent.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            agent.state.p_vel = np.zeros(self.dim_p)
            agent.last_state.set(other=agent.state)
            agent.tasks = []
        # random properties for landmarks
        for task in self.tasks:
            task.agent = None
            buyer = task.buyer
            buyer.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            buyer.state.p_vel = np.zeros(self.dim_p)
            buyer.occupied = False
            merchant = task.merchant
            merchant.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            merchant.state.p_vel = np.zeros(self.dim_p)
            merchant.occupied = False
        # random properties for barriers
        for i, barrier in enumerate(self.barriers):
            barrier.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            barrier.state.p_vel = np.zeros(self.dim_p)
        return np.array([self.observation(agent) for agent in self.agents])

    def task_assignment(self, scheme):
        agent_id = np.argmax(scheme, axis=1)
        for i, task in zip(agent_id, self.tasks):
            task.assign_to(self.agents[i])

    def observation_meta(self):
        obs_n = []
        for task in self.tasks:
            entity_pos = []
            for agent in self.agents:
                entity_pos.append(task.pick_pos - agent.state.p_pos)
            obs = np.concatenate([task.pick_pos, ] + entity_pos)
            obs_n.append(obs)
        return np.array(obs_n)

    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        task_pos = []
        for task in self.tasks:  # world.entities:
            if task not in agent.tasks:
                task_pos.append(np.zeros(self.dim_p))
                continue

            if task.is_picked:
                task_pos.append(np.zeros(self.dim_p))
            else:
                task_pos.append(task.pick_pos - agent.state.p_pos)
        # communication of all other agents
        agent_pos = []
        for other in self.agents:
            if other is agent:
                continue
            agent_pos.append(other.state.p_pos - agent.state.p_pos)
        # communication of all barriers
        barrier_pos = []
        for barrier in self.barriers:
            barrier_pos.append(barrier.state.p_pos - agent.state.p_pos)
        obs = [agent.state.p_pos, ] + task_pos + agent_pos + barrier_pos
        return np.concatenate(obs)

    def reward(self, agent, idx):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        done = True
        rew = distance(agent.state.p_pos, agent.last_state.p_pos)
        for task in agent.tasks:
            if task.is_picked: continue
            if task.check_occupied(agent):
                rew = 10.0
                continue
            done = False

        box = bbox(agent.state.p_pos, delta=(agent.size, agent.size))
        for i in idx.intersection(box):
            if agent.is_collision(self.entities[i]):
                rew -= 10.0
        return rew, done

    # update state of the world
    def step(self):
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)

        idx = self.__build_rtree()
        obs_n, reward_n, done_n = [], [], []
        for agent in self.agents:
            agent.update_mass()
            obs_n.append(self.observation(agent))
            rew, done = self.reward(agent, idx)
            reward_n.append(rew)
            done_n.append(done)
        # reward_n = [sum(reward_n) for _ in self.agents]
        return np.array(obs_n), reward_n, done_n, {}

    def __build_rtree(self):
        idx = Index(properties=Property(dimension=self.dim_p))
        for i, entity in enumerate(self.entities):
            idx.insert(i, bbox(entity.state.p_pos))
        return idx

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, entity in enumerate(self.entities):
            if not hasattr(entity, 'action'):
                continue
            if entity.movable:
                noise = np.random.randn(*entity.action.shape)
                noise = noise * entity.u_noise if entity.u_noise else 0.0
                p_force[i] = entity.action + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a: continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None: p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None: p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue

            entity.last_state.set(other=entity.state)
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself

        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
