import numpy as np

from env.core import Agent, Task, Buyer, Merchant
from env.utils import distance


def is_collision(obj1, obj2):
    # print(obj1.state.p_pos, obj2.state.p_pos, distance(obj1.state.p_pos, obj2.state.p_pos), (obj1.size + obj2.size))
    return distance(obj1.state.p_pos, obj2.state.p_pos) <= (obj1.size + obj2.size)


class Scenario:
    def __init__(self, args):
        # add agents
        self.agents = [Agent() for _ in range(args.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.name = 'agent %d'%i
            agent.collide = True,
            agent.movable = True
            agent.size = 0.1
            agent.color = (89, 89, 217)
        # add tasks
        self.tasks = [Task(buyer=Buyer(), merchant=Merchant()) for _ in range(args.num_tasks)]
        for i, task in enumerate(self.tasks):
            task.name = '%d' % i
            task.buyer.name = 'buyer %d' % i
            task.buyer.collide = False
            task.buyer.movable = False
            task.buyer.color = (255, 204, 153)
            task.merchant.name = 'merchant %d' % i
            task.merchant.collide = False
            task.merchant.movable = False
            task.merchant.color = (153, 153, 255)
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
        self.range_p = (-5, +5)
        # the size of scenario (for rendering)
        self.size = (800, 800)
        # make initial conditions
        self.reset()

    @property
    def entities(self):
        return self.agents

    def reset(self):
        # random properties for agents
        for i, agent in enumerate(self.agents):
            agent.state.p_pos = np.random.uniform(*self.range_p, self.dim_p)
            agent.state.p_vel = np.zeros(self.dim_p)
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
        scheme = np.argmax(scheme, axis=1)
        # print(scheme.shape, scheme)
        for i, task in zip(scheme, self.tasks):
            agent = self.agents[i]
            task.agent = agent
            agent.tasks.append(task)

    def reward_meta(self):
        rew = 0.0
        for task in self.tasks:
            if task.is_finished(): continue
            m = task.merchant
            b = task.buyer
            a = task.agent
            if not m.occupied:
                rew -= np.sqrt(np.sum(np.square(a.state.p_pos - m.state.p_pos)))
                rew -= np.sqrt(np.sum(np.square(m.state.p_pos - b.state.p_pos)))
            else:
                rew -= np.sqrt(np.sum(np.square(a.state.p_pos - b.state.p_pos)))
        return rew

    def reward(self, agent):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0.0
        for task in agent.tasks:
            if task.is_finished(): continue
            m = task.merchant
            b = task.buyer
            if not m.occupied:
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - m.state.p_pos)))
                rew -= np.sqrt(np.sum(np.square(m.state.p_pos - b.state.p_pos)))
                m.occupied = is_collision(agent, m)
            else:
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - b.state.p_pos)))
                b.occupied = is_collision(agent, b)
        if agent.collide:
            for a in self.agents:
                if is_collision(a, agent): rew -= 1.0
        return rew

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
            merchant = task.merchant
            if merchant.occupied:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(merchant.state.p_pos - agent.state.p_pos)
            buyer = task.buyer
            if buyer.occupied:
                entity_pos.append(np.zeros(self.dim_p))
            else:
                entity_pos.append(buyer.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def done(self, agent):
        for task in agent.tasks:
            if not task.is_finished():
                return False
        return True

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

        obs_n, reward_n, done_n = [], [], []
        for agent in self.agents:
            obs_n.append(self.observation(agent))
            reward_n.append(self.reward(agent))
            done_n.append(self.done(agent))
        reward_beta = self.reward_meta()
        reward_n = [sum(reward_n) for _ in self.agents]
        return np.array(obs_n), (reward_n, reward_beta), done_n, {}

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action + noise
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
            if not entity.movable: continue
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