import numpy as np
from env.core import World, Agent, Landmark, Task


def is_collision(agent1, agent2):
    if agent1 == agent2:
        return False

    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


class Scenario:
    def __init__(self):
        world = World()
        # set any world properties first
        num_agents = 3
        num_tasks = 6
        num_barriers = 10
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add pickups
        world.pickups = [Landmark() for _ in range(num_tasks)]
        for i, landmark in enumerate(world.pickups):
            landmark.name = 'pickup %d' % i
            landmark.collide = False
            landmark.movable = False
        # add pickups
        world.deliveries = [Landmark() for _ in range(num_tasks)]
        for i, landmark in enumerate(world.deliveries):
            landmark.name = 'delivery %d' % i
            landmark.collide = False
            landmark.movable = False
        # add pickups
        world.barriers = [Landmark() for _ in range(num_barriers)]
        for i, landmark in enumerate(world.barriers):
            landmark.name = 'barrier %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
        self.tasks = None
        self.world = world
        # make initial conditions
        self.reset()

    def reset(self):
        # random properties and initial states for agents
        world = self.world
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        # random properties and initial states for landmarks
        for i, landmark in enumerate(world.pickups):
            landmark.color = np.array([0.00, 0.56, 0.56])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.occupied = False
        # random properties and initial states for landmarks
        for i, landmark in enumerate(world.deliveries):
            landmark.color = np.array([1.00, 0.85, 0.73])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.occupied = False
        # random properties and initial states for landmarks
        for i, landmark in enumerate(world.barriers):
            landmark.color = np.array([0.30, 0.30, 0.30])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.tasks = [Task(p, d) for p, d in zip(world.pickups, world.deliveries)]

    def done(self):
        for task in self.tasks:
            if not task.is_over():
                return False
        return True

    # -634.28, 26%-28%
    def reward(self, agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        world = self.world
        rew = 0.0
        for task in self.tasks:
            if task.is_over(): continue
            p, d = task.p, task.d
            if not p.occupied:
                if is_collision(agent, p):
                    p.occupied = True
                    task.agent = agent
                else:
                    dists = [np.sqrt(np.sum(np.square(a.state.p_pos - p.state.p_pos))) for a in world.agents]
                    rew -= min(dists)

            if not p.occupied or d.occupied:
                continue
            if is_collision(agent, d) and task.agent == agent:
                d.occupied = True
            else:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - d.state.p_pos))) for a in world.agents]
                rew -= min(dists)
        # Penalized for collisions
        if agent.collide:
            for a in world.agents:
                if is_collision(a, agent):
                    rew -= 1.0
            for b in world.barriers:
                if is_collision(b, agent):
                    rew -= 100.0
        return rew

    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        world = self.world
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if entity.occupied:
                entity_pos.append(np.zeros(self.world.dim_p))
            else:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel, agent.state.p_pos] + entity_pos + other_pos)
