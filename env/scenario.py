import numpy as np
from env.core import World, Agent, Landmark, Task


def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


class Scenario:
    def __init__(self):
        world = World()
        # set any world properties first
        num_agents = 3
        num_tasks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add pickups
        world.landmarks = [Landmark() for _ in range(num_tasks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
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
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.00, 0.56, 0.56])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.occupied = False

    def done(self):
        for landmark in self.world.landmarks:
            if not landmark.occupied:
                return False
        return True

    # -634.28, 26%-28%
    def reward(self, agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        world = self.world
        rew = 0.0
        for p in self.world.landmarks:
            if p.occupied: continue
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - p.state.p_pos))) for a in world.agents]
            rew -= min(dists)
            if is_collision(agent, p):
                p.occupied = True
        # Penalized for collisions
        if agent.collide:
            for a in world.agents:
                if is_collision(a, agent):
                    rew -= 1.0
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
