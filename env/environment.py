import numpy as np
import gym
from gym import spaces

from env.scenario import Scenario
from env.rendering import CVRender


class CityEnv(gym.Env):
    def __init__(self, args):
        self.n = args.num_agents
        scenario = Scenario(args)

        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = scenario.discrete_action if hasattr(scenario, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = scenario.collaborative if hasattr(self, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space_n = []
        self.observation_space_n = []
        for agent in scenario.agents:
            # physical action space
            if self.discrete_action_space:
                action_space = spaces.Discrete(scenario.dim_p * 2 + 1)
            else:
                action_space = spaces.Box(
                    low=-agent.u_range,
                    high=+agent.u_range,
                    shape=(scenario.dim_p,),
                    dtype=np.float32
                )
            if agent.movable:
                self.action_space_n.append(action_space)
            # observation space
            self.observation_space_n.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(100,100), dtype=np.float32)
            )
        self.scenario = scenario
        self.cv_render = None

    def reset(self, **kwargs):
        return self.scenario.reset()

    def _set_action(self, action, agent, action_space):
        agent.action = np.zeros(self.scenario.dim_p)
        action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action = np.zeros(self.scenario.dim_p)
                # process discrete action
                if action[0] == 1: agent.action[0] = -1.0
                if action[0] == 2: agent.action[0] = +1.0
                if action[0] == 3: agent.action[1] = -1.0
                if action[0] == 4: agent.action[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action[0] += action[0][1] - action[0][2]
                    agent.action[1] += action[0][3] - action[0][4]
                else:
                    agent.action = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action *= sensitivity
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action_n):
        # set action for each agent
        for i, agent in enumerate(self.scenario.agents):
            self._set_action(action_n[i], agent, self.action_space_n[i])
        # advance scenario state
        return self.scenario.step()

    def render(self, mode='human', show=False):
        if self.cv_render is None:
            self.cv_render = CVRender(self)
        self.cv_render.draw(show=show)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
