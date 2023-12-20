import numpy as np
import gym
from gym import spaces

from env.scenario import Scenario
from env.rendering import CVRender


class CityEnv(gym.Env):
    def __init__(self, args):
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
        if self.discrete_action_space:
            self.act_space_ctrl = spaces.Discrete(scenario.dim_p * 2 + 1)
        else:
            self.act_space_ctrl = spaces.Box(
                low=-scenario.agents[0].u_range,
                high=scenario.agents[0].u_range,
                shape=(scenario.dim_p,),
                dtype=np.float32
            )
        self.act_space_meta = spaces.Discrete(args.num_agents)
        dim_obs_ctrl = scenario.observation(scenario.agents[0]).shape[0]
        self.obs_space_ctrl = spaces.Box(low=-np.inf, high=+np.inf, shape=(dim_obs_ctrl,), dtype=np.float32)
        dim_obs_meta = scenario.observation_meta().shape[-1]
        self.obs_space_meta = spaces.Box(low=-np.inf, high=+np.inf, shape=(dim_obs_meta,), dtype=np.float32)

        self.scenario = scenario
        self.cv_render = None

    def task_assignment(self, scheme):
        self.scenario.task_assignment(scheme)

    def observation_meta(self):
        return self.scenario.observation_meta()

    def reset(self, **kwargs):
        return self.scenario.reset()

    def _set_action(self, action, agent):
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
            self._set_action(action_n[i], agent)
        # advance scenario state
        return self.scenario.step()

    def render(self, mode='human', show=False):
        if self.cv_render is None:
            self.cv_render = CVRender(self)
        self.cv_render.draw(show=show)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
