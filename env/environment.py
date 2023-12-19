import gym

from env.core import Drone
from env.rendering import CVRender


class CityEnv(gym.Env):
    def __init__(self, args):
        self.riders = [Drone() for _ in range(args.num_riders)]
        self.tasks = []
        self.cv_render = None

    def reset(self, **kwargs):
        pass

    def step(self, action):
        pass

    def render(self, mode='human', show=False):
        if self.cv_render is None:
            self.cv_render = CVRender(self)
        self.cv_render.draw(show=show)

    def close(self):
        pass
