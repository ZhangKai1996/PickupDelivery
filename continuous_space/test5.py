import numpy as np

from continuous_space.algo import *
from continuous_space.env import Platform

from parameters import args as args_


def test_env(args, env):
    step = 0
    while True:
        actions = np.random.uniform(-1.0, 1.0, size=(args.num_agents, 2))
        env.step(actions)
        env.render()
        step += 1
        if step >= 100:
            break


if __name__ == '__main__':
    env_ = Platform(
        args_.size,
        args_.radius,
        time_flow=args_.time_flow,
        drones=args_.num_agents,
        merchants=args_.num_merchants,
        buyers=args_.num_buyers,
        walls=args_.num_walls
    )
    # todo: 程序可运行，待完善
    VOPlanner(env_).simulate(num_iter=100, step=.1)
    # test_env(args_, env_)

