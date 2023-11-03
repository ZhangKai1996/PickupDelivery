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


def main(env):
    merchants = env.merchants
    outputs = {}

    for i, drone in enumerate(env.drones):
        start = drone.position
        goal = list(merchants)[i]
        print('Start: ', start)
        print('Goal:', goal)

        planner = SIPPPlanner(drone.name, start, goal, env)

        if planner.compute_plan():
            plan = planner.get_plan()
            print(plan)
            outputs.update(plan)
            map["dynamic_obstacles"].update(plan)

    return
    print('Visual the paths:')
    last_pos, i = {}, 0
    while True:
        env.step()
        env.drones[0].position = pos
        if last_pos is not None:
            env.drones[0].last_pos = last_pos
        env.render(show=True, visited=visited)
        last_pos = {}

        i += 1
        if i >= 100:
            break


if __name__ == '__main__':
    env_ = Platform(
        size=args_.size,
        time_flow=args_.time_flow,
        drones=args_.num_agents,
        merchants=args_.num_merchants,
        buyers=args_.num_buyers,
        walls=args_.ratio_walls
    )

    main(env_)
    # test_env(args_, env_)

