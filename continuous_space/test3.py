import time
import copy

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
    start = env.drones[0].position
    goal = list(env.merchants)[0]
    print('Start: ', start)
    print('Goal:', goal)

    # RRT
    start_time = time.time()
    rrt = RRT(start, goal, env)
    path1 = rrt.planning()[::-1]
    visited1 = [rrt.vertex, ]
    print('Time Consumption (RRT):', time.time() - start_time, len(path1))

    # RRT connect
    start_time = time.time()
    rrt_conn = RRTConnect(start, goal, env)
    path2 = rrt_conn.planning()
    visited2 = [rrt_conn.V1, rrt_conn.V2]
    print('Time Consumption (RRT_C):', time.time() - start_time, len(path2))

    # Dynamic RRT (re-planning后的轨迹不完美)
    start_time = time.time()
    d_rrt = DynamicRRT(start, goal, env)
    path3 = d_rrt.planning()[::-1]
    visited3 = [d_rrt.vertex, ]
    print('Time Consumption (D_RRT):', time.time() - start_time, len(path3))

    walls = copy.deepcopy(env.walls)
    new_walls = []
    for i, pos in enumerate(path3):
        pos = (pos[0] + 0.6, pos[1] + 0.6)
        if i > 0 and i % 5 == 0:
            env.walls.add(pos)
            new_walls.append(pos)
    start_time = time.time()
    d_rrt = DynamicRRT(start, goal, env)
    path3_ = d_rrt.planning()[::-1]
    visited3_ = [d_rrt.vertex, ]
    print('Time Consumption (D_RRT):', time.time() - start_time, len(path3))

    env.walls = copy.deepcopy(walls)
    path = path3_[:]
    visited = visited3_
    last_pos, i = None, 0
    while True:
        env.step()
        pos = path.pop(0)
        if i > 0 and i % 5 == 0:
            if len(new_walls) > 0:
                env.cv_render.draw_pos([new_walls.pop(0), ])

        env.drones[0].position = pos
        if last_pos is not None:
            env.drones[0].last_pos = last_pos
        env.render(show=True, visited=visited)
        if i == 0:
            env.cv_render.draw_line(path3)
        last_pos = pos

        i += 1
        if len(path) <= 0:
            break


if __name__ == '__main__':
    env_ = Platform(
        size=args_.size,
        time_flow=args_.time_flow,
        drones=args_.num_agents,
        merchants=args_.num_merchants,
        buyers=args_.num_buyers,
        walls=args_.num_walls
    )

    main(env_)
    # test_env(args_, env_)
