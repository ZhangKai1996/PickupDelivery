import time

from discrete_space.algo import *
from discrete_space.env import Platform

from parameters import args


def main():
    env = Platform(
        size=args.size,
        drones=args.num_agents,
        merchants=args.num_merchants,
        buyers=args.num_buyers,
        walls=args.ratio_walls,
        time_flow=args.time_flow
    )

    start = env.drones[0].position
    goal = env.merchants[0]
    size = env.size
    print(start, goal)

    # A*
    start_time = time.time()
    astar = AStar(start, goal, "euclidean", env.walls, size)
    path, visited1 = astar.searching()
    path1 = path[::-1]
    print('Time Consumption ( A):', time.time()-start_time)

    # D*
    start_time = time.time()
    d_star = DStar(start, goal, env)
    path2, visited2 = d_star.searching()
    print('Time Consumption ( D):', time.time()-start_time)

    print(len(path1), path1)
    print(len(path2), path2)

    path = path2
    visited = [visited2, ]
    last_pos, i = None, 0
    while True:
        env.step()
        pos = path.pop(0)
        if i > 0 and i % 5 == 0 and pos != goal:
            print(pos, last_pos, len(env.walls))
            env.add_obstacles([pos, ])
            env.cv_render.draw_dynamic([pos, ])
            path.insert(0, pos)
            path.insert(0, last_pos)
            print(len(path), path)

            start_time = time.time()
            astar = AStar(last_pos, goal, "euclidean", env.walls, size)
            path_new1, _ = astar.searching()
            print('\t>>> Time Consumption ( A):', time.time()-start_time)
            path = path_new1[::-1]
            print(len(path), path)

            print(pos in d_star.obs)
            print(len(d_star.path), d_star.path)
            start_time = time.time()
            path_new2, _ = d_star.on_press(pos, start=last_pos)
            print('\t>>> Time Consumption ( D):', time.time()-start_time)
            path = path_new2[:]
            print(len(path_new2), path_new2)

            pos = path.pop(0)
        env.drones[0].position = pos

        if last_pos is not None:
            env.drones[0].last_pos = last_pos
        env.render(visited=visited, size=env.size, show=True)
        last_pos = pos
        i += 1

        if len(path) <= 0:
            break


if __name__ == '__main__':
    main()
