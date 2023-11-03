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
    obs = env.walls
    size = env.size
    print(start, goal)

    # A*
    start_time = time.time()
    astar = AStar(start, goal, "euclidean", obs, size)
    path, visited1 = astar.searching()
    path1 = path[::-1]
    print('Time Consumption ( A):', time.time()-start_time)

    # Dijkstra
    start_time = time.time()
    dijkstra = Dijkstra(start, goal, 'None', obs, size)
    path, visited2 = dijkstra.searching()
    path2 = path[::-1]
    print('Time Consumption ( D):', time.time()-start_time)

    # Best first
    start_time = time.time()
    best_first = BestFirst(start, goal, 'euclidean', obs, size)
    path, visited3 = best_first.searching()
    path3 = path[::-1]
    print('Time Consumption ( B):', time.time()-start_time)

    start_time = time.time()
    b_astar = BidirectionalAStar(start, goal, "euclidean", obs)
    path4, visited_fore, visited_back = b_astar.searching()
    print('Time Consumption (BA):', time.time()-start_time)

    print(len(path1), path1)
    print(len(path2), path2)
    print(len(path3), path3)
    print(len(path4), path4)

    path = path4
    visited = [visited_fore, visited_back]
    for i, pos in enumerate(path):
        env.step()
        env.drones[0].position = pos
        if i > 0:
            env.drones[0].last_pos = path[i-1]
        env.render(visited=visited, size=env.size)


if __name__ == '__main__':
    main()
