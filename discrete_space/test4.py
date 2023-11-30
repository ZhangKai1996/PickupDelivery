import numpy as np

from discrete_space.algo import *
from discrete_space.env import Platform

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


def main_cbs(env):
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print(" Solution not found")
        return

    outputs = {
        "schedule": solution,
        "cost": cbs.compute_solution_cost(solution)
    }

    drone_dict = {drone.name: drone for drone in env.drones}
    tracks = {}
    for key, value in outputs['schedule'].items():
        for pos_dict in value:
            t = pos_dict['t']
            if t in tracks.keys():
                tracks[t][key] = (pos_dict['x'], pos_dict['y'])
            else:
                tracks[t] = {key: (pos_dict['x'], pos_dict['y'])}
    # print(tracks)

    print('Visual the paths:')
    for t in range(max(tracks.keys())+1):
        env.step()
        if t not in tracks.keys():
            continue

        for key, value in tracks[t].items():
            drone = drone_dict[key]
            drone.last_pos = drone.position
            drone.position = value
            drone.distance += 1
        env.render(show=False)

    dist = [drone.distance for drone in env.drones]
    print(dist, sum(dist))


def main_sipp(env):
    merchants = env.merchants
    outputs = {}

    drone_dict = {}
    for i, drone in enumerate(env.drones):
        start = drone.position
        goal = list(merchants)[i]
        drone_dict[drone.name] = drone
        print('Start: ', start)
        print('Goal:', goal)

        planner = SIPPPlanner(drone.name, start, goal, env)

        if planner.compute_plan():
            plan = planner.get_plan()
            print(plan)
            outputs.update(plan)
            env.dynamic_obs.update(plan)

    tracks = {}
    for key, value in outputs.items():
        for pos_dict in value:
            t = pos_dict['t']
            if t in tracks.keys():
                tracks[t][key] = (pos_dict['x'], pos_dict['y'])
            else:
                tracks[t] = {key: (pos_dict['x'], pos_dict['y'])}
    print(tracks)

    print('Visual the paths:')
    for t in range(max(tracks.keys())+1):
        env.step()
        if t not in tracks.keys():
            continue

        for key, value in tracks[t].items():
            drone = drone_dict[key]
            drone.last_pos = drone.position
            drone.position = value
        env.render(show=True)


if __name__ == '__main__':
    env_ = Platform(
        size=args_.size,
        time_flow=args_.time_flow,
        drones=args_.num_agents,
        merchants=args_.num_merchants,
        buyers=args_.num_buyers,
        walls=args_.num_walls
    )

    # main_sipp(env_)
    main_cbs(env_)
    # test_env(args_, env_)

