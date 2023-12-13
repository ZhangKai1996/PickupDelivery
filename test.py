import time
from math import pow, sqrt
import numpy as np

import algo.tf_util as U
from env import MultiAgentEnv

from train import get_trainers


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    # parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./trained/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./trained/policy/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--display", action="store_true", default=True)
    return parser.parse_args()


def distance(p0, p1):
    return sqrt(pow(p0[0]-p1[0], 2)+pow(p0[1]-p1[1], 2))


def test():
    args = parse_args()

    with U.single_threaded_session():
        # Create environment
        env = MultiAgentEnv()
        # Create agent trainers
        trainers = get_trainers(env, args)
        print('Using good policy {} and adv policy {}'.format(args.good_policy, args.adv_policy))
        # Initialize
        U.initialize()
        # Load previous results, if necessary
        if args.display and args.load_dir is not None:
            print('Loading previous state...')
            U.load_state(args.load_dir)

        episode_rewards = []  # sum of rewards for all agents
        path = []
        success = []
        obs_n = env.reset()
        info = env.get_info()

        episode_step = 0
        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done, info_n = env.step(action_n)
            episode_step += 1
            terminal = done or (episode_step >= args.max_episode_len)
            path.append([distance(s1, s0) for s1, s0 in zip(new_obs_n, obs_n)])
            obs_n = new_obs_n

            # for displaying learned policies
            if args.display:
                pass
                time.sleep(0.1)
                env.render()

            # save model, display training output
            if terminal:
                episode_step = 0
                mean_path = list(np.sum(path, axis=0))
                ret = []
                print(len(episode_rewards), end=',')
                for i, (a, dists) in enumerate(info.items()):
                    print(a, dists, round(mean_path[i], 2), end=',')
                    ret.append(mean_path[i] / min(dists))
                print()
                episode_rewards.append(ret)
                success.append(int(done))
                path = []
                if len(episode_rewards) >= 500:
                    break
                obs_n = env.reset()
                info = env.get_info()

        print(np.mean(episode_rewards, axis=0), np.mean(success))


if __name__ == '__main__':
    test()
