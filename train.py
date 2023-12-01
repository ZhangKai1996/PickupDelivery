import numpy as np
import time

import algo.tf_util as U
from algo.maddpg import MADDPGAgentTrainer
from env.environment import MultiAgentEnv

# import tf_slim.layers as layers
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
import tensorflow.contrib.layers as layers


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
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
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./trained/curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(inputs, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = inputs
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def get_trainers(env, args):
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

    trainers = []
    for i in range(env.n):
        trainers.append(
            MADDPGAgentTrainer(
                "agent_%d" % i, mlp_model,
                obs_shape_n, env.action_space,
                i, args,
                local_q_func=args.good_policy == 'ddpg'
            )
        )
    return trainers


def train():
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

        episode_rewards = [0.0]  # sum of rewards for all agents
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done, info_n = env.step(action_n)
            episode_step += 1
            terminal = (episode_step >= args.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if args.display:
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % args.save_rate == 0):
                U.save_state(args.save_dir, saver=saver)
                # print statement depends on whether there are adversaries or not
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-args.save_rate:]),
                    round(time.time() - t_start, 3)))
                t_start = time.time()

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > args.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    train()
