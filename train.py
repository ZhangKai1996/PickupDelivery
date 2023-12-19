from collections import defaultdict

import numpy as np
import torch.autograd as autograd

from env.mdp import StochasticEnv
from algo.hdqn_mdp import HieDQN


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)


def train(env, trainer, num_episodes, gamma=1.0):
    """The h-DQN learning algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env:
        gym environment to train on.
    trainer:
        a h-DQN agent consists of a meta-controller and controller.
    num_episodes:
        Number (can be divided by 1000) of episodes to run for. Ex: 12000
    gamma: float
        Discount Factor
    """
    total_timestep = 0
    meta_timestep = 0
    ctrl_timestep = defaultdict(int)

    for i_thousand_episode in range(int(np.floor(num_episodes / 1000))):
        for i_episode in range(1000):
            episode_length = 0
            state = env.reset(onehot=True)

            done = False
            while not done:
                meta_timestep += 1
                goal = trainer.select_goal(state, total_timestep)[0]

                total_extrinsic_rew = 0
                while True:
                    total_timestep += 1
                    episode_length += 1
                    ctrl_timestep[goal] += 1
                    # Get annealing exploration rate (epsilon) from exploration_schedule
                    joint_state_goal = np.concatenate([state, goal], axis=1)
                    action = trainer.select_action(joint_state_goal, total_timestep)[0]
                    # Step the env and store the transition
                    n_state, extrinsic_rew, done, _ = env.step(action)
                    goal_reached = n_state == goal

                    intrinsic_rew = trainer.get_intrinsic_reward(goal, n_state)

                    joint_n_state_goal = np.concatenate([n_state, goal], axis=1)
                    trainer.ctrl_replay_memory.push(joint_state_goal, action, joint_n_state_goal, intrinsic_rew, done)
                    # Update Both meta-controller and controller
                    trainer.update_meta_controller(gamma)
                    trainer.update_controller(gamma)

                    total_extrinsic_rew += extrinsic_rew
                    state = n_state

                    if done or goal_reached:
                        # Goal Finished
                        trainer.meta_replay_memory.push(state, goal, n_state, total_extrinsic_rew, done)
                        break


# def train(env, trainer, num_episodes, schedule, gamma=1.0):
#     """The h-DQN learning algorithm.
#     All schedules are w.r.t. total number of steps taken in the environment.
#     Parameters
#     ----------
#     env:
#         gym environment to train on.
#     trainer:
#         a h-DQN agent consists of a meta-controller and controller.
#     num_episodes:
#         Number (can be divided by 1000) of episodes to run for. Ex: 12000
#     schedule:
#         schedule for probability of choosing random action.
#     gamma: float
#         Discount Factor
#     """
#     total_timestep = 0
#     meta_timestep = 0
#     ctrl_timestep = defaultdict(int)
#
#     for i_thousand_episode in range(int(np.floor(num_episodes / 1000))):
#         for i_episode in range(1000):
#             episode_length = 0
#             current_state = env.reset()
#             encoded_current_state = one_hot(current_state)
#
#             done = False
#             while not done:
#                 meta_timestep += 1
#                 # Get annealing exploration rate (epsilon) from exploration_schedule
#                 meta_epsilon = schedule.value(total_timestep)
#
#                 goal = trainer.select_goal(encoded_current_state, meta_epsilon)[0]
#                 encoded_goal = one_hot(goal)
#
#                 total_extrinsic_rew = 0
#                 goal_reached = False
#                 while not done and not goal_reached:
#                     total_timestep += 1
#                     episode_length += 1
#                     ctrl_timestep[goal] += 1
#                     # Get annealing exploration rate (epsilon) from exploration_schedule
#                     ctrl_epsilon = schedule.value(total_timestep)
#                     joint_state_goal = np.concatenate([encoded_current_state, encoded_goal], axis=1)
#                     action = trainer.select_action(joint_state_goal, ctrl_epsilon)[0]
#                     # Step the env and store the transition
#                     next_state, extrinsic_rew, done, _ = env.step(action)
#
#                     encoded_n_state = one_hot(next_state)
#                     intrinsic_rew = trainer.get_intrinsic_reward(goal, next_state)
#                     goal_reached = next_state == goal
#
#                     joint_n_state_goal = np.concatenate([encoded_n_state, encoded_goal], axis=1)
#                     trainer.ctrl_replay_memory.push(joint_state_goal, action, joint_n_state_goal, intrinsic_rew, done)
#                     # Update Both meta-controller and controller
#                     trainer.update_meta_controller(gamma)
#                     trainer.update_controller(gamma)
#
#                     total_extrinsic_rew += extrinsic_rew
#                     current_state = next_state
#                     encoded_current_state = encoded_n_state
#                 # Goal Finished
#                 trainer.meta_replay_memory.push(encoded_current_state, goal, encoded_n_state, total_extrinsic_rew, done)


def main(args):
    lr = 0.00025

    trainer = HieDQN(
        replay_memory_size=int(1e6),
        batch_size=args.batch_size,
        lr=lr
    )
    train(
        env=StochasticEnv(),
        trainer=trainer,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--num-riders", type=int, default=5, help="number of the agent (drone or car)")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument('--memory-length', default=int(1e6), type=int, help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=50, help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-4, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model are loaded")

    args_ = parser.parse_args()
    main(args=args_)
