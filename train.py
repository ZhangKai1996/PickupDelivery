import time

import numpy as np

from env.environment import CityEnv
from algo.framework import HieTrainer


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=5, help="number of the agent (drone or car)")
    parser.add_argument("--num-tasks", type=int, default=10, help="number of tasks (the pair of <m,b>)")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000000, help="number of episodes")
    parser.add_argument('--memory-length', default=int(1e6), type=int, help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=5000, help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-4, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-4, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every several episodes")
    parser.add_argument("--load-dir", type=str, default=None, help="directory in which model are loaded")

    return parser.parse_args()


def train(env, trainer, num_episodes, max_episode_len, save_rate):
    rew_stats, sr_stats = [], []
    step = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs_n, done = env.reset(), False
        obs_n_meta = env.observation_meta()
        scheme = trainer.select_scheme(obs_n_meta, episode)
        env.task_assignment(scheme)

        episode_step = 0
        rew_sum = 0.0
        while True:
            step += 1
            episode_step += 1
            act_n = trainer.select_action(obs_n, step)
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, _ = env.step(act_n)
            done = all(done_n)
            terminal = done or episode_step >= max_episode_len
            # env.render(
            #     mode='Episode:{}, Step:{}'.format(episode, episode_step),
            #     clear=terminal,
            #     show=True
            # )
            # Store the experience for controller
            trainer.add(obs_n, act_n, next_obs_n, rew_n, done_n, label='ctrl')
            # Update controller
            trainer.update_controller(step)

            rew_sum += min(rew_n)
            obs_n = next_obs_n
            if terminal:
                break
        # Store experience for meta-controller
        next_obs_n_beta = env.observation_meta()
        trainer.add(obs_n_meta, scheme, next_obs_n_beta, rew_sum, 1.0, label='meta')
        # print(ctrl_step, rew_sum)
        rew_stats.append(rew_sum)
        sr_stats.append(int(done))
        # Update meta-controller
        trainer.update_meta_controller(episode)

        if episode % save_rate == 0:
            end_time = time.time()
            print('Episode:{:>6d}, Step:{:>7d}, Rew:{:>+7.2f}, SR:{:>3.2f}, Time:{:>6.3f}'.format(
                episode, step, np.mean(rew_stats), np.mean(sr_stats), end_time - start_time))
            trainer.scalar(key='reward', value=np.mean(rew_stats), episode=episode)
            trainer.scalar(key='sr', value=np.mean(sr_stats), episode=episode)
            rew_stats, sr_stats = [], []
            start_time = end_time
            # Save the model every fixed several episodes.
            trainer.save_model()


def make_exp_id(args):
    return 'exp_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.exp_name, args.num_agents, args.num_tasks, args.seed,
        args.a_lr, args.c_lr, args.batch_size, args.gamma
    )


def main():
    # Parse hyper-parameters
    args = parse_args()
    # Create environment
    env = CityEnv(args=args)
    # Create a hierarchical trainer
    trainer = HieTrainer(
        env=env,
        num_tasks=args.num_tasks,
        num_agents=args.num_agents,
        folder=make_exp_id(args),
        tau=args.tau,
        a_lr=args.a_lr,
        c_lr=args.c_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_start=args.learning_start,
        memory_length=args.memory_length,
    )
    # Train with interaction.
    train(
        env=env,
        trainer=trainer,
        num_episodes=args.num_episodes,
        max_episode_len=args.max_episode_len,
        save_rate=args.save_rate
    )
    env.close()


if __name__ == '__main__':
    main()
