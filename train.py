import time

import numpy as np

from env.environment import CityEnv
from env.utils import one_hot
from algo.framework import HieTrainer


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=2, help="number of the agent (drone or car)")
    parser.add_argument("--num-tasks", type=int, default=6, help="number of tasks (the pair of <m,b>)")
    parser.add_argument("--max-episode-len", type=int, default=40, help="maximum episode length")
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
    parser.add_argument("--exp-name", type=str, default='v1.0', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every several episodes")
    parser.add_argument("--load-dir", type=str, default=None, help="directory in which model are loaded")

    return parser.parse_args()


def fixed_scheme(obs_n, num_tasks):
    scheme_idx, num_agents = [], len(obs_n)
    for i in range(num_tasks):
        scheme_idx.append(i % num_agents)
    np.random.shuffle(scheme_idx)
    scheme = []
    for i in range(num_tasks):
        scheme.append(one_hot(scheme_idx[i]+1, num=num_agents))
    scheme = np.stack(scheme)
    return scheme


def train(env, trainer, num_episodes, max_episode_len, save_rate, num_tasks):
    # trainer.load_model()

    rew_stats, sr_stats = [], []
    step = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs_n, done = env.reset(), False
        obs_n_meta = env.observation_meta()
        scheme = trainer.select_scheme(obs_n_meta, episode)
        # scheme = fixed_scheme(obs_n, num_tasks)  # fixed scheme
        env.task_assignment(scheme)

        episode_step = 0
        rew_sum = []
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

            rew_sum.append(rew_n)
            obs_n = next_obs_n
            if terminal:
                break
        # print(ctrl_step, rew_sum)
        rew_stats.append(np.sum(rew_sum, axis=0))
        sr_stats.append(int(done))

        if episode % save_rate == 0:
            end_time = time.time()
            mean_rew = np.mean(rew_stats, axis=0)
            print('Episode:{:>6d}, Step:{:>7d}'.format(episode, step), end=', ')
            value = {}
            for i, r in enumerate(mean_rew):
                value['agent_'+str(i)] = r
                print('Rew_{}:{:>+7.2f}'.format(i, r), end=', ')
            value['total'] = sum(mean_rew)
            print('SR:{:>3.2f}, Time:{:>6.3f}'.format(np.mean(sr_stats), end_time - start_time))
            trainer.scalars(key='reward', value=value, episode=episode)
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
        save_rate=args.save_rate,
        num_tasks=args.num_tasks
    )
    env.close()


if __name__ == '__main__':
    main()
