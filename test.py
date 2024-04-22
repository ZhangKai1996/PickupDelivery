import time

import numpy as np

from env.environment import CityEnv
from algo.framework import HieTrainer

from train import parse_args, make_exp_id, fixed_scheme


def train(env, trainer, num_episodes, max_episode_len, num_tasks):
    rew_stats, sr_stats = [], []
    step = 0

    for episode in range(1, num_episodes + 1):
        obs_n, done = env.reset(), False
        obs_n_meta = env.observation_meta()
        scheme = trainer.select_scheme(obs_n_meta, t=episode)
        # scheme = fixed_scheme(obs_n, num_tasks)  # fixed scheme
        env.task_assignment(scheme)

        episode_step = 0
        rew_sum = []
        while True:
            step += 1
            episode_step += 1
            act_n = trainer.select_action(obs_n, t=step)
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, _ = env.step(act_n)
            done = all(done_n)
            terminal = done or episode_step >= max_episode_len
            env.render(
                mode='Episode:{}, Step:{}'.format(episode, episode_step),
                clear=terminal,
                show=True
            )
            time.sleep(0.1)
            rew_sum.append(rew_n)
            obs_n = next_obs_n
            if terminal:
                break
        # print(ctrl_step, rew_sum)
        mean_rew_epi = np.sum(rew_sum, axis=0)
        rew_stats.append(mean_rew_epi)
        sr_stats.append(int(done))
        print('Episode:{:>5d}, Step:{:>7d}, Done:{}'.format(
            episode, step, int(done)), end=', '
        )
        for i, r in enumerate(mean_rew_epi):
            print('Rew_{}:{:>+8.2f}'.format(i, r), end=', ')
        print()
    mean_rew = np.mean(rew_stats, axis=0)
    for i, r in enumerate(mean_rew):
        print('Rew_{}:{:>+7.2f}'.format(i, r), end=', ')
    print('Rew: {:>7.2f},SR:{:>3.2f}'.format(sum(mean_rew), np.mean(sr_stats)))


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
    trainer.load_model()
    # Train with interaction.
    train(
        env=env,
        trainer=trainer,
        num_episodes=1000,
        max_episode_len=args.max_episode_len,
        num_tasks=args.num_tasks
    )
    env.close()


if __name__ == '__main__':
    main()
