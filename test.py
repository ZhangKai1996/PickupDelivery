import time

import numpy as np

from env.environment import CityEnv
from algo.framework import HieTrainer

from train import parse_args, make_exp_id, reset


def train(env, trainer, num_steps, num_agents, num_orders):
    obs_n, done, terminated = reset(env, num_agents, num_orders, render=True)

    step = 1
    rew_stats = []
    for step in range(1, num_steps + 1):
        act_n = trainer.select_action(obs_n)
        print(obs_n, act_n, [agent.state for agent in env.agents], end=' ')
        print([order.status() for order in env.orders], end=' ')
        # Step the env and return outputs
        next_obs_n, rew_n, done_n, terminated = env.step(act_n)
        print([agent.state for agent in env.agents], next_obs_n, rew_n, done_n)
        done = all(done_n)
        env.render(mode='Step:{}'.format(step), show=True)
        # time.sleep(0.1)
        rew_stats.append(rew_n)
        obs_n = next_obs_n
        if done or terminated:
            break

    # print(ctrl_step, rew_sum)
    mean_rew_epi = np.sum(rew_stats, axis=0)
    rew_stats.append(mean_rew_epi)
    print('Step:{:>7d}, Done:{}'.format(step, int(done)), end=', ')
    for i, r in enumerate(mean_rew_epi):
        print('Rew_{}:{:>+8.2f}'.format(i, r), end=', ')
    print()
    mean_rew = np.mean(rew_stats, axis=0)
    for i, r in enumerate(mean_rew):
        print('Rew_{}:{:>+7.2f}'.format(i, r), end=', ')
    print('Rew: {:>7.2f}'.format(sum(mean_rew)))


def main():
    # Parse hyper-parameters
    args = parse_args()
    # Create environment
    env = CityEnv(args=args)
    # Create a hierarchical trainer
    trainer = HieTrainer(
        env=env,
        num_orders=args.num_orders,
        num_agents=args.num_agents,
        folder=make_exp_id(args),
        test=True,
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
        num_steps=int(1e4),
        num_agents=args.num_agents,
        num_orders=args.num_orders
    )
    env.close()


if __name__ == '__main__':
    main()
