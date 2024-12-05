import time

import numpy as np

from env.environment import CityEnv
from env.utils import one_hot
from algo.framework import HieTrainer


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--size", type=int, default=500, help="range of grid environment")
    parser.add_argument("--num-agents", type=int, default=1, help="number of the agent (drone or car)")
    parser.add_argument("--num-orders", type=int, default=1, help="number of tasks (the pair of <m,b>)")
    parser.add_argument("--num-stones", type=int, default=0, help="number of barriers")
    parser.add_argument("--num-episodes", type=int, default=int(1e6), help="number of episodes")
    parser.add_argument('--memory-length', type=int, default=int(1e6), help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=int(1e5), help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-3, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=1024, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--index", type=str, default='0', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=int(1e2), help="save model once every several episodes")
    parser.add_argument("--load-dir", type=str, default=None, help="directory in which model are loaded")
    return parser.parse_args()


def fixed_scheme(num_agents, num_tasks):
    scheme = []
    for _ in range(num_tasks):
        idx = np.random.randint(0, num_agents)
        scheme.append(one_hot(idx + 1, num=num_agents))
    scheme = np.stack(scheme)
    return scheme


def reset(env, num_agents, num_orders, render=False):
    env.reset(render=render)
    scheme = fixed_scheme(num_agents, num_orders)  # fixed scheme
    env.task_assignment(scheme)
    return env.observation(), False, False


def train(env, trainer, num_episodes, save_rate, max_len, num_agents, num_orders):
    start_time = time.time()

    step = 0
    rew_stats, sr_stats, occ_stats = [], [], []
    for episode in range(1, num_episodes + 1):
        obs_n, done, terminated = reset(env, num_agents, num_orders)

        episode_rew = []
        episode_step = 0
        while True:
            # Get action from trainer
            act_n = trainer.select_action(obs_n, t=step)
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, terminated = env.step(act_n)
            # Store the experience for controller
            trainer.add(obs_n, act_n, next_obs_n, rew_n, done_n)
            # Update controller
            trainer.update(step)

            step += 1
            episode_step += 1
            done = all(done_n)
            episode_rew.append(rew_n)
            obs_n = next_obs_n

            if done or terminated or episode_step >= max_len:
                break

        sr_stats.append(int(done and not terminated))
        occ_stats.append(env.dot())
        rew_stats.append(np.sum(episode_rew, axis=0))

        if episode % save_rate == 0:
            mean_sr = np.mean(sr_stats)
            mean_occ = np.mean(occ_stats, axis=0)
            mean_rew = np.sum(rew_stats, axis=0)
            decay_value = trainer.controller.decay(t=step)

            print('Episode:{:>7d}, Step:{:>7d}'.format(episode, step), end=', ')
            print('Decay:{:>5.3f}'.format(decay_value), end=', ')
            value_rew, value_occ = {}, {}
            for i, (r, o) in enumerate(zip(mean_rew, mean_occ)):
                value_rew['agent_'+str(i)] = r
                value_occ['agent_'+str(i)] = o
                print('Rew_{}:{:>+7.1f}'.format(i, r), end=', ')
                print('Occ_{}:{:>6.2f}'.format(i, o), end=', ')
            value_rew['total'] = sum(mean_rew)
            print('SR:{:>5.2f}'.format(mean_sr), end=', ')

            trainer.scalars(key='reward', value=value_rew, episode=episode)
            trainer.scalars(key='dot', value=value_occ, episode=episode)
            trainer.scalars(key='sr', value={'sr': mean_sr, 'decay': decay_value}, episode=episode)

            end_time = time.time()
            print('Time: {:>6.3f}'.format(end_time - start_time))
            start_time = end_time
            rew_stats, sr_stats, occ_stats = [], [], []
            # Save the model every fixed several episodes.
            trainer.save_model()


def make_exp_id(args):
    return 'exp_{}_{}({}_{}_{}_{}_{}_{}_{}_{}_{})'.format(
        args.exp_name, args.index,
        args.size,
        args.num_agents, args.num_orders, args.num_stones,
        args.seed,
        args.a_lr, args.c_lr,
        args.batch_size, args.gamma
    )


def main():
    # Parse hyper-parameters
    args = parse_args()
    np.random.seed(args.seed)
    # Create environment
    env = CityEnv(args=args)
    # Create a hierarchical trainer
    trainer = HieTrainer(
        env=env,
        num_orders=args.num_orders,
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
    # trainer.load_model()
    # Train with interaction.
    train(
        env=env,
        trainer=trainer,
        num_episodes=args.num_episodes,
        save_rate=args.save_rate,
        max_len=args.num_orders*args.size*3,
        num_agents=args.num_agents,
        num_orders=args.num_orders,
    )
    env.close()


if __name__ == '__main__':
    main()
