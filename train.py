import time

import numpy as np

from env.environment import CityEnv
from env.utils import one_hot
from algo.framework import HieTrainer


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--size", type=int, default=300, help="range of grid environment")
    parser.add_argument("--num-agents", type=int, default=1, help="number of the agent (drone or car)")
    parser.add_argument("--num-orders", type=int, default=30, help="number of tasks (the pair of <m,b>)")
    parser.add_argument("--num-stones", type=int, default=0, help="number of barriers")
    parser.add_argument("--num-steps", type=int, default=int(1e9), help="number of episodes")
    parser.add_argument('--memory-length', type=int, default=int(1e6), help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=int(1e6), help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-3, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--index", type=str, default='0', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every several episodes")
    parser.add_argument("--load-dir", type=str, default=None, help="directory in which model are loaded")
    return parser.parse_args()


def reset(env, num_agents, num_orders, render=False):
    env.reset(render=render)
    # obs_n_meta = env.observation_meta()
    # scheme = trainer.select_scheme(obs_n_meta, t=episode)
    scheme = fixed_scheme(num_agents, num_orders)  # fixed scheme
    # scheme = np.array([[1.0] for i in range(num_orders)])
    env.task_assignment(scheme)
    return env.observation(), False, False



def fixed_scheme(num_agents, num_tasks):
    scheme = []
    for _ in range(num_tasks):
        idx = np.random.randint(0, num_agents)
        scheme.append(one_hot(idx + 1, num=num_agents))
    scheme = np.stack(scheme)
    return scheme


def train(env, trainer, num_steps, save_rate, num_agents, num_orders):
    rew_stats = []
    start_time = time.time()

    obs_n, done, terminated = reset(env, num_agents, num_orders)

    for step in range(1, num_steps + 1):
        # Get action from trainer
        act_n = trainer.select_action(obs_n, t=step)
        # Step the env and return outputs
        next_obs_n, rew_n, done_n, terminated = env.step(act_n)
        # Store the experience for controller
        trainer.add(obs_n, act_n, next_obs_n, rew_n, done_n, label='ctrl')
        # Update controller
        trainer.update_controller(step)

        done = all(done_n)
        rew_stats.append(rew_n)

        if done or terminated:
            obs_n, done, terminated = reset(env, num_agents, num_orders)
            # print(ctrl_step, rew_sum)
            # trainer.add(obs_n_meta, scheme, obs_n_meta, np.sum(rew_sum), True, label='meta')
            # trainer.update_meta_controller(step)
        else:
            obs_n = next_obs_n

        if step % save_rate == 0:
            mean_rew = np.sum(rew_stats, axis=0)
            print('Step:{:>7d}'.format(step), end=', ')
            value = {}
            for i, r in enumerate(mean_rew):
                value['agent_'+str(i)] = r
                print('Rew_{}:{:>+7.2f}'.format(i, r), end=', ')
            value['total'] = sum(mean_rew)
            trainer.scalars(key='reward', value=value, step=step)

            end_time = time.time()
            print('Time: {:>6.3f}'.format(end_time - start_time))
            start_time = end_time
            rew_stats = []

            # Save the model every fixed several episodes.
            trainer.save_model()


def make_exp_id(args):
    return 'exp_{}_{}:{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.exp_name, args.index,
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
        num_steps=args.num_steps,
        save_rate=args.save_rate,
        num_agents=args.num_agents,
        num_orders=args.num_orders,
    )
    env.close()


if __name__ == '__main__':
    main()
