import numpy as np

from env.utils import one_hot


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--size", type=int, default=20, help="range of grid environment")
    parser.add_argument("--num-agents", type=int, default=1, help="number of the agent (drone or car)")
    parser.add_argument("--num-orders", type=int, default=40, help="number of tasks (the pair of <m,b>)")
    parser.add_argument("--num-stones", type=int, default=0, help="number of barriers")
    parser.add_argument("--num-episodes", type=int, default=int(1e6), help="number of episodes")
    parser.add_argument('--memory-length', type=int, default=int(1e6), help='number of experience replay pool')
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-3, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
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
    for i in range(num_tasks):
        # idx = np.random.randint(0, num_agents)
        # scheme.append(one_hot(idx + 1, num=num_agents))
        scheme.append(one_hot(i%num_agents + 1, num=num_agents))
    scheme = np.stack(scheme)
    return scheme


def initialize(env, num_agents, num_orders, render=False):
    env.reset(render=render)
    scheme = fixed_scheme(num_agents, num_orders)  # fixed scheme
    env.task_assignment(scheme)


def make_exp_id(args):
    return 'exp_{}_{}({}_{}_{}_{}_{}_{}_{}_{}_{})'.format(
        args.exp_name, args.index,
        args.size,
        args.num_agents, args.num_orders, args.num_stones,
        args.seed,
        args.a_lr, args.c_lr,
        args.batch_size, args.gamma
    )
