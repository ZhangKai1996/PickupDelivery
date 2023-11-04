
def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=15, help="number of the drone")
    parser.add_argument("--size", type=int, default=30, help="the size of grid world")
    parser.add_argument("--ratio-walls", type=float, default=.1, help="ratio of barriers")
    parser.add_argument("--num-buyers", type=int, default=30, help="number of buyers")
    parser.add_argument("--num-merchants", type=int, default=20, help="number of merchants")
    parser.add_argument("--time-flow", type=bool, default=False, help="number of buyers")
    # Basic training parameters
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument('--memory-length', default=int(1e6), type=int, help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=50, help="start updating after this number of step")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=10, help="save model once many episodes")
    parser.add_argument("--load-dir", type=str, default=None, help="directory in which model are loaded")

    return parser.parse_args()


args = parse_args()
