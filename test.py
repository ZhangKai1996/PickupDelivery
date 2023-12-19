import torch as th
import numpy as np

from env.environment import CityEnv
from train import parse_args

print(th.cuda.is_available())


def main():
    args = parse_args()
    env = CityEnv(args=args)
    dim_act_n = (args.num_agents, env.action_space_n[0].n)

    state, done = env.reset(), False
    print(np.array(state).shape)
    while not done:
        while True:
            action = np.random.uniform(low=-1.0, high=1.0, size=dim_act_n)
            n_state, rew, done, *_ = env.step(action)
            env.render()
            state = n_state
            if done:
                break
    env.close()

if __name__ == '__main__':
    main()
