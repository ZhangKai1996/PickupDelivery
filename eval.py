import time
from tqdm import tqdm
import numpy as np

from env.environment import CityEnv
from env.utils import one_hot
from algo.framework import HieTrainer
from train import parse_args, make_exp_id


def fixed_scheme(obs_n, num_tasks):
    scheme_idx, num_agents = [], len(obs_n)
    for i in range(num_tasks):
        scheme_idx.append(i % num_agents)
    np.random.shuffle(scheme_idx)
    num = np.random.randint(0, 4)
    scheme = []
    for i in range(num_tasks):
        scheme.append(one_hot(scheme_idx[i]+1, num=num_agents))
    scheme = np.stack(scheme)
    return scheme


def test(env, trainer, num_steps):
    obs_n, done = env.reset(), False

    rew_sum = 0.0
    for step in tqdm(range(1, num_steps + 1), desc='Testing'):
        obs_n_meta = env.observation_meta()
        scheme = trainer.select_scheme(obs_n_meta, t=step)
        # scheme = fixed_scheme(obs_n, num_tasks)   # fixed scheme
        env.task_assignment(scheme)

        act_n = trainer.select_action(obs_n, t=step)
        # Step the env and return outputs
        next_obs_n, rew_n, _, _ = env.step(act_n)
        env.render(
            mode='Step:{}'.format(step),
            show=True
        )
        time.sleep(0.1)
        rew_sum += sum(rew_n)
        obs_n = next_obs_n
    print('Rew:{:>+6.2f}'.format(rew_sum))


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
    test(env=env, trainer=trainer, num_steps=int(1e5))
    env.close()


if __name__ == '__main__':
    main()
