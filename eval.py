import time
from tqdm import tqdm
import numpy as np

from env.environment import CityEnv
from algo.framework import HieTrainer
from train import parse_args, make_exp_id


def test(env, trainer, num_steps, num_tasks):
    obs_n, done = env.reset(), False
    scheme = np.array([[1.0] for _ in range(num_tasks)])
    env.task_assignment(scheme, test=True)

    rew_sum = 0.0
    for step in tqdm(range(1, num_steps + 1), desc='Testing'):
        act_n = trainer.select_action(obs_n)
        next_obs_n, rew_n, done_n, _ = env.step(act_n, test=True)
        env.render(
            mode='Step:{}'.format(step),
            show=True
        )
        # time.sleep(0.1)
        rew_sum += sum(rew_n)
        obs_n = next_obs_n
        if any(done_n):
            break
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
    test(
        env=env,
        trainer=trainer,
        num_steps=int(1e5),
        num_tasks=args.num_tasks
    )
    env.close()


if __name__ == '__main__':
    main()
