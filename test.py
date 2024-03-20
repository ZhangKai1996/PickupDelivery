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

    # scheme, num_agents = [], len(obs_n)
    # for i in range(num_tasks):
    #     if i in [2, 3]:
    #         scheme.append(one_hot(1, num=num_agents))
    #     else:
    #         scheme.append(one_hot(2, num=num_agents))
    # scheme = np.stack(scheme)
    return scheme


def test(env, trainer, max_episode_len, num_tasks):
    step_stats, rew_stats, sr_stats = [], [], []
    ctrl_step = 0
    for episode in tqdm(range(1, 1000 + 1), desc='Testing'):
        obs_n, done = env.reset(), False
        obs_n_meta = env.observation_meta()
        # scheme = trainer.select_scheme(obs_n_meta, episode, test=True)
        scheme = trainer.select_scheme(obs_n_meta, episode)
        # scheme = fixed_scheme(obs_n, num_tasks)   # fixed scheme
        env.task_assignment(scheme)

        episode_step = 0
        rew_sum = 0.0
        while True:
            ctrl_step += 1
            episode_step += 1
            act_n = trainer.select_action(obs_n, ctrl_step, test=True)
            # print(act_n)
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
            rew_sum += sum(rew_n)
            obs_n = next_obs_n
            if terminal:
                break

        rew_stats.append(rew_sum)
        sr_stats.append(int(done))
        step_stats.append(episode_step)
    print('Step:{:>6.2f},Rew:{:>+6.2f},SR:{:>4.2f}'.format(
        np.mean(step_stats), np.mean(rew_stats), np.mean(sr_stats))
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
        max_episode_len=args.max_episode_len,
        num_tasks=args.num_tasks
    )
    env.close()


if __name__ == '__main__':
    main()
