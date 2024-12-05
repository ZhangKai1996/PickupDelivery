import numpy as np

from env.environment import CityEnv
from algo.framework import HieTrainer

from train import parse_args, make_exp_id, reset


def train(env, trainer, num_episodes, num_agents, num_orders, max_len, render=True):
    step = 0
    rew_stats, sr_stats, occ_stats = [], [], []
    for episode in range(1, num_episodes + 1):
        obs_n, done, terminated = reset(env, num_agents, num_orders, render=render)

        episode_step = 0
        episode_rew = []
        while True:
            act_n = trainer.select_action(obs_n)
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, terminated = env.step(act_n, verbose=render)
            if render:
                # print('\t', episode_step, obs_n, np.argmax(act_n, axis=-1), next_obs_n, rew_n, done_n)
                env.render(mode='Episode:{}, Step:{}'.format(episode, episode_step), show=True)
            obs_n = next_obs_n

            step += 1
            episode_step += 1
            done = all(done_n)
            episode_rew.append(rew_n)

            if done or terminated or episode_step >= max_len:
                break

        sr_stats.append(int(done and not terminated))
        occ_stats.append(env.dot())
        mean_rew_epi = np.sum(episode_rew, axis=0)
        rew_stats.append(mean_rew_epi)
        print('Episode:{:>5d}, Step:{:>7d}, Done:{}'.format(episode, step, int(done)), end=', ')
        for i, (r, o) in enumerate(zip(mean_rew_epi, env.dot())):
            print('Rew_{}:{:>+7.1f}'.format(i, r), end=', ')
            print('Occ_{}:{:>6.2f}'.format(i, o), end=', ')
        print()

    mean_rew = np.mean(rew_stats, axis=0)
    mean_occ = np.mean(occ_stats, axis=0)
    for i, (r, o) in enumerate(zip(mean_rew, mean_occ)):
        print('Rew_{}:{:>+7.1f}'.format(i, r), end=', ')
        print('Occ_{}:{:>6.2f}'.format(i, o), end=', ')
    print('Rew: {:>7.1f}, SR:{:>5.2f}'.format(sum(mean_rew), np.mean(sr_stats)))


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
        num_episodes=int(1e4),
        num_agents=args.num_agents,
        num_orders=args.num_orders,
        max_len=args.num_orders*args.size*4,
        render=True
    )
    env.close()


if __name__ == '__main__':
    main()
