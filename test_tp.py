from env.environment import CityEnv
from algo.framework import HieTrainer

from parameters import *


def train(env, trainer, num_episodes, num_agents, num_orders, max_len, render=True):
    step = 0

    int_rew_stats, ext_rew_stats = [], []
    int_sr_stats, ext_sr_stats, occ_stats = [], [], []
    for episode in range(1, num_episodes + 1):
        initialize(env, num_agents, num_orders, render=render)

        obs_n_pf = env.observation(label='pf')
        sequences = trainer.select_action(obs_n_pf, label='pf')
        env.path_planning(sequences)
        # env.path_planning()

        done_pf = False
        obs_n, done, terminated = env.observation(label='tp'), False, False

        episode_step = 0
        episode_intrinsic_rew = []
        while True:
            act_n = trainer.select_action(obs_n, label='tp')
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, terminated, done_pf = env.step(act_n, verbose=render)
            if render:
                # print('\t', episode_step, obs_n, np.argmax(act_n, axis=-1), next_obs_n, rew_n, done_n)
                env.render(mode='Episode:{}, Step:{}'.format(episode, episode_step), show=True)
            obs_n = next_obs_n

            step += 1
            episode_step += 1
            done = all(done_n)
            episode_intrinsic_rew.append(rew_n)

            if done or terminated or episode_step >= max_len:
                break

        count_occ = env.dot()
        episode_rew_sum = np.sum(episode_intrinsic_rew, axis=0)
        if done_pf:
            extrinsic_rew_n = [rew/150 for rew in episode_rew_sum]
        else:
            extrinsic_rew_n = [(count_occ[i] - len(env.agents[i].orders)) * 3.0
                               for i, _ in enumerate(episode_rew_sum)]
        act_n_sl = np.expand_dims(sequences, axis=-1)
        obs_n_sl = np.concatenate([obs_n_pf, act_n_sl], axis=2)
        ext_rew_n = trainer.select_action(obs_n_sl, label='sl')
        loss = np.mean(np.square(ext_rew_n-extrinsic_rew_n))

        occ_stats.append(count_occ)
        int_sr_stats.append(int(done and not terminated))
        ext_sr_stats.append(int(done_pf))
        int_rew_stats.append(ext_rew_n)
        ext_rew_stats.append(extrinsic_rew_n)

        print('Episode:{:>5d}, Step:{:>7d}, Done:{}, Done_pf:{}, Loss:{:>5.2f}'.format(
            episode, step, int(done), int(done_pf), loss), end=', ')

        for i, (int_r, ext_r) in enumerate(zip(ext_rew_n, extrinsic_rew_n)):
            print('Int_rew_{}:{:>+7.1f}'.format(i, int_r), end=', ')
            print('Ext_rew_{}:{:>+7.1f}'.format(i, ext_r), end=', ')
            print('Ext_occ_{}:{:>6.2f}'.format(i, occ_stats[-1][i]), end=', ')
        print()

    mean_int_sr = np.mean(int_sr_stats)
    mean_ext_sr = np.mean(ext_sr_stats)
    mean_int_rew = np.mean(int_rew_stats, axis=0)
    mean_ext_rew = np.mean(ext_rew_stats, axis=0)
    mean_occ = np.mean(occ_stats, axis=0)
    for i, (int_r, ext_r) in enumerate(zip(mean_int_rew, mean_ext_rew)):
        print('Int_rew_{}:{:>+7.1f}'.format(i, int_r), end=', ')
        print('Ext_rew_{}:{:>+7.1f}'.format(i, ext_r), end=', ')
        print('Ext_occ_{}:{:>6.2f}'.format(i, mean_occ[i]), end=', ')
    print('Ext_SR:{:>5.2f}, Int_SR:{:>5.2f}'.format(mean_ext_sr, mean_int_sr))


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
        max_len=args.num_orders*args.size*2,
        render=False
    )
    env.close()


if __name__ == '__main__':
    main()
