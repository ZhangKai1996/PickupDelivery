import time

from env.environment import CityEnv
from algo.framework import HieTrainer

from parameters import *


def train(env, trainer, num_episodes, save_rate, max_len, num_agents, num_orders):
    start_time = time.time()

    step = 0
    int_rew_stats, ext_rew_stats = [], []
    int_sr_stats, ext_sr_stats, occ_stats = [], [], []
    for episode in range(1, num_episodes + 1):
        initialize(env, num_agents, num_orders)

        obs_n_pf = env.observation(label='pf')
        sequences = trainer.select_action(obs_n_pf, t=episode, label='pf')
        env.path_planning(sequences)

        done_pf = False
        obs_n, done_tp, terminated = env.observation(label='tp'), False, False

        episode_intrinsic_rew = []
        episode_step = 0
        while True:
            # Get action from trainer
            act_n = trainer.select_action(obs_n, t=step, label='tp')
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, terminated, done_pf  = env.step(act_n)
            # Store the experience for controller
            trainer.add(obs_n, act_n, next_obs_n, rew_n, done_n, label='tp')
            # Update controller
            trainer.update(step, label='tp')

            step += 1
            episode_step += 1
            done_tp = all(done_n)
            episode_intrinsic_rew.append(rew_n)
            obs_n = next_obs_n

            if done_tp or terminated or episode_step >= max_len:
                break

        count_occ = env.dot()
        episode_rew_sum = np.sum(episode_intrinsic_rew, axis=0)
        # extrinsic_rew_n = [rew/100 if done_pf else -10.0 for rew in episode_rew_sum]
        # if done_pf:
        #     extrinsic_rew_n = [rew/150 for rew in episode_rew_sum]
        # else:
        #     extrinsic_rew_n = [(count_occ[i] - len(env.agents[i].orders)) * 3.0
        #                        for i, _ in enumerate(episode_rew_sum)]
        extrinsic_rew_n = [(count_occ[i] - len(env.agents[i].orders)) * 0.5
                           for i, _ in enumerate(episode_rew_sum)]
        trainer.add(obs_n_pf, sequences, obs_n_pf, extrinsic_rew_n, True, label='pf')
        trainer.update(episode, label='pf')

        int_sr_stats.append(int(done_tp and not terminated))
        ext_sr_stats.append(int(done_pf))
        int_rew_stats.append(episode_rew_sum)
        ext_rew_stats.append(extrinsic_rew_n)
        occ_stats.append(env.dot())

        if episode % save_rate == 0:
            mean_int_sr = np.mean(int_sr_stats)
            mean_ext_sr = np.mean(ext_sr_stats)
            mean_occ = np.mean(occ_stats, axis=0)
            mean_int_rew = np.mean(int_rew_stats, axis=0)
            mean_ext_rew = np.mean(ext_rew_stats, axis=0)

            print('Episode:{:>7d}, Step:{:>7d}'.format(episode, step), end=', ')
            value_int_rew, value_ext_rew, value_occ = {}, {}, {}
            for i, (int_r, ext_r) in enumerate(zip(mean_int_rew, mean_ext_rew)):
                value_int_rew['agent_'+str(i)] = int_r
                value_ext_rew['agent_'+str(i)] = ext_r
                value_occ['agent_'+str(i)] = mean_occ[i]
                print('Int_rew_{}:{:>+7.1f}'.format(i, int_r), end=', ')
                print('Ext_rew_{}:{:>+7.1f}'.format(i, ext_r), end=', ')
                print('Ext_occ_{}:{:>6.2f}'.format(i, mean_occ[i]), end=', ')
            value_int_rew['total'] = sum(mean_int_rew)
            value_ext_rew['total'] = sum(mean_ext_rew)
            print('Ext_SR:{:>5.2f}, Int_SR:{:>5.2f}'.format(mean_ext_sr, mean_int_sr), end=', ')

            trainer.scalars(key='Int reward', value=value_int_rew, episode=episode)
            trainer.scalars(key='Ext reward', value=value_ext_rew, episode=episode)
            trainer.scalars(key='dot', value=value_occ, episode=episode)
            trainer.scalars(key='sr', value={'int': mean_int_sr,
                                             'ext': mean_ext_sr}, episode=episode)
            end_time = time.time()
            print('Time: {:>6.3f}'.format(end_time - start_time))
            start_time = end_time

            int_rew_stats, ext_rew_stats = [], []
            int_sr_stats, ext_sr_stats, occ_stats = [], [], []
            # Save the model every fixed several episodes.
            trainer.save_model()


def main():
    # Parse hyper-parameters
    args = parse_args()
    np.random.seed(args.seed)
    # Create environment
    env = CityEnv(args=args)
    # Create a hierarchical trainer
    trainer = HieTrainer(
        env=env,
        num_agents=args.num_agents,
        num_orders=args.num_orders,
        folder=make_exp_id(args),
        tau=args.tau,
        a_lr=args.a_lr,
        c_lr=args.c_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_length=args.memory_length,
    )
    # trainer.load_model()
    # Train with interaction.
    train(
        env=env,
        trainer=trainer,
        num_episodes=args.num_episodes,
        save_rate=args.save_rate,
        max_len=args.num_orders*args.size*2,
        num_agents=args.num_agents,
        num_orders=args.num_orders,
    )
    env.close()


if __name__ == '__main__':
    main()
