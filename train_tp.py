import time

from env.environment import CityEnv
from algo.framework import HieTrainer

from parameters import *


def train(env, trainer, num_episodes, save_rate, max_len, num_agents, num_orders):
    start_time = time.time()

    step = 0
    ext_sr_stats = []
    rew_stats, sr_stats, occ_stats = [], [], []
    for episode in range(1, num_episodes + 1):
        initialize(env, num_agents, num_orders)
        env.path_planning()

        done_pf = False
        obs_n, done, terminated = env.observation(label='tp'), False, False

        episode_rew = []
        episode_step = 0
        while True:
            # Get action from trainer
            act_n = trainer.select_action(obs_n, t=step, label='tp')
            # Step the env and return outputs
            next_obs_n, rew_n, done_n, terminated, done_pf = env.step(act_n)
            # Store the experience for controller
            trainer.add(obs_n, act_n, next_obs_n, rew_n, done_n, label='tp')
            # Update controller
            trainer.update(step, label='tp')

            step += 1
            episode_step += 1
            done = all(done_n)
            episode_rew.append(rew_n)
            obs_n = next_obs_n

            if done or terminated or episode_step >= max_len:
                break

        sr_stats.append(int(done and not terminated))
        ext_sr_stats.append(int(done_pf))
        occ_stats.append(env.dot())
        rew_stats.append(np.sum(episode_rew, axis=0))

        if episode % save_rate == 0:
            mean_sr = np.mean(sr_stats)
            mean_ext_sr = np.mean(ext_sr_stats)
            mean_occ = np.mean(occ_stats, axis=0)
            mean_rew = np.sum(rew_stats, axis=0)

            print('Episode:{:>7d}, Step:{:>7d}'.format(episode, step), end=', ')
            value_rew, value_occ = {}, {}
            for i, (r, o) in enumerate(zip(mean_rew, mean_occ)):
                value_rew['agent_'+str(i)] = r
                value_occ['agent_'+str(i)] = o
                print('Rew_{}:{:>+7.1f}'.format(i, r), end=', ')
                print('Occ_{}:{:>6.2f}'.format(i, o), end=', ')
            value_rew['total'] = sum(mean_rew)
            print('Ext_SR:{:>5.2f}, Int_SR:{:>5.2f}'.format(mean_ext_sr, mean_sr), end=', ')

            trainer.scalars(key='reward', value=value_rew, episode=episode)
            trainer.scalars(key='dot', value=value_occ, episode=episode)
            trainer.scalars(key='sr', value={'int': mean_sr,
                                             'ext': mean_ext_sr}, episode=episode)

            end_time = time.time()
            print('Time: {:>6.3f}'.format(end_time - start_time))
            start_time = end_time

            ext_sr_stats = []
            rew_stats, sr_stats, occ_stats = [], [], []
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
        learning_start=args.learning_start,
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
