import time

from env.environment import CityEnv
from algo.framework import HieTrainer

from parameters import *


def train(env, trainer, num_episodes, save_rate, max_len, num_agents, num_orders):
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        initialize(env, num_agents, num_orders)

        obs_n_pf = env.observation(label='pf')
        sequences = trainer.select_action(obs_n_pf, t=episode, label='pf')
        env.path_planning(sequences)

        act_n_sl = np.expand_dims(sequences, axis=-1)
        obs_n_sl = np.concatenate([obs_n_pf, act_n_sl], axis=2)
        ext_rew_n = trainer.select_action(obs_n_sl, label='sl')
        trainer.add(obs_n_pf, sequences, obs_n_pf, ext_rew_n, True, label='pf')
        trainer.update(episode, label='pf')

        if episode % save_rate == 0:
            end_time = time.time()
            print('Episode: {:>5d}, Time: {:>6.3f}'.format(
                episode, end_time - start_time))
            start_time = end_time
            # Save the model every fixed several episodes.
            trainer.save_model(label='pf')


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
    trainer.load_model(label='sl')
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
