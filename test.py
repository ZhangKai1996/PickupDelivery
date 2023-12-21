import numpy as np

from env.environment import CityEnv
from algo.framework import HieTrainer
from train import parse_args, make_exp_id

def test(env, trainer, max_episode_len):
    rew_stats = []
    ctrl_step = 0
    for episode in range(1, 100+1):
        obs_n, done = env.reset(), False
        obs_n_meta = env.observation_meta()
        scheme = trainer.select_scheme(obs_n_meta, episode)
        env.task_assignment(scheme)

        episode_step = 0
        rew_sum = [0.0, 0.0]
        while True:
            ctrl_step += 1
            episode_step += 1
            act_n = trainer.select_action(obs_n, ctrl_step)
            # Step the env and return outputs
            next_obs_n, (rew_n, rew_beta), done_n, _ = env.step(act_n)
            done = all(done_n)
            terminal = done or episode_step >= max_episode_len
            env.render(
                mode='Episode:{}, Step:{}'.format(episode, episode_step),
                clear=terminal,
                show=True
            )
            rew_sum[0] += min(rew_n)
            rew_sum[-1] += rew_beta
            obs_n = next_obs_n
            if terminal:
                break
        rew_stats.append(rew_sum)
        if episode % 100 == 0:
            mean_rew = np.mean(rew_stats, axis=1)
            print('Episode:{:>4d}, Step:{:>6d},Rew(ctrl):{:>+6.2f},Rew(meta):{:>+6.2f}'.format(
                episode, ctrl_step, mean_rew[0], mean_rew[1]))
            rew_stats = []


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
        tau=args.tau,
        a_lr=args.a_lr,
        c_lr=args.c_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_start=args.learning_start,
        memory_length = args.memory_length,
    )
    trainer.load_model()
    # Train with interaction.
    test(
        env=env,
        trainer=trainer,
        max_episode_len=args.max_episode_len
    )
    env.close()


if __name__ == '__main__':
    main()
