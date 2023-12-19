import numpy as np
import torch.autograd as autograd

from env.environment import CityEnv
from algo.framework import HieTrainer


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)


def train(env, trainer, num_episodes, gamma=1.0):
    total_step = 0
    for episode in range(num_episodes):
        state = env.reset()
        print(np.array(state).shape)

        done = False
        while not done:
            goal = trainer.select_goal(state, total_step)[0]
            total_rew = 0
            while True:
                total_step += 1

                # Select the action according to joint state and goal
                joint_state_goal = np.concatenate([state, goal], axis=1)
                action = trainer.select_action(joint_state_goal, total_step)[0]

                # Step the env and return outputs
                n_state, rew, done, _ = env.step(action)

                # Get the intrinsic reward for controller
                intrinsic_rew = trainer.get_intrinsic_reward(goal, n_state)

                # Store the experience for controller
                joint_n_state_goal = np.concatenate([n_state, goal], axis=1)
                trainer.ctrl_replay_memory.push(joint_state_goal, action, joint_n_state_goal, intrinsic_rew, done)

                # Update Both meta-controller and controller
                trainer.update_meta_controller(gamma)
                trainer.update_controller(gamma)

                total_rew += rew
                state = n_state

                if done or n_state == goal:
                    break
            # Goal Finished and store experience for meta-controller
            trainer.meta_replay_memory.push(state, goal, n_state, total_rew, done)

def main(args):
    env = CityEnv(args=args)
    trainer = HieTrainer(args=args)
    train(
        env=env,
        trainer=trainer,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    parser.add_argument("--num-agents", type=int, default=5, help="number of the agent (drone or car)")
    parser.add_argument("--num-tasks", type=int, default=10, help="number of tasks (the pair of <m,b>)")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument('--memory-length', default=int(1e6), type=int, help='number of experience replay pool')
    parser.add_argument("--learning-start", type=int, default=50, help="start updating after this number of step")
    parser.add_argument("--good-policy", type=str, default="algo", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="algo", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--a-lr", type=float, default=1e-4, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--c-lr", type=float, default=1e-3, help="learning rate for Critic Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument('--tau', default=0.001, type=float, help='rate of soft update')
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--seed", type=int, default=1111, help="name of the experiment")
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model are loaded")

    args_ = parser.parse_args()
    main(args=args_)
