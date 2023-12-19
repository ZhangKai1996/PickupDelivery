import random

import torch as th
import torch.optim as optim
import torch.nn.functional as F

from algo.memory import ReplayMemory
from algo.misc import LinearSchedule
from algo.trainer import MetaController, Controller

d_type = th.cuda.FloatTensor


class HieTrainer:
    def __init__(self, args):
        self.args = args

        self.num_goal = args.num_tasks
        self.batch_size = args.batch_size

        # Construct meta-controller and controller
        self.meta_controller = MetaController().type(d_type)
        self.controller = Controller().type(d_type)
        # Construct target meta-controller and target controller
        self.target_meta_controller = MetaController().type(d_type)
        self.target_controller = Controller().type(d_type)
        # Construct the optimizers for meta-controller and controller
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=args.a_lr)
        self.ctrl_optimizer = optim.Adam(self.controller.parameters(), lr=args.a_lr)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayMemory(args.memory_length)
        self.ctrl_replay_memory = ReplayMemory(args.memory_length)

        self.schedule = LinearSchedule(50000, 0.1, 1)

    def get_intrinsic_reward(self, goal, state):
        return 1.0 if goal == state else 0.0

    def select_goal(self, state, t):
        if random.random() > self.schedule.value(t):
            state = th.from_numpy(state).type(d_type)
            return self.meta_controller(state).data.max(1)[1].cpu()
        return th.IntTensor([random.randrange(self.num_goal)])

    def select_action(self, joint_state_goal, t):
        if random.random() > self.schedule.value(t):
            joint_state_goal = th.from_numpy(joint_state_goal).type(d_type)
            return self.controller(joint_state_goal).data.max(1)[1].cpu()
        return th.IntTensor([random.randrange(self.num_action)])

    def update_meta_controller(self, gamma=1.0):
        if len(self.meta_replay_memory) < self.batch_size:
            return

        state_batch, goal_batch, n_state_batch, rew_batch, done_mask = self.meta_replay_memory.sample(self.batch_size)
        state_batch = th.from_numpy(state_batch).type(d_type)
        goal_batch = th.from_numpy(goal_batch).long()
        n_state_batch = th.from_numpy(n_state_batch).type(d_type)
        rew_batch = th.from_numpy(rew_batch).type(d_type)
        not_done_mask = th.from_numpy(1 - done_mask).type(d_type)
        # Compute current Q value, meta_controller takes only state and output value for every state-goal pair
        # We choose Q based on goal chosen.
        current_Q_values = self.meta_controller(state_batch).gather(1, goal_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = self.target_meta_controller(n_state_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = rew_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

        # Copy Q to target Q before updating parameters of Q
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        # Optimize the model
        self.meta_optimizer.zero_grad()
        loss.backward()
        for param in self.meta_controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step()

    def update_controller(self, gamma=1.0):
        if len(self.ctrl_replay_memory) < self.batch_size:
            return
        state_goal_batch, action_batch, next_state_goal_batch, in_reward_batch, done_mask = \
            self.ctrl_replay_memory.sample(self.batch_size)
        state_goal_batch = th.from_numpy(state_goal_batch).type(d_type)
        action_batch = th.from_numpy(action_batch).long()
        next_state_goal_batch = th.from_numpy(next_state_goal_batch).type(d_type)
        in_reward_batch = th.from_numpy(in_reward_batch).type(d_type)
        not_done_mask = th.from_numpy(1 - done_mask).type(d_type)
        # Compute current Q value, controller takes (state,goal) and output value for every (state,goal)-action pair
        # We choose Q based on action taken.
        current_Q_values = self.controller(state_goal_batch).gather(1, action_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = self.target_controller(next_state_goal_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = in_reward_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

        # Copy Q to target Q before updating parameters of Q
        self.target_controller.load_state_dict(self.controller.state_dict())
        # Optimize the model
        self.ctrl_optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.ctrl_optimizer.step()
