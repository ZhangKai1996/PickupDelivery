import random
from collections import namedtuple

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from algo.replay_memory import ReplayMemory
from algo.misc import LinearSchedule

dtype = th.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)


class MetaController(nn.Module):
    def __init__(self, in_features=6, out_features=6):
        """
        Initialize a Meta-Controller of Hierarchical DQN network for the discrete mdp experiment
             in_features: number of features of input.
            out_features: number of features of output.
                      Ex: goal for meta-controller or action for controller
        """
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Controller(nn.Module):
    def __init__(self, in_features=12, out_features=2):
        """
        Initialize a Controller(given goal) of h-DQN for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


class HieDQN:
    """
    The Hierarchical-DQN Agent
    Parameters
    ----------
        num_goal: int
            The number of goal that agent can choose from
        num_action: int
            The number of action that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
    """

    def __init__(self, num_goal=6, num_action=2, replay_memory_size=10000, lr=1e-4, batch_size=128):
        # BUILD MODEL
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        # Construct meta-controller and controller
        self.meta_controller = MetaController().type(dtype)
        self.target_meta_controller = MetaController().type(dtype)
        self.controller = Controller().type(dtype)
        self.target_controller = Controller().type(dtype)
        # Construct the optimizers for meta-controller and controller
        opt = OptimizerSpec(constructor=optim.RMSprop, kwargs=dict(lr=lr, alpha=.95, eps=.01))
        self.meta_optimizer = opt.constructor(self.meta_controller.parameters(), **opt.kwargs)
        self.ctrl_optimizer = opt.constructor(self.controller.parameters(), **opt.kwargs)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayMemory(replay_memory_size)
        self.ctrl_replay_memory = ReplayMemory(replay_memory_size)
        self.schedule = LinearSchedule(50000, 0.1, 1)

    def get_intrinsic_reward(self, goal, state):
        return 1.0 if goal == state else 0.0

    def select_goal(self, state, t):
        sample = random.random()
        if sample > self.schedule.value(t):
            state = th.from_numpy(state).type(dtype)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return self.meta_controller(Variable(state, volatile=True)).data.max(1)[1].cpu()
        else:
            return th.IntTensor([random.randrange(self.num_goal)])

    def select_action(self, joint_state_goal, t):
        sample = random.random()
        if sample > self.schedule.value(t):
            joint_state_goal = th.from_numpy(joint_state_goal).type(dtype)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return self.controller(Variable(joint_state_goal, volatile=True)).data.max(1)[1].cpu()
        else:
            return th.IntTensor([random.randrange(self.num_action)])

    def update_meta_controller(self, gamma=1.0):
        if len(self.meta_replay_memory) < self.batch_size:
            return

        state_batch, goal_batch, n_state_batch, rew_batch, done_mask = self.meta_replay_memory.sample(self.batch_size)
        state_batch = Variable(th.from_numpy(state_batch).type(dtype))
        goal_batch = Variable(th.from_numpy(goal_batch).long())
        n_state_batch = Variable(th.from_numpy(n_state_batch).type(dtype))
        rew_batch = Variable(th.from_numpy(rew_batch).type(dtype))
        not_done_mask = Variable(th.from_numpy(1 - done_mask)).type(dtype)
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
        state_goal_batch = Variable(th.from_numpy(state_goal_batch).type(dtype))
        action_batch = Variable(th.from_numpy(action_batch).long())
        next_state_goal_batch = Variable(th.from_numpy(next_state_goal_batch).type(dtype))
        in_reward_batch = Variable(th.from_numpy(in_reward_batch).type(dtype))
        not_done_mask = Variable(th.from_numpy(1 - done_mask)).type(dtype)
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
