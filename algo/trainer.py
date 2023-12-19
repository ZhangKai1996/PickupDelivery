from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam

from .memory import ReplayMemory, Experience
from .network import Critic, Actor
from .misc import soft_update, FloatTensor, LinearSchedule


class Trainer:
    def act(self, obs_n, t): pass
    def update(self, t): pass
    def close(self): pass


class Controller(Trainer):
    def __init__(self, dim_obs, dim_act, args):
        super(Controller, self).__init__()
        self.n_agents = args.num_agents
        self.n_actions = dim_act
        self.dim_obs = dim_obs

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau

        self.schedule = LinearSchedule(50000, 0.1, 1)
        self.memory = ReplayMemory(args.memory_length)

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(self.n_agents):
            actor = Actor(dim_obs, dim_act).cuda()
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(Adam(actor.parameters(), lr=args.a_lr))
            critic = Critic(self.n_agents, dim_obs, dim_act).cuda()
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(Adam(critic.parameters(), lr=args.c_lr))
        self.mse_loss = nn.MSELoss().cuda()

    def dim(self, label='actor', batch_size=1):
        if label == 'actor':
            dim_input = (batch_size,) + self.dim_obs
            dim_output = (batch_size, self.n_actions)
        elif label == 'critic':
            dim_input = (batch_size, self.n_agents,) + self.dim_obs
            dim_output = (batch_size, self.n_agents, self.n_actions)
        else:
            raise NotImplementedError
        return [dim_input, dim_output]

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, t):
        decay = np.random.random() <= self.schedule.value(t)
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.zeros(self.n_agents, self.n_actions)
        for i in range(self.n_agents):
            obs = obs_n[i, :].detach().unsqueeze(0)
            act = self.actors[i](obs).squeeze()
            if decay:
                act += th.from_numpy(np.random.randn(self.n_actions)).type(FloatTensor)
                act = th.clamp(act, -1.0, 1.0)
            act_n[i, :] = act
        return act_n.data.cpu().numpy()

    def update(self, t):
        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            state_batch = th.from_numpy(np.array(batch.state)).type(FloatTensor)
            action_batch = th.from_numpy(np.array(batch.action)).type(FloatTensor)
            n_states_batch = th.from_numpy(np.array(batch.next_state)).type(FloatTensor)
            reward_batch = th.from_numpy(np.array(batch.reward)).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(np.array(batch.done)).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            current_q = self.critics[i](state_batch, action_batch)
            n_actions = th.stack([self.actors_target[i](n_states_batch[:, i]) for i in range(self.n_agents)], dim=-1)
            target_next_q = self.critics_target[i](n_states_batch, n_actions)
            target_q = target_next_q * self.gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actors[i](state_batch[:, i, :])
            loss_p = -self.critics[i](state_batch, ac).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            if t % 100 == 0:
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())
        return c_loss, a_loss


class MetaController(Trainer):
    def __init__(self, dim_obs, args):
        super(MetaController, self).__init__()
        self.n_tasks = args.num_tasks
        self.n_actions = args.num_agents
        self.dim_obs = dim_obs

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.schedule = LinearSchedule(50000, 0.1, 1)
        self.memory = ReplayMemory(args.memory_length)

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for i in range(self.n_tasks):
            actor = Actor(dim_obs, self.n_actions).cuda()
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(Adam(actor.parameters(), lr=args.a_lr))
            critic = Critic(self.n_tasks, dim_obs, self.n_actions).cuda()
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(Adam(critic.parameters(), lr=args.c_lr))
        self.mse_loss = nn.MSELoss().cuda()

    def dim(self, label='actor', batch_size=1):
        if label == 'actor':
            dim_input = (batch_size,) + self.dim_obs
            dim_output = (batch_size, self.n_actions)
        elif label == 'critic':
            dim_input = (batch_size, self.n_tasks,) + self.dim_obs
            dim_output = (batch_size, self.n_tasks, self.n_actions)
        else:
            raise NotImplementedError
        return [dim_input, dim_output]

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, t, var_decay=False):
        decay = np.random.random() > self.schedule.value(t)
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.zeros(self.n_tasks, self.n_actions)
        for i in range(self.n_tasks):
            obs = obs_n[i, :].detach().unsqueeze(0)
            act = self.actors[i](obs).squeeze()
            if decay:
                act += th.from_numpy(np.random.randn(self.n_actions)).type(FloatTensor)
                act = th.clamp(act, -1.0, 1.0)
            act_n[i, :] = act
        return act_n.data.cpu().numpy()

    def update(self, t):
        c_loss, a_loss = [], []
        for i in range(self.n_tasks):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            state_batch = th.from_numpy(np.array(batch.state)).type(FloatTensor)
            action_batch = th.from_numpy(np.array(batch.action)).type(FloatTensor)
            n_states_batch = th.from_numpy(np.array(batch.next_state)).type(FloatTensor)
            reward_batch = th.from_numpy(np.array(batch.reward)).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(np.array(batch.done)).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            current_q = self.critics[i](state_batch, action_batch)
            n_actions = th.stack([self.actors_target[i](n_states_batch[:, i]) for i in range(self.n_tasks)], dim=-1)
            target_next_q = self.critics_target[i](n_states_batch, n_actions)
            target_q = target_next_q * self.gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actors[i](state_batch[:, i, :])
            loss_p = -self.critics[i](state_batch, ac).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            if t % 100 == 0:
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())
        return c_loss, a_loss
