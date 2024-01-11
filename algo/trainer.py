import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam

from .memory import ReplayMemory
from .network import Critic, Actor, MetaActor
from .misc import soft_update, FloatTensor, LinearSchedule


class Trainer:
    def act(self, obs_n, t): pass
    def update(self, t): pass
    def close(self): pass


class Controller(Trainer):
    def __init__(self, num_agents, dim_obs, dim_act, **kwargs):
        super(Controller, self).__init__()
        self.n_agents = num_agents
        self.n_actions = dim_act
        self.dim_obs = dim_obs

        self.kwargs = kwargs
        self.schedule = LinearSchedule(2000000, 0.1, 1)
        self.memory = ReplayMemory(kwargs['memory_length'])

        # Actor and target actor
        self.actor = Actor(dim_obs, dim_act)
        self.actor_target = Actor(dim_obs, dim_act)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=kwargs['a_lr'])
        # Critic and target critic
        self.critic = Critic(self.n_agents, dim_obs, dim_act)
        self.critic_target = Critic(self.n_agents, dim_obs, dim_act)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=kwargs['c_lr'])
        self.mse_loss = nn.MSELoss()

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

    def act(self, obs_n, t, test=False):
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = self.actor(obs_n.detach())
        if np.random.random() < self.schedule.value(t) and not test:
            noise = np.random.randn(self.n_agents, self.n_actions)
            act_n += th.from_numpy(noise).type(FloatTensor)
            act_n = th.tanh(act_n)
        return act_n.data.cpu().numpy()

    def update(self, t):
        if t <= self.kwargs['learning_start']:
            return None, None

        gamma = self.kwargs['gamma']

        transitions = self.memory.sample(self.kwargs['batch_size'])
        state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
        action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
        n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
        reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
        done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            self.critic_optimizer.zero_grad()
            current_q = self.critic(state_batch, action_batch)  # (32,1)
            n_actions = self.actor_target(n_states_batch)  # (32,3,5)
            target_next_q = self.critic_target(n_states_batch, n_actions)  # (32,1)
            target_q = target_next_q * gamma * (1 - done_batch[:, i]) + reward_batch[:, i]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actor(state_batch[:, i, :])
            loss_p = -self.critic(state_batch, ac).mean()
            loss_p.backward()
            self.actor_optimizer.step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())

        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critic_target, self.critic, self.kwargs['tau'])
                soft_update(self.actor_target, self.actor, self.kwargs['tau'])
        return c_loss, a_loss


class MetaController(Trainer):
    def __init__(self, n_agents, dim_obs, dim_act, **kwargs):
        super(MetaController, self).__init__()
        self.n_agents = n_agents
        self.n_actions = dim_act
        self.dim_obs = dim_obs
        self.kwargs = kwargs
        self.schedule = LinearSchedule(10000, 0.1, 1)
        self.memory = ReplayMemory(kwargs['memory_length'])

        # Actor and target actor
        self.actor = MetaActor(dim_obs, dim_act)
        self.actor_target = MetaActor(dim_obs, dim_act)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=kwargs['a_lr'])
        # Critic and target critic
        self.critic = Critic(n_agents, dim_obs, dim_act)
        self.critic_target = Critic(n_agents, dim_obs, dim_act)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=kwargs['c_lr'])
        self.mse_loss = nn.MSELoss()

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

    def add_experience(self, obs_n, act_n, next_obs_n, rew, done):
        rew_n = np.array([rew, ] * self.n_agents)
        done_n = np.array([done, ] * self.n_agents)
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, t, test=False):
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = self.actor(obs_n.detach())
        if np.random.random() < self.schedule.value(t) and not test:
            noise = np.random.randn(self.n_agents, self.n_actions)
            act_n += th.from_numpy(noise).type(FloatTensor)
            act_n = th.softmax(act_n, dim=0)
        return act_n.data.cpu().numpy()

    def update(self, t):
        if t <= self.kwargs['learning_start']:
            return None, None

        gamma = self.kwargs['gamma']

        transitions = self.memory.sample(self.kwargs['batch_size'])
        state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
        action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
        n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
        reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
        done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            self.critic_optimizer.zero_grad()
            current_q = self.critic(state_batch, action_batch)
            n_actions = self.actor_target(n_states_batch)
            target_next_q = self.critic_target(n_states_batch, n_actions)
            target_q = target_next_q * gamma * (1 - done_batch[:, i]) + reward_batch[:, i]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actor(state_batch[:, i, :])
            loss_p = -self.critic(state_batch, ac).mean()
            loss_p.backward()
            self.actor_optimizer.step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())

        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critic_target, self.critic, self.kwargs['tau'])
                soft_update(self.actor_target, self.actor, self.kwargs['tau'])
        return c_loss, a_loss
