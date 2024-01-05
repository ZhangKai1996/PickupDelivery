import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam

from .memory import ReplayMemory
from .network import Critic, Actor
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

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(self.n_agents):
            # Actor and target actor
            actor = Actor(dim_obs, dim_act)
            self.actors.append(actor)
            actor_target = Actor(dim_obs, dim_act)
            actor_target.load_state_dict(actor.state_dict())
            self.actors_target.append(actor_target)
            self.actors_optimizer.append(Adam(actor.parameters(), lr=kwargs['a_lr']))
            # Critic and target critic
            critic = Critic(self.n_agents, dim_obs, dim_act)
            self.critics.append(critic)
            critic_target = Critic(self.n_agents, dim_obs, dim_act)
            critic_target.load_state_dict(critic.state_dict())
            self.critics_target.append(critic_target)
            self.critics_optimizer.append(Adam(critic.parameters(), lr=kwargs['c_lr']))
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
        decay = np.random.random() <= self.schedule.value(t) and not test
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
        if t <= self.kwargs['learning_start']:
            return None, None

        batch_size = self.kwargs['batch_size']
        gamma = self.kwargs['gamma']
        tau = self.kwargs['tau']

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            transitions = self.memory.sample(batch_size)
            state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
            action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
            n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
            reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            current_q = self.critics[i](state_batch, action_batch)
            n_actions = [self.actors_target[i](n_states_batch[:, i, :]) for i in range(self.n_agents)]
            n_actions = th.stack(n_actions, dim=1)
            target_next_q = self.critics_target[i](n_states_batch, n_actions)
            target_q = target_next_q * gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actors[i](state_batch[:, i, :])
            loss_p = -self.critics[i](state_batch, ac).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())
        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], tau)
                soft_update(self.actors_target[i], self.actors[i], tau)
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

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for i in range(n_agents):
            # Actor and target actor
            actor = Actor(dim_obs, dim_act, activate='softmax')
            self.actors.append(actor)
            actor_target = Actor(dim_obs, dim_act, activate='softmax')
            actor_target.load_state_dict(actor.state_dict())
            self.actors_target.append(actor_target)
            self.actors_optimizer.append(Adam(actor.parameters(), lr=kwargs['a_lr']))
            # Critic and target critic
            critic = Critic(n_agents, dim_obs, dim_act)
            self.critics.append(critic)
            critic_target = Critic(n_agents, dim_obs, dim_act)
            critic_target.load_state_dict(critic.state_dict())
            self.critics_target.append(critic_target)
            self.critics_optimizer.append(Adam(critic.parameters(), lr=kwargs['c_lr']))
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
        decay = np.random.random() > self.schedule.value(t) and not test
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.zeros(self.n_agents, self.n_actions)
        for i in range(self.n_agents):
            obs = obs_n[i, :].detach().unsqueeze(0)
            act = self.actors[i](obs).squeeze()
            if decay:
                act += th.from_numpy(np.random.randn(self.n_actions)).type(FloatTensor)
                act = th.softmax(act, dim=0)
            act_n[i, :] = act
        return act_n.data.cpu().numpy()

    def update(self, t):
        if t <= self.kwargs['learning_start']:
            return None, None

        batch_size = self.kwargs['batch_size']
        gamma = self.kwargs['gamma']
        tau = self.kwargs['tau']

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            transitions = self.memory.sample(batch_size)

            state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
            action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
            n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
            reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            current_q = self.critics[i](state_batch, action_batch)
            n_actions = th.stack([self.actors_target[i](n_states_batch[:, i]) for i in range(self.n_agents)], dim=1)
            target_next_q = self.critics_target[i](n_states_batch, n_actions)
            target_q = target_next_q * gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actors[i](state_batch[:, i, :])
            loss_p = -self.critics[i](state_batch, ac).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())
        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], tau)
                soft_update(self.actors_target[i], self.actors[i], tau)
        return c_loss, a_loss
