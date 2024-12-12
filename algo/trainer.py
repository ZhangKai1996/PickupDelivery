from copy import deepcopy

import numpy as np

from .memory import ReplayMemory
from .network import *
from .misc import *


class Trainer:
    def __init__(self, **kwargs):
        self.tau = kwargs['tau']
        self.gamma = kwargs['gamma']
        self.batch_size = kwargs['batch_size']

        self.memory = ReplayMemory(kwargs['memory_length'])
        self.schedule = LinearSchedule(int(1e5), 0.1, 1)

    def decay(self, t): return self.schedule.value(t)
    def act(self, obs_n, t): pass
    def add_experience(self, *args): pass
    def update(self, t): pass
    def close(self): pass


class TAController(Trainer):
    def __init__(self, n_agents, dim_obs, dim_act, **kwargs):
        super(TAController, self).__init__(**kwargs)
        self.n_agents = n_agents
        self.dim_act = dim_act
        self.dim_obs = dim_obs

        self.schedule = LinearSchedule(int(1e5), 0.1, 1)

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(self.n_agents):
            actor = MLP(input_size=dim_obs[0], output_size=dim_act, fn='softmax')
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(th.optim.Adam(actor.parameters(), lr=kwargs['a_lr']))
            # Critic and target critic
            critic = MLP(input_size=(dim_obs[0] + dim_act) * n_agents, output_size=1)
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(th.optim.Adam(critic.parameters(), lr=kwargs['c_lr']))

        self.mse_loss = nn.MSELoss()

    def dim_input(self, batch_size=1):
        return {'actor': (batch_size,) + self.dim_obs,
                'critic': (batch_size, (self.dim_obs[0] + self.dim_act) * self.n_agents, )}

    def add_experience(self, obs_n, act_n, next_obs_n, rew, done):
        rew_n = np.array([rew, ] * self.n_agents)
        done_n = np.array([done, ] * self.n_agents)
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)
        return True

    def act(self, obs_n, t=None):
        explore = t is not None and np.random.random() <= self.decay(t)

        obs_n = th.from_numpy(obs_n).type(FloatTensor).unsqueeze(0)
        act_n = []
        for i in range(self.n_agents):
            act = self.actors[i](obs_n[:, i, :])
            if explore:
                noise = np.random.randn(*act.shape)
                act += th.from_numpy(noise).type(FloatTensor)
                act = th.softmax(act, dim=-1)
            act_n.append(act)
        act_n = th.cat(act_n, dim=0)
        return act_n.data.cpu().numpy()

    def update(self, t):
        if t <= int(1e3) or t >= int(1e5): return None, None

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)

            state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
            action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
            n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
            reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            current_q = self.critics[i](th.cat([state_batch, action_batch]).view(self.batch_size, -1))

            n_actions_batch = self.actors_target[i](n_states_batch)
            target_next_q = self.critics_target[i](th.cat([n_states_batch, n_actions_batch]).view(self.batch_size, -1))
            target_q = target_next_q * self.gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actors[i](state_batch[:, i, :])
            loss_p = -self.critics[i](th.cat([state_batch, ac]).view(self.batch_size, -1)).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())

        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        return c_loss, a_loss


class PFController(Trainer):
    def __init__(self, n_agents, dim_obs, dim_act, **kwargs):
        super(PFController, self).__init__(**kwargs)
        self.n_agents = n_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act

        self.schedule = LinearSchedule(int(2e4), 0.1, 1)

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(self.n_agents):
            actor = BiRNN1(input_size=dim_obs[0], output_size=dim_act, fn='softmax')
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(th.optim.Adam(actor.parameters(), lr=kwargs['a_lr']))
            # Critic and target critic
            critic = BiRNN2(input_size=(dim_obs[0]+dim_act)*n_agents, output_size=1)
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(th.optim.Adam(critic.parameters(), lr=kwargs['c_lr']))
        self.mse_loss = nn.MSELoss()

    def dim_input(self, batch_size=1):
        return {'actor': (batch_size, 2) + self.dim_obs,
                'critic': (batch_size, 2, (self.dim_obs[0]+self.dim_act)*self.n_agents, )}

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done):
        done_n = np.array([done, ] * self.n_agents)
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, t=None):
        explore = t is not None and np.random.random() <= self.decay(t)

        obs_n = th.from_numpy(obs_n).type(FloatTensor).unsqueeze(0)
        act_n = []
        for i in range(self.n_agents):
            act = self.actors[i](obs_n[:, i, :])
            if explore:
                noise = np.random.randn(*act.shape)
                act += th.from_numpy(noise).type(FloatTensor)
                act = th.softmax(act, dim=-1)
            act_n.append(act)
        act_n = th.cat(act_n, dim=0)
        return act_n.data.cpu().numpy()

    def update(self, t):
        if t <= int(1e3) or t>= int(1e5): return None, None

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)

            state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
            action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
            n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
            reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            sa_batch = th.cat([state_batch, action_batch.unsqueeze(-1)], dim=-1)
            sa_batch = sa_batch.view(self.batch_size, sa_batch.shape[2], -1)
            current_q = self.critics[i](sa_batch)

            n_actions = [self.actors_target[i](n_states_batch[:, i]) for i in range(self.n_agents)]
            n_actions = th.stack(n_actions, dim=1)

            sa_n_batch = th.cat([n_states_batch, n_actions.unsqueeze(-1)], dim=-1)
            sa_n_batch = sa_batch.view(self.batch_size, sa_n_batch.shape[2], -1)
            target_next_q = self.critics_target[i](sa_n_batch)

            target_q = target_next_q * self.gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = self.actors[i](state_batch[:, i, :])
            sa_batch = th.cat([state_batch, ac.unsqueeze(-1)], dim=-1)
            sa_batch = sa_batch.view(self.batch_size, sa_batch.shape[2], -1)
            loss_p = -self.critics[i](sa_batch).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())

        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        return c_loss, a_loss


class TPController(Trainer):
    def __init__(self, n_agents, dim_obs, dim_act, **kwargs):
        super(TPController, self).__init__(**kwargs)
        self.n_agents = n_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act

        self.schedule = LinearSchedule(int(1e5), 0.1, 1)

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(self.n_agents):
            # Actor and target actor
            actor = MLP(input_size=dim_obs[0], output_size=dim_act)
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(th.optim.Adam(actor.parameters(), lr=kwargs['a_lr']))
            # Critic and target critic
            critic = MLP(input_size=(dim_obs[0]+dim_act)*n_agents, output_size=1)
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(th.optim.Adam(critic.parameters(), lr=kwargs['c_lr']))
        self.mse_loss = nn.MSELoss()

    def dim_input(self, batch_size=1):
        return {'actor': (batch_size,) + self.dim_obs,
                'critic': (batch_size, (self.dim_obs[0] + self.dim_act) * self.n_agents,)}

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, t=None):
        explore = t is not None and np.random.random() <= self.decay(t)

        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.zeros(self.n_agents, self.dim_act)
        for i in range(self.n_agents):
            obs = obs_n[i, :].detach().unsqueeze(0)
            act_n[i, :] = self.actors[i](obs).squeeze()
        act_n = gumbel_softmax(act_n, hard=True) if explore else onehot_from_logit(act_n)
        return act_n.data.cpu().numpy()

    def update(self, t):
        if t <= int(1e4) or t >= int(1e6): return None, None
        if t % 100 != 0: return None, None

        c_loss, a_loss = [], []
        for i in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
            action_batch = th.from_numpy(transitions[1]).type(FloatTensor)
            n_states_batch = th.from_numpy(transitions[2]).type(FloatTensor)
            reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)
            done_batch = th.from_numpy(transitions[4]).type(FloatTensor).unsqueeze(dim=-1)

            self.critics_optimizer[i].zero_grad()
            sa_batch = th.cat([state_batch, action_batch], dim=2).view(self.batch_size, -1)
            current_q = self.critics[i](sa_batch)

            n_actions = [onehot_from_logit(self.actors_target[j](n_states_batch[:, i, :]))
                         for j in range(self.n_agents)]
            n_actions = th.stack(n_actions, dim=1)
            sa_n_batch = th.cat([n_states_batch, n_actions], dim=2).view(self.batch_size, -1)
            target_next_q = self.critics_target[i](sa_n_batch)
            target_q = target_next_q * self.gamma * (1 - done_batch[:, i, :]) + reward_batch[:, i, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[i].step()

            self.actors_optimizer[i].zero_grad()
            ac = action_batch.clone()
            ac[:, i, :] = gumbel_softmax(self.actors[i](state_batch[:, i, :]), hard=True)
            for j in range(self.n_agents):
                if j == i:
                    continue
                ac[:, j, :] = onehot_from_logit(self.actors[j](state_batch[:, j, :]))

            sa_batch = th.cat([state_batch, ac], dim=2).view(self.batch_size, -1)
            loss_p = -self.critics[i](sa_batch).mean()
            loss_p.backward()
            self.actors_optimizer[i].step()

            c_loss.append(loss_q.item())
            a_loss.append(loss_p.item())

        if t % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        return c_loss, a_loss


class SLController(Trainer):
    def __init__(self, n_agents, dim_obs, dim_act, **kwargs):
        super(SLController, self).__init__(**kwargs)
        self.n_agents = n_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act

        self.actors = []
        self.optimizers = []
        for i in range(n_agents):
            actor = BiRNN2(input_size=dim_obs[0], output_size=dim_act)
            self.actors.append(actor)
            self.optimizers.append(th.optim.Adam(actor.parameters(), lr=kwargs['a_lr']))
        self.mse_loss = nn.MSELoss()

    def dim_input(self, batch_size=1):
        return {'actor': (batch_size,) + self.dim_obs,
                'critic': (batch_size, (self.dim_obs[0] + self.dim_act) * self.n_agents,)}

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def act(self, obs_n, t=None):
        obs_n = th.from_numpy(obs_n).type(FloatTensor)
        act_n = th.zeros(self.n_agents, self.dim_act)
        for i in range(self.n_agents):
            obs = obs_n[i, :].detach().unsqueeze(0)
            act_n[i, :] = self.actors[i](obs).squeeze()
        return act_n.data.cpu().numpy()[:, 0]

    def update(self, t):
        if t <= int(1e2) or t >= int(1e5): return None, None

        losses = []
        for i in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            state_batch = th.from_numpy(transitions[0]).type(FloatTensor)
            action_batch = th.from_numpy(transitions[1]).type(FloatTensor).unsqueeze(dim=-1)
            reward_batch = th.from_numpy(transitions[3]).type(FloatTensor).unsqueeze(dim=-1)

            self.optimizers[i].zero_grad()
            sa_batch = th.cat([state_batch[:, i, :], action_batch[:, i, :]], dim=2)
            outputs = self.actors[i](sa_batch)
            loss = self.mse_loss(outputs, reward_batch[:, i, :].detach())
            loss.backward()
            self.optimizers[i].step()
            losses.append(loss.item())
        return None, losses