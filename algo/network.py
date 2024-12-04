import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_obs, dim_act):
        super(Critic, self).__init__()
        self.n_agent = n_agent

        self.FC1 = nn.Linear(dim_obs[0], 128)
        self.FC2 = nn.Linear((128 + dim_act) * n_agent, 256)
        self.FC3 = nn.Linear(256, 128)
        self.FC4 = nn.Linear(128, 1)
        self.fn = F.relu

    def forward(self, obs_n, act_n):
        batch_size = obs_n.size(0)

        out = self.fn(self.FC1(obs_n))
        out = th.cat([out, act_n], 2).view(batch_size, -1)
        out = self.fn(self.FC2(out))
        out = self.fn(self.FC3(out))
        out = self.FC4(out)
        return out


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_act):
        super(Actor, self).__init__()

        self.FC1 = nn.Linear(dim_obs[0], 256)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(128, dim_act)
        self.fn = F.relu

    def forward(self, obs_n):
        out = self.fn(self.FC1(obs_n))
        out = self.fn(self.FC2(out))
        out = self.FC3(out)
        return out

