import numpy as np
import random
from collections import namedtuple

Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sample_batch = zip(*random.sample(self.memory, batch_size))
        obs_batch, act_batch, next_obs_batch, rew_batch, done_mask = sample_batch
        obs_batch = np.stack(obs_batch)
        act_batch = np.array(act_batch)
        next_obs_batch = np.stack(next_obs_batch)
        rew_batch = np.array(rew_batch)
        done_mask = np.array(done_mask)
        return obs_batch, act_batch, next_obs_batch, rew_batch, done_mask

    def __len__(self):
        return len(self.memory)
