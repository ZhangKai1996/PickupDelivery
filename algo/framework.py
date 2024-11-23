from copy import deepcopy

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from algo.misc import get_folder, FloatTensor
from algo.trainer import Controller
from algo.visual import net_visual


class HieTrainer:
    def __init__(self, env, num_agents, folder=None, test=False, **kwargs):
        self.controller = Controller(
            num_agents,
            env.obs_space_ctrl.shape,
            env.act_space_ctrl.n,
            **kwargs
        )
        # Record the data of meta-controller and controller during training
        self.c_losses, self.a_losses = [], []
        # Build the save path of file, such as graph, log and parameters
        self.__addiction(folder=folder, test=test)

    def __addiction(self, folder, test):
        self.writer = None
        if folder is None:
            return

        self.path = get_folder(folder, makedir=(not test))
        if test: return

        if self.path['log_path'] is not None:
            self.writer = SummaryWriter(self.path['log_path'])

        if self.path['graph_path'] is not None:
            print('>>> Draw the net of Actor and Critic in Controller!')
            net_visual(
                dim_input=[self.controller.dim(label='actor')[0], ],
                net=self.controller.actors[0],
                d_type=FloatTensor,
                filename='actor',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            net_visual(
                dim_input=self.controller.dim(label='critic'),
                net=self.controller.critics[0],
                d_type=FloatTensor,
                filename='critic',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            print()

    def select_action(self, state, **kwargs):
        return self.controller.act(state, **kwargs)

    def add(self, *args):
        self.controller.add_experience(*args)

    def update(self, t):
        c_loss, a_loss = self.controller.update(t)
        if c_loss is None or a_loss is None:
            return

        self.c_losses.append(c_loss)
        self.a_losses.append(a_loss)
        if t % 100 == 0:
            prefix = 'ctrl'
            # Record and visual the loss value of Actor and Critic
            mean_c_loss = np.mean(self.c_losses, axis=0)
            self.scalars(
                key=prefix + '_critic_loss',
                value={'agent_{}'.format(i + 1): v for i, v in enumerate(mean_c_loss)},
                episode=t
            )
            mean_a_loss = np.mean(self.a_losses, axis=0)
            self.scalars(
                key=prefix + '_actor_loss',
                value={'agent_{}'.format(i + 1): v for i, v in enumerate(mean_a_loss)},
                episode=t
            )
            self.c_losses, self.a_losses = [], []

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.path['model_path']

        if load_path is not None:
            iterator = zip(self.controller.actors, self.controller.critics)
            for i, (a, c) in enumerate(iterator):
                a_state_dict = th.load(load_path + 'actor_{}.pth'.format(i)).state_dict()
                c_state_dict = th.load(load_path + 'critic_{}.pth'.format(i)).state_dict()
                a.load_state_dict(a_state_dict)
                c.load_state_dict(c_state_dict)
                self.controller.actors_target[i] = deepcopy(a)
                self.controller.critics_target[i] = deepcopy(c)
        else:
            print('Load path is empty!')
            raise NotImplementedError

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            iterator = zip(self.controller.actors, self.controller.critics)
            for i, (a, c) in enumerate(iterator):
                th.save(a, save_path + 'actor_{}.pth'.format(i))
                th.save(c, save_path + 'critic_{}.pth'.format(i))
        else:
            print('Save path is empty!')
            raise NotImplementedError

    def scalars(self, key, value, episode):
        self.writer.add_scalars(key, value, episode)

    def scalar(self, key, value, episode):
        self.writer.add_scalar(key, value, episode)

    def close(self):
        if self.writer is not None:
            self.writer.close()
