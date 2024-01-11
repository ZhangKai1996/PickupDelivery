from copy import deepcopy

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from algo.misc import get_folder, FloatTensor
from algo.trainer import MetaController, Controller
from algo.visual import net_visual


class HieTrainer:
    def __init__(self, env, num_tasks, num_agents, folder=None, test=False, **kwargs):
        # Construct meta-controller and controller
        self.meta_controller = MetaController(
            num_tasks,
            env.obs_space_meta.shape,
            env.act_space_meta.n,
            **kwargs
        )
        self.controller = Controller(
            num_agents,
            env.obs_space_ctrl.shape,
            env.act_space_ctrl.n,
            **kwargs
        )
        # Record the data of meta-controller and controller during training
        self.meta_c_losses, self.meta_a_losses = [], []
        self.ctrl_c_losses, self.ctrl_a_losses = [], []
        # Build the save path of file, such as graph, log and parameters
        self.__addiction(folder=folder, test=test)

    def __addiction(self, folder, test):
        self.writer = None
        if folder is None:
            return

        self.path = get_folder(folder, makedir=(not test), allow_exist=True)
        if test: return

        if self.path['log_path'] is not None:
            self.writer = SummaryWriter(self.path['log_path'])

        if self.path['graph_path'] is not None:
            print('>>> Draw the net of Actor and Critic in Controller!')
            prefix = 'ctrl'
            net_visual(
                dim_input=[self.controller.dim(label='actor')[0], ],
                net=self.controller.actor,
                d_type=FloatTensor,
                filename=prefix + '_actor',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            net_visual(
                dim_input=self.controller.dim(label='critic'),
                net=self.controller.critic,
                d_type=FloatTensor,
                filename=prefix + '_critic',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            print('>>> Draw the net of Actor and Critic in Meta-Controller!')
            prefix = 'meta'
            net_visual(
                dim_input=[self.meta_controller.dim(label='actor')[0], ],
                net=self.meta_controller.actor,
                d_type=FloatTensor,
                filename=prefix + '_actor',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            net_visual(
                dim_input=self.meta_controller.dim(label='critic'),
                net=self.meta_controller.critic,
                d_type=FloatTensor,
                filename=prefix + '_critic',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            print()

    def select_scheme(self, state, t, **kwargs):
        return self.meta_controller.act(state, t, **kwargs)

    def select_action(self, state, t, **kwargs):
        return self.controller.act(state, t, **kwargs)

    def add(self, *args, label='ctrl'):
        if label == 'ctrl':
            self.controller.add_experience(*args)
        elif label == 'meta':
            self.meta_controller.add_experience(*args)
        else:
            raise NotImplementedError

    def update_controller(self, t):
        c_loss, a_loss = self.controller.update(t)
        if c_loss is None or a_loss is None:
            return

        self.ctrl_c_losses.append(c_loss)
        self.ctrl_a_losses.append(a_loss)
        if t % 100 == 0:
            # Record and visual the loss value of Actor and Critic
            mean_c_loss = np.mean(self.ctrl_c_losses, axis=0)
            self.scalars(
                key='ctrl_critic_loss',
                value={'agent_{}'.format(i + 1): v for i, v in enumerate(mean_c_loss)},
                episode=t
            )
            mean_a_loss = np.mean(self.ctrl_a_losses, axis=0)
            self.scalars(
                key='ctrl_actor_loss',
                value={'agent_{}'.format(i + 1): v for i, v in enumerate(mean_a_loss)},
                episode=t
            )
            self.ctrl_c_losses, self.ctrl_a_losses = [], []

    def update_meta_controller(self, t):
        c_loss, a_loss = self.meta_controller.update(t)
        if c_loss is None or a_loss is None:
            return

        self.meta_c_losses.append(c_loss)
        self.meta_a_losses.append(a_loss)
        if t % 100 == 0:
            mean_c_loss = np.mean(self.meta_c_losses, axis=0)
            self.scalars(
                key='meta_critic_loss',
                value={'agent_{}'.format(i + 1): v for i, v in enumerate(mean_c_loss)},
                episode=t
            )
            mean_a_loss = np.mean(self.meta_a_losses, axis=0)
            self.scalars(
                key='meta_actor_loss',
                value={'agent_{}'.format(i + 1): v for i, v in enumerate(mean_a_loss)},
                episode=t
            )
            self.meta_c_losses, self.meta_a_losses = [], []

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.path['model_path']

        if load_path is not None:
            a, c = self.controller.actor, self.controller.critic
            a_state_dict = th.load(load_path + 'ctrl_actor.pth').state_dict()
            c_state_dict = th.load(load_path + 'ctrl_critic.pth').state_dict()
            a.load_state_dict(a_state_dict)
            c.load_state_dict(c_state_dict)
            self.controller.actor_target = deepcopy(a)
            self.controller.critic_target = deepcopy(c)

            a, c = self.meta_controller.actor, self.meta_controller.critic
            a_state_dict = th.load(load_path + 'meta_actor.pth').state_dict()
            c_state_dict = th.load(load_path + 'meta_critic.pth').state_dict()
            a.load_state_dict(a_state_dict)
            c.load_state_dict(c_state_dict)
            self.meta_controller.actor_target = deepcopy(a)
            self.meta_controller.critic_target = deepcopy(c)
        else:
            print('Load path is empty!')
            raise NotImplementedError

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            th.save(self.controller.actor, save_path + 'ctrl_actor.pth')
            th.save(self.controller.critic, save_path + 'ctrl_critic.pth')
            th.save(self.meta_controller.actor, save_path + 'meta_actor.pth')
            th.save(self.meta_controller.critic, save_path + 'meta_critic.pth')
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
