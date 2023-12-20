from copy import deepcopy

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from algo.misc import get_folder, FloatTensor
from algo.trainer import MetaController, Controller
from algo.visual import net_visual



class HieTrainer:
    def __init__(self, env, num_tasks, num_agents, folder=None, **kwargs):
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
        self.__addiction(folder=folder)

    def __addiction(self, folder):
        self.writer = None
        if folder is None:
            return

        self.path = get_folder(folder, has_graph=True, has_log=True, allow_exist=True)
        if self.path['log_path'] is not None:
            self.writer = SummaryWriter(self.path['log_path'])

        if self.path['graph_path'] is not None:
            print('>>> Draw the net of Actor and Critic in Controller!')
            prefix = 'ctrl'
            net_visual(
                dim_input=[self.controller.dim(label='actor')[0], ],
                net=self.controller.actors[0],
                d_type=FloatTensor,
                filename=prefix+'_actor',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            net_visual(
                dim_input=self.controller.dim(label='critic'),
                net=self.controller.critics[0],
                d_type=FloatTensor,
                filename=prefix+'_critic',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            print('>>> Draw the net of Actor and Critic in Meta-Controller!')
            prefix = 'meta'
            net_visual(
                dim_input=[self.meta_controller.dim(label='actor')[0], ],
                net=self.meta_controller.actors[0],
                d_type=FloatTensor,
                filename=prefix+'_actor',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            net_visual(
                dim_input=self.meta_controller.dim(label='critic'),
                net=self.meta_controller.critics[0],
                d_type=FloatTensor,
                filename=prefix+'_critic',
                directory=self.path['graph_path'],
                format='png',
                cleanup=True
            )
            print()

    def select_scheme(self, state, t):
        return self.meta_controller.act(state, t)

    def select_action(self, joint_state_goal, t):
        return self.controller.act(joint_state_goal, t)

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
            prefix = 'ctrl'
            # Record and visual the loss value of Actor and Critic
            mean_c_loss = np.mean(self.ctrl_c_losses, axis=0)
            self.scalars(
                key=prefix+'_critic_loss',
                value={'agent_{}'.format(i+1): v for i, v in enumerate(mean_c_loss)},
                episode=t
            )
            mean_a_loss = np.mean(self.ctrl_a_losses, axis=0)
            self.scalars(
                key=prefix+'_actor_loss',
                value={'agent_{}'.format(i+1): v for i, v in enumerate(mean_a_loss)},
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
            prefix = 'meta'
            mean_c_loss = np.mean(self.meta_c_losses, axis=0)
            self.scalars(
                key=prefix+'_critic_loss',
                value={'agent_{}'.format(i+1): v for i, v in enumerate(mean_c_loss)},
                episode=t
            )
            mean_a_loss = np.mean(self.meta_a_losses, axis=0)
            self.scalars(
                key=prefix+'_actor_loss',
                value={'agent_{}'.format(i+1): v for i, v in enumerate(mean_a_loss)},
                episode=t
            )
            self.meta_c_losses, self.meta_a_losses = [], []

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.path['model_path']

        if load_path is not None:
            prefix = 'ctrl'
            iterator = zip(self.controller.actors, self.controller.critics)
            for i, (a, c) in enumerate(iterator):
                a_state_dict = th.load(load_path + prefix+'_actor_{}.pth'.format(i)).state_dict()
                c_state_dict = th.load(load_path + prefix+'_critic_{}.pth'.format(i)).state_dict()
                a.load_state_dict(a_state_dict)
                c.load_state_dict(c_state_dict)
                self.controller.actors_target[i] = deepcopy(a)
                self.controller.critics_target[i] = deepcopy(c)
            prefix = 'meta'
            iterator = zip(self.meta_controller.actors, self.meta_controller.critics)
            for i, (a, c) in enumerate(iterator):
                a_state_dict = th.load(load_path + prefix+'_actor_{}.pth'.format(i)).state_dict()
                c_state_dict = th.load(load_path + prefix+'_critic_{}.pth'.format(i)).state_dict()
                a.load_state_dict(a_state_dict)
                c.load_state_dict(c_state_dict)
                self.meta_controller.actors_target[i] = deepcopy(a)
                self.meta_controller.critics_target[i] = deepcopy(c)
        else:
            print('Load path is empty!')
            raise NotImplementedError

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            prefix = 'ctrl'
            iterator = zip(self.controller.actors, self.controller.critics)
            for i, (a, c) in enumerate(iterator):
                th.save(a, save_path + prefix + '_actor_{}.pth'.format(i))
                th.save(c, save_path + prefix + '_critic_{}.pth'.format(i))
            prefix = 'meta'
            iterator = zip(self.meta_controller.actors, self.meta_controller.critics)
            for i, (a, c) in enumerate(iterator):
                th.save(a, save_path + prefix + '_actor_{}.pth'.format(i))
                th.save(c, save_path + prefix + '_critic_{}.pth'.format(i))
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
