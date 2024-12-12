from copy import deepcopy

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from algo.misc import get_folder, FloatTensor
from algo.trainer import TAController, PFController, TPController
from algo.visual import net_visual


class HieTrainer:
    def __init__(self, env, num_agents, num_orders, folder=None, test=False, **kwargs):
        self.ta_controller = TAController(
            n_agents=num_orders,
            dim_obs=env.obs_space_ta.shape,
            dim_act=env.act_space_ta.n,
            **kwargs
        )
        self.pf_controller = PFController(
            n_agents=num_agents,
            dim_obs=env.obs_space_pf.shape,
            dim_act=1,
            **kwargs
        )
        self.tp_controller = TPController(
            n_agents=num_agents,
            dim_obs=env.obs_space_tp.shape,
            dim_act=env.act_space_tp.n,
            **kwargs
        )
        # Record the data of meta-controller and controller during training
        self.losses = {'ta': {'actor': [], 'critic': []},
                       'pf': {'actor': [], 'critic': []},
                       'tp': {'actor': [], 'critic': []}}
        # Build the save path of file, such as graph, log and parameters
        self.__addiction(folder=folder, test=test)

    def __draw(self, label='ta'):
        print('>>> Draw the net of Actor and Critic in {}!'.format(label))
        ctrl = self.__get_controller(label=label)
        dim_input = ctrl.dim_input()
        net_visual(dim_input=dim_input['actor'],
                   net=ctrl.actors[0],
                   d_type=FloatTensor,
                   filename=label + '_actor',
                   directory=self.path['graph_path'],
                   format='png',
                   cleanup=True)
        net_visual(dim_input=dim_input['critic'],
                   net=ctrl.critics[0],
                   d_type=FloatTensor,
                   filename=label + '_critic',
                   directory=self.path['graph_path'],
                   format='png',
                   cleanup=True)

    def __addiction(self, folder, test):
        self.writer = None
        if folder is None: return

        self.path = get_folder(folder, makedir=(not test))
        if test: return

        if self.path['log_path'] is not None:
            self.writer = SummaryWriter(self.path['log_path'])

        if self.path['graph_path'] is not None:
            self.__draw(label='ta')
            self.__draw(label='pf')
            self.__draw(label='tp')

    def __get_controller(self, label='ta'):
        if label == 'ta':
            return self.ta_controller
        if label == 'pf':
            return self.pf_controller
        if label == 'tp':
            return self.tp_controller
        raise NotImplementedError

    def select_action(self, state, label='ta', **kwargs):
        return self.__get_controller(label=label).act(state, **kwargs)

    def add(self, *args, label='ta'):
        self.__get_controller(label=label).add_experience(*args)

    def update(self, t, label='ta'):
        controller = self.__get_controller(label=label)

        c_loss, a_loss = controller.update(t)
        if c_loss is None or a_loss is None:
            return

        self.losses[label]['critic'].append(c_loss)
        self.losses[label]['actor'].append(a_loss)

        if t % 100 == 0:
            # Record and visual the loss value of Actor and Critic
            mean_c_loss = np.mean(self.losses[label]['critic'], axis=0)
            value = {'agent_{}'.format(i + 1): v for i, v in enumerate(mean_c_loss)}
            self.scalars(key=label + '_critic_loss', value=value, episode=t)

            mean_a_loss = np.mean(self.losses[label]['actor'], axis=0)
            value = {'agent_{}'.format(i + 1): v for i, v in enumerate(mean_a_loss)}
            self.scalars(key=label + '_actor_loss', value=value, episode=t)
            self.scalar(key=label+'_decay', value=controller.decay(t), episode=t)

            self.losses[label]['critic'] = []
            self.losses[label]['actor'] = []

    def __load(self, load_path, label='ta'):
        controller = self.__get_controller(label=label)
        iterator = zip(controller.actors, controller.critics)
        for i, (a, c) in enumerate(iterator):
            a_state_dict = th.load(load_path + label + '_actor_{}.pth'.format(i)).state_dict()
            c_state_dict = th.load(load_path + label + '_critic_{}.pth'.format(i)).state_dict()
            a.load_state_dict(a_state_dict)
            c.load_state_dict(c_state_dict)
            controller.actors_target[i] = deepcopy(a)
            controller.critics_target[i] = deepcopy(c)

    def __save(self, save_path, label='tp'):
        controller = self.__get_controller(label=label)
        iterator = zip(controller.actors, controller.critics)
        for i, (a, c) in enumerate(iterator):
            th.save(a, save_path + label + '_actor_{}.pth'.format(i))
            th.save(c, save_path + label + '_critic_{}.pth'.format(i))

    def load_model(self, load_path=None, label=None):
        if load_path is None:
            load_path = self.path['model_path']

        if load_path is not None:
            if label is None:
                self.__load(load_path, label='ta')
                self.__load(load_path, label='pf')
                self.__load(load_path, label='tp')
            else:
                self.__load(load_path, label=label)
        else:
            print('Load path is empty!')
            raise NotImplementedError

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            self.__save(save_path, label='ta')
            self.__save(save_path, label='pf')
            self.__save(save_path, label='tp')
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
