import os

import torch as th

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
FloatTensor = th.FloatTensor if not th.cuda.is_available() else th.cuda.FloatTensor
ByteTensor = th.ByteTensor if not th.cuda.is_available() else th.cuda.ByteTensor


def get_folder(folder, root='trained', has_log=False, has_graph=False, has_model=True, allow_exist=False):
    """
    数据记录（计算图、logs和网络参数）的保存文件路径
    """
    folder = os.path.join(root, folder)
    if os.path.exists(folder):
        if not allow_exist:
            raise FileExistsError

    if has_log:
        log_path = os.path.join(folder, 'logs/')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    else:
        log_path = None

    if has_graph:
        graph_path = os.path.join(folder, 'graph/')
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
    else:
        graph_path = None

    if has_model:
        model_path = os.path.join(folder, 'model/')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        model_path = None

    return {'folder': folder,
            'log_path': log_path,
            'graph_path': graph_path,
            'model_path': model_path}


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class LinearSchedule(object):
    def __init__(self, time_steps, final_p, initial_p=1.0):
        self.time_steps = time_steps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.time_steps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
