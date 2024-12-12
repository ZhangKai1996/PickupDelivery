import torch as th
from torchviz import make_dot


def net_visual(dim_input, net, d_type=th.FloatTensor, **kwargs):
    """
    支持MLP、CNN和RNN三种网络的可视化。
    """
    print('------------Net Visualization-----------------')
    print('-Name:', kwargs['filename'])
    print('-Inputs:', dim_input)
    x = th.randn(*dim_input).type(d_type).requires_grad_(True)  # 定义一个网络的输入值
    y = net(x)  # 获取网络的预测值
    print('-Outputs:', y.shape)
    net_vis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    net_vis.render(**kwargs)     # 生成文件
    print('-Save to ' + kwargs['directory'] + '{}.{}'.format(kwargs['filename'], kwargs['format']))
    print('----------------------------------------------')
