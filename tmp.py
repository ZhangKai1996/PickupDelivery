import math

import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的 RNN 单元
class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=math.sqrt(5))
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, inputs, hidden):
        hy = torch.tanh(torch.mm(inputs, self.W_ih) + self.b_ih + torch.mm(hidden, self.W_hh) + self.b_hh)
        return hy


# 定义双向 RNN
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_fw = SimpleRNNCell(input_size, hidden_size)
        self.rnn_bw = SimpleRNNCell(input_size, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, inputs):
        seq_len, batch_size, _ = inputs.size()
        h_fw = torch.zeros(batch_size, self.hidden_size)
        h_bw = torch.zeros(batch_size, self.hidden_size)

        output_fw = []
        output_bw = []

        for t in range(seq_len):
            h_fw = self.rnn_fw(inputs[t], h_fw)
            output_fw.append(h_fw)

        for t in range(seq_len - 1, -1, -1):
            h_bw = self.rnn_bw(inputs[t], h_bw)
            output_bw.append(h_bw)

        output_fw = torch.stack(output_fw, dim=0)
        output_bw = torch.stack(output_bw[::-1], dim=0)
        print(output_fw.shape, output_bw.shape)

        outputs = torch.cat((output_fw, output_bw), dim=2)
        outputs = self.fc(outputs)
        return outputs


def main():
    # 定义模型参数
    input_size = 10
    hidden_size = 20
    output_size = 1
    seq_len = 7
    batch_size = 32

    # 创建模型
    model = BiRNN(input_size, hidden_size, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 生成一些随机输入和目标输出
    inputs = torch.randn(seq_len, batch_size, input_size)
    print(inputs.shape)
    target = torch.randn(seq_len, batch_size, output_size)
    print(target.shape)
    # 训练步骤
    outputs = model(inputs)
    print(outputs.shape)
    loss = criterion(outputs, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Loss: {loss.item()}')


if __name__ == '__main__':
    main()
