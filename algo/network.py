import torch as th
import torch.nn as nn
import torch.nn.functional as F


funcs = {'relu': F.relu, 'tanh': F.tanh, 'softmax': F.softmax}


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, fn=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fn = funcs[fn] if fn is not None else None

    def forward(self, inputs):
        outputs = F.relu(self.fc1(inputs))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)
        return outputs if self.fn is None else self.fn(outputs)


class BiRNN1(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, fn=None):
        super(BiRNN1, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_fw = nn.RNNCell(input_size, hidden_size)
        self.rnn_bw = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.fn = funcs[fn] if fn is not None else None

    def forward(self, inputs):
        batch_size, seq_len, *_ = inputs.size()

        h_fw = th.zeros(batch_size, self.hidden_size)
        output_fw = []
        for t in range(seq_len):
            h_fw = self.rnn_fw(inputs[:, t], h_fw)
            output_fw.append(h_fw)

        h_bw = th.zeros(batch_size, self.hidden_size)
        output_bw = []
        for t in range(seq_len - 1, -1, -1):
            h_bw = self.rnn_bw(inputs[:, t], h_bw)
            output_bw.append(h_bw)

        output_fw = th.stack(output_fw, dim=1)
        output_bw = th.stack(output_bw[::-1], dim=1)
        outputs = th.cat((output_fw, output_bw), dim=2)
        outputs = self.fc(outputs).squeeze(-1)
        return outputs if self.fn is None else self.fn(outputs)

class BiRNN2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, fn=None):
        super(BiRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_fw = nn.RNNCell(input_size, hidden_size)
        self.rnn_bw = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.fn = funcs[fn] if fn is not None else None

    def forward(self, inputs):
        batch_size, seq_len, *_ = inputs.size()

        h_fw = th.zeros(batch_size, self.hidden_size)
        output_fw = []
        for t in range(seq_len):
            h_fw = self.rnn_fw(inputs[:, t], h_fw)
            output_fw.append(h_fw)

        h_bw = th.zeros(batch_size, self.hidden_size)
        output_bw = []
        for t in range(seq_len - 1, -1, -1):
            h_bw = self.rnn_bw(inputs[:, t], h_bw)
            output_bw.append(h_bw)

        outputs = th.cat((output_fw[-1], output_bw[-1]), dim=-1)
        outputs = self.fc(outputs)
        return outputs if self.fn is None else self.fn(outputs)
