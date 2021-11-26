import torch
from torch import nn
from multi_categorical import MultiCategorical
from utils.dag_gnn_utils import preprocess_adj_new1
from torch.autograd.variable import Variable


class LGNGenerator(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[2, 2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9):
        super(LGNGenerator, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=False, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # 把noise增加一个维度 [bs d] -> [bs d 1]
        noise = noise.unsqueeze(2).double()
        mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        return self.output(hidden, training=training, temperature=temperature), adj_A1


class LGNDiscriminator(nn.Module):

    def __init__(self, input_size=20, hidden_sizes=[100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)