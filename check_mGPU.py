import torch
import torch.nn as nn
import numpy as np
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import data_loader
from torch.autograd.variable import Variable
from torch.optim import Adam
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from formats import data_formats, loaders
from datasets import Dataset
from utils.categorical import load_variable_sizes_from_metadata
from utils.initialization import load_or_initialize
from utils.cuda import to_cuda_if_available, to_cpu_if_available
from utils.logger import Logger
from utils.wgan_gp import calculate_gradient_penalty
from utils.dag_gnn_utils import matrix_poly
from check_torchshape import encode_conti_onehot
from lgn_module import LGNGenerator, LGNDiscriminator
from lgn_module_Azaixia import LGNGenerator_Azaixia_5_c  # 默认是包含9*9矩阵的
from mulACGAN_module import Generator, Discriminator
import torch
from torch import nn
from multi_categorical import MultiCategorical
from utils.dag_gnn_utils import preprocess_adj_new1
from torch.autograd.variable import Variable

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
ngpu = 3
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class check_5_c(nn.Module):

    def __init__(self, noise_size=50, c_size=5, output_size=[1, 1, 1, 1, 1], hidden_sizes=[100], bn_decay=0.9): # 对应修改恩正维度
        super(check_5_c, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.zeros(5, 5).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0
        self.exp = nn.Linear(c_size, c_size)

        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size + c_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # pre -> 100
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

    def forward(self, noise, c, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            
            hidden = self.hidden_layers(torch.cat((noise, self.exp(c)), 1))
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])



# 设置权重初始值
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


net_A = check_5_c().to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    net_A = nn.DataParallel(net_A, list(range(ngpu)))

net_A.apply(weights_init)


fake_c = torch.randint(2, (3, 5), dtype=torch.float32, device=device)

# 将随机标签与noise1结合经过因果矩阵，并得到结果Loss和约束loss。
net_A.zero_grad()
noise1 = Variable(torch.FloatTensor(3, 50).normal_()).to(device)
fake_features, adj_A1, output_10, Wa = net_A(noise1, fake_c, training=True)
print('adj_A1', adj_A1)  # 通过将参数矩阵乘以3得到的值矩阵返回出来，这样3张卡就有3倍矩阵
adj_A11 = 0
# for name in net_A.state_dict():
#         print('name', name)
for name, item in net_A.named_parameters():
        if name == 'module.adj_A':
            print('item', item)
            adj_A11 = torch.sinh(3.*item)
print('adj_A11', adj_A11)  # 试着直接从模型里面取
print(fake_features)

# print(net_A)