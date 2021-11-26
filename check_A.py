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

lr=0.0002
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 2  # Number of GPUs available. Use 0 for CPU mode.
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def check_A():
    # 看看A训练得怎么样
    ckpt_path = 'try_LGN/try_ACGANwithA/ACGANwithA_Samples/checkpoint_iteration_63299.tar'
    net_G = Generator(ngpu).to(device)
    net_D = Discriminator(ngpu).to(device)
    adj_A = np.zeros((5, 5))
    net_A = LGNGenerator_Azaixia_5_c(adj_A=adj_A).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net_G = nn.DataParallel(net_G, list(range(ngpu)))
        net_D = nn.DataParallel(net_D, list(range(ngpu)))
        net_A = nn.DataParallel(net_A, list(range(ngpu)))
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path)
    last_epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    last_i = checkpoint['last_current_iteration']
    sample_noise = checkpoint['sample_noise']

    list_loss_D = checkpoint['list_loss_D']
    list_loss_G = checkpoint['list_loss_G']
    list_loss_A = checkpoint['list_loss_A']

    loss_D = list_loss_D[-1]
    loss_G = list_loss_G[-1]
    loss_A = list_loss_A[-1]

    net_D.load_state_dict(checkpoint['netD_state_dict'])
    net_G.load_state_dict(checkpoint['netG_state_dict'])
    net_A.load_state_dict(checkpoint['netA_state_dict'])

    # Setup Adam optimizers
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_A = optim.Adam(net_A.parameters(), lr=0.001)
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_A.load_state_dict(checkpoint['optimizer_A.state_dict'])
    net_D.eval()
    net_G.eval()
    net_A.eval()

    for name, item in net_A.named_parameters():
        if name == 'module.adj_A':
            adj_A1 = torch.sinh(3.*item)
            print(adj_A1)

# check_A()

# tensor([[-0.0024,  0.2127, -0.0550, -0.1200, -0.1394],
#         [-0.0604, -0.0263,  0.5781, -0.0562, -0.3762],
#         [-0.3464,  0.0271,  0.0200,  0.0753, -0.0200],
#         [ 0.2488,  0.4326, -0.3006, -0.0687,  0.3899],
#         [ 0.3085,  0.0413, -0.2387,  0.1794,  0.1918]], device='cuda:0',
#        dtype=torch.float64, grad_fn=<SinhBackward>)

def check_A_1():
    adj_A = np.zeros((5, 5))
    net_A = LGNGenerator_Azaixia_5_c(adj_A=adj_A).to(device)
    net_A = nn.DataParallel(net_A, list(range(ngpu)))
    net_A.load_state_dict(torch.load('try_ACGANwithA/ACGANwithA_Samples_1/5_net_A.pth'))
    for name, item in net_A.named_parameters():
        # print(name)
        if name == 'module.adj_A':
            # adj_A1 = torch.sinh(3.*item)
            # print(adj_A1)
            item = item.cpu().detach().numpy()
            print('adj_A', np.around(item, decimals=2))

check_A_1()
    