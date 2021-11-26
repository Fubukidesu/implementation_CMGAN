import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lgn_module_Azaixia import LGNGenerator_Azaixia_5, LGNGenerator_Azaixia_5_c
from torch.autograd.variable import Variable
from utils.cuda import to_cuda_if_available
from utils.initialization import load_or_initialize
import numpy as np
import torch
import torch.nn as nn
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
manualSeed = 999
torch.manual_seed(manualSeed)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def check_A():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # then train the discriminator only with fake data
    noise = Variable(torch.FloatTensor(3, 10).normal_())
    noise = to_cuda_if_available(noise)
    print('noise', noise)
    adj_A = np.zeros((5, 5))
    generator = LGNGenerator_Azaixia_5(adj_A = adj_A)
    load_or_initialize(generator, None)
    generator = to_cuda_if_available(generator)
    fake_features, adj_A1, output_10, Wa = generator(noise, training=True)
    fake_features = fake_features.detach()
    print(fake_features)
# check_A() 


def check_A_todevice():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # then train the discriminator only with fake data
    noise = Variable(torch.FloatTensor(3, 10).normal_()).to(device)
    print('noise', noise)
    adj_A = np.zeros((5, 5))
    generator = LGNGenerator_Azaixia_5(adj_A = adj_A)
    load_or_initialize(generator, None)
    generator = to_cuda_if_available(generator)
    fake_features, adj_A1, output_10, Wa = generator(noise, training=True)
    fake_features = fake_features.detach()
    print(fake_features)
# check_A_todevice()

def check_A_tod_app():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # then train the discriminator only with fake data
    noise = Variable(torch.FloatTensor(3, 10).normal_()).to(device)
    print('noise', noise)
    adj_A = np.zeros((5, 5))
    generator = LGNGenerator_Azaixia_5(adj_A = adj_A)
    generator.apply(weights_init)
    generator = generator.to(device)
    fake_features, adj_A1, output_10, Wa = generator(noise, training=True)
    fake_features = fake_features.detach()
    print(fake_features)
check_A_tod_app()

def check_A_c():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # then train the discriminator only with fake data
    noise = Variable(torch.FloatTensor(3, 10).normal_())
    noise = to_cuda_if_available(noise)
    fake_c = torch.randint(2, (3, 5), dtype=torch.float32)
    fake_c = to_cuda_if_available(fake_c)
    adj_A = np.zeros((5, 5))
    generator = LGNGenerator_Azaixia_5_c(adj_A = adj_A)
    load_or_initialize(generator, None)
    generator = to_cuda_if_available(generator)
    fake_features, adj_A1, output_10, Wa = generator(noise,fake_c, training=True)
    fake_features = fake_features.detach()
    print(fake_features)

# check_A_c()


