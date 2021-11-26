import torchvision.utils as vutils
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
latent_dim = 100
image_size = 128
n_channels = 3  # Number of channels in the training images. For color images this is 3
ngf = 128  # Size of feature maps in generator
ndf = 128  # Size of feature maps in discriminator
# sample_noise 按照特征的数量，按照二进制顺序产生噪声一致但标签不同的sample对。
n_classes = 5
n_sample = 32  # 生成图片数量 最好是32的倍数
manualSeed = 337
torch.manual_seed(manualSeed)
# manual_seed的作用期很短


# Pytorch 没有nn.Reshape, 且不推荐使用 Why？？
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # 自动取得batch维
        return x.view((x.size(0),) + self.shape)
        # 若使用下式，batch的维数只能用-1代指
        # return x.view(self.shape)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.latent_class_dim = 10  # 包含分类信息的噪声维数
        self.exp = nn.Linear(n_classes, self.latent_class_dim)
        self.main = nn.Sequential(

            nn.Linear(latent_dim + self.latent_class_dim, ngf * 8 * (image_size // 16) ** 2),
            Reshape(ngf * 8, image_size // 16, image_size // 16),
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x (image_size//8) x (image_size//8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x (image_size//4) x (image_size//4)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x (image_size//2) x (image_size//2)

            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x image_size x image_size
        )

    def forward(self, z, c):
        return self.main(torch.cat((z, self.exp(c)), 1))


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x image_size x image_size
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x (image_size//2) x (image_size//2)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 2, image_size // 4, image_size // 4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x (image_size//4) x (image_size//4)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 4, image_size // 8, image_size // 8]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x (image_size//8) x (image_size//8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 8, image_size // 16, image_size // 16]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x (image_size//16) x (image_size//16)
            Reshape(ndf * 8 * (image_size // 16) ** 2),
        )

        self.adv = nn.Sequential(
            nn.Linear(ndf * 8 * (image_size // 16) ** 2, 1),
            # 注意没有WGAN-GP没有nn.Sigmoid()
        )

        self.aux = nn.Sequential(
            nn.Linear(ndf * 8 * (image_size // 16) ** 2, n_classes),
            # nn.Softmax(1)
            # 不能使用Softmax，因为各标签不相关，和也不一定为1
            nn.Sigmoid()
        )

    def forward(self, input):
        feature = self.main(input)
        v = self.adv(feature)
        c = self.aux(feature)
        return v, c

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

net_G = Generator(ngpu).to(device)
# net_D = Discriminator(ngpu).to(device)

checkpoint = torch.load('cgan-acgan-celeba-multi-label-pytorch-master/ACGAN+_Samples/checkpoint_iteration_150000.tar')
# net_D.load_state_dict(checkpoint['netD_state_dict'])
net_G.load_state_dict(checkpoint['netG_state_dict'])
# net_D.eval()
net_G.eval()



# 测试干预效果的代码
sample_noise = torch.randn(n_sample, latent_dim, device=device)
for i in range(n_sample):
    sample_noise[i] = sample_noise[i - i % 4]
# 4: 'Bald'  20: 'Male' 22: 'Mustache' 24: 'No_Beard' 39: 'Young'
# Male_Young
# s = [[0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0],
# [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0],
# [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0],
# [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0]]
# Bald_Male
# s = [[1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1],
# [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1],
# [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1],
# [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1]]
# 女性带秃头，带beard
# s = [[1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0],
# [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0],
# [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0],
# [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0]]
# Male->No_Beard(no_beard=0时，大多数Male=1'也有young变化'，干预之后发现，也可以male=0)
# s = [[0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0],
# [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0],
# [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0],
# [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
# Male->Young(Young=0时，大多数Male=1，干预之后，可以有Male=0)
s = [[0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0],
[0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0],
[0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0],
[0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0]]
sample_labels = torch.tensor(s, dtype=torch.float32, device=device)

samples = net_G(sample_noise, sample_labels).cpu()
vutils.save_image(samples, os.path.join('cgan-acgan-celeba-multi-label-pytorch-master/combine_samples/male_young', 'check_combine_337_Male->Young.jpg'), padding=2, normalize=True)

# 生成图片用于测试图像质量的代码
# sample_noise = torch.randn(n_sample, latent_dim, device=device)
# sample_labels = torch.zeros(n_sample, n_classes, dtype=torch.float32, device=device)
# # 生成噪声相同，但标签不同的示例噪声序列
# for i in range(n_sample):
#     sample_noise[i] = sample_noise[i - i % 32]
#     # print(i % n_condition)
#     bi = bin(i % 32)[2:].zfill(n_classes)
#     # print(bi)
#     for j in range(len(bi)):
#         if bi[j] == '1':
#             sample_labels[i][j] = 1
#     print('第%d个噪声标签对设定完毕' % i)

# img_index = 1
# for idx in range(0, 3200, 32): # 再分Mini批次给G生成图片，以免G负担太大
#     sample_noise_mini = sample_noise[idx: idx+32]
#     sample_labels_mini = sample_labels[idx: idx+32]
#     # print('sample_noise_mini', sample_noise_mini)
#     # print('sample_labels_mini', sample_labels_mini)
#     samples_mini = net_G(sample_noise_mini, sample_labels_mini).cpu()
#     print('第%d批(100批,每批32张)图片生成完毕' % idx)
#     for sample in samples_mini:
#         # vutils.save_image(sample, os.path.join('cgan-acgan-celeba-multi-label-pytorch-master/combine_samples', 'check_combine.jpg'), padding=2, normalize=True)
#         # print(sample)
#         vutils.save_image(sample, os.path.join('SNGAN_Projection_Pytorch-master/dataset1/exp2_3200_fake', 'img_%d.jpg' % img_index), normalize=True)
#         print('已生成%d张图片' % img_index)
#         img_index = img_index + 1

#一张看看效果 在0卡上没问题
# sample_noise = torch.randn(n_sample, latent_dim, device=device)
# # for i in range(n_sample):
# #     sample_noise[i] = sample_noise[i - i % 4]
# # 4: 'Bald'  20: 'Male' 22: 'Mustache' 24: 'No_Beard' 39: 'Young'
# s = [[0, 0, 0, 1, 1]]
# sample_labels = torch.tensor(s, dtype=torch.float32, device=device)

# samples = net_G(sample_noise, sample_labels).cpu()
# vutils.save_image(samples, os.path.join('cgan-acgan-celeba-multi-label-pytorch-master/combine_samples', 'check_combine.jpg'), padding=2, normalize=True)