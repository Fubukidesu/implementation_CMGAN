import torch
import torch.nn as nn

n_classes = 5
image_size = 128  # All images will be resized to this size using a transformer.
latent_dim = 50  # Size of z latent vector (i.e. size of generator input)
n_channels = 3  # Number of channels in the training images. For color images this is 3
ngf = 128  # Size of feature maps in generator
ndf = 128  # Size of feature maps in discriminator
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
        self.latent_class_dim = 5  # 包含分类信息的噪声维数
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