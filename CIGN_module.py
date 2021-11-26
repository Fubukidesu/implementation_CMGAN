# cign模组
# 暂时不加矩阵，先把wgan的acgan实现出来。
import torch
import torch.nn as nn

class CIGNGenerator(nn.Module):
    def __init__(self, z_dim=128, class_num=5, output_dim=3, input_size=64):
        super(CIGNGenerator, self).__init__()
        self.z_dim = z_dim
        self.class_num = class_num
        self.output_dim = output_dim
        self.input_size = input_size

        # self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        # self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(
            nn.Linear(self.z_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size//4) ** 2),
            nn.BatchNorm1d(128 * (self.input_size//4) ** 2),
            nn.ReLU()
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, labels, noise):
        gen_input = torch.cat((labels, noise), 1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.input_size // 4, self.input_size // 4)
        img = self.conv_blocks(out)
        return img

class CIGNdiscriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, input_size=64, class_num=5):
        super(CIGNdiscriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) ** 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) ** 2)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c