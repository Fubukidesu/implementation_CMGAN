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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

is_load = False
ckpt_path = 'try_LGN/try_ACGANwithA/ACGANwithA_Samples/checkpoint_iteration_63299.tar'

lr=0.0002
n_critic = 2
lam_gp = 10
latent_dim = 50
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 2  # Number of GPUs available. Use 0 for CPU mode.
workers = 0  # Number of workers for dataloader
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# 数据库部分
# Root directory for dataset
batch_size = 64  # Batch size during training
image_size = 128  # All images will be resized to this size using a transformer.
img_path = 'data/celebA/img_align_celeba'
attr_path = 'data/celebA/list_attr_celeba.txt'
samples_path = 'try_ACGANwithA/ACGANwithA_Samples'
os.makedirs(samples_path, exist_ok=True)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
#celebA数据
dataset = data_loader.CelebA_Slim(img_path=img_path,
                                    attr_path=attr_path,
                                    transform=transform,
                                    slice=[0, -1])  # CelebA_Slim的attr参数列表选择了若干特征

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0, drop_last=True)
classes = dataset.idx2attr
n_classes = len(classes)
print('classes:', classes)  # classes: {0: 'Bald', 1: 'Male', 2: 'Mustache', 3: 'No_Beard', 4: 'Young'}

# --------------------------------------------------------------------------------------------------
# sample_noise 按照特征的数量，按照二进制顺序产生噪声一致但标签不同的sample对。
n_sample = 64  # 生成图片数量
manualSeed = 999
torch.manual_seed(manualSeed)
# manual_seed的作用期很短

n_condition = 2 ** n_classes

# global sample_noise
# global sample_labels
sample_noise = torch.randn(n_sample, latent_dim, device=device)
sample_labels = torch.zeros(n_sample, n_classes, dtype=torch.float32, device=device)

# 生成噪声相同，但标签不同的示例噪声序列
for i in range(n_sample):
    sample_noise[i] = sample_noise[i - i % n_condition]
    # print(i % n_condition)
    bi = bin(i % n_condition)[2:].zfill(n_classes)
    # print(bi)
    for j in range(len(bi)):
        if bi[j] == '1':
            sample_labels[i][j] = 1
print('sample对构造完毕')
print('sample_noise', sample_noise)
# print('sample_labels', sample_labels)

# 设置权重初始值
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

net_G = Generator(ngpu).to(device)
net_D = Discriminator(ngpu).to(device)
adj_A = np.zeros((5, 5))
net_A = LGNGenerator_Azaixia_5_c(adj_A=adj_A).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    net_G = nn.DataParallel(net_G, list(range(ngpu)))
    net_D = nn.DataParallel(net_D, list(range(ngpu)))
    net_A = nn.DataParallel(net_A, list(range(ngpu)))

net_G.apply(weights_init)
net_D.apply(weights_init)
net_A.apply(weights_init)


# print(net_G)
# print(net_D)

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)  # (I+(A*A)/m)^m
    h_A = torch.trace(expm_A) - m  # tr[(I+(A*A)/m)^m] - m，即h_a
    return h_A

# Initialize Loss function定义损失函数方法（多标签的二值交叉熵损失
def MBCE(input, target, esp=1e-19):
    loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
        (1 - target) * torch.log((1 - input).clamp_min(esp)))
    return loss

# Setup Adam optimizers
optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_A = optim.Adam(net_A.parameters(), lr=0.001)

# 计算梯度惩罚值
def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device=device)
    z = x + alpha * (y - x)

    # gradient penalty
    # z = Variable(z, requires_grad=True).to(device)
    # z = z.to(device)
    z.requires_grad = True
    o = f(z)[0]
    ones = torch.ones(o.size(), device=device)
    g = autograd.grad(o, z, grad_outputs=ones, create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp
# 画图
def save_diagram(list_1, list_2, list_3,
                    label1="D", label2="G", label3='A',
                    title="Generator and Discriminator and Anet loss During Training",
                    x_label="iterations", y_label="Loss",
                    path=samples_path,
                    name='loss.jpg'
                    ):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(list_1, label=label1)
    plt.plot(list_2, label=label2)
    plt.plot(list_3, label=label3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(path, name))
    plt.close()

# 训练中断后初始化模型
def ini_model(ckpt_path):
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
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_A.load_state_dict(checkpoint['optimizer_A.state_dict'])
    net_D.eval()
    net_G.eval()
    net_A.eval()
    

# 训练函数
def train(lambda_A, c_A, sample_noise, sample_labels, is_load=False, max_epochs=20):  # 不应该吧模型初始化放在train里面，外面初始，里面保持false

    lambda_A = lambda_A.cuda()
    c_A = c_A.cuda()
    if is_load:
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
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A.state_dict'])
        net_D.eval()
        net_G.eval()
        net_A.eval()

    else:
        last_epoch = 0
        iteration = 0

        list_loss_G = []
        list_loss_D = []
        list_loss_A = []

    print("Starting Training Loop...")
    for epoch in range(last_epoch, max_epochs):
        # 若读取，重置当前周期，且只执行一次
        str_i = 0
        if is_load:
            str_i = last_i
        for i, (real_img, real_c) in enumerate(dataloader, str_i):

            # -----------------------------------------------------------
            # Initial batch
            real_img, real_c = real_img.to(device), real_c.to(device)
            real_batch_size = real_img.size(0)
            noise2 = torch.randn(real_batch_size, latent_dim, device=device)
            # random label for computer loss
            fake_c = torch.randint(2, (real_batch_size, n_classes), dtype=torch.float32, device=device)

            # 将随机标签与noise1结合经过因果矩阵，并得到结果Loss和约束loss。
            net_A.zero_grad()
            noise1 = Variable(torch.FloatTensor(real_batch_size, latent_dim).normal_()).to(device)
            fake_features, adj_A11, output_10, Wa = net_A(noise1, fake_c, training=True)
            adj_A1 = 0
            for name, item in net_A.named_parameters():
                if name == 'module.adj_A':
                    adj_A1 = torch.sinh(3.*item)
            # print('adj_A1', adj_A1)
            loss_A_v = MBCE(fake_features, real_c)
            # compute h(A)
            # adj_A1 = adj_A1.double()
            h_A = _h_A(adj_A1, 5).type(torch.cuda.FloatTensor)  # 修改对应特征维度
            # print(h_A)
            lambda_A = lambda_A.type(torch.cuda.FloatTensor)
            # print(lambda_A)
            c_A = c_A.type(torch.cuda.FloatTensor)
            loss_A_tr = lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(adj_A1*adj_A1).type(torch.cuda.FloatTensor)

            fake_img = net_G(noise2, fake_features)

            # -----------------------------------------------------------
            # Update D network: minimize: -(D(x) - D(G(z)))+ lambda_gp * gp + class_loss
            net_D.zero_grad()
            
            v, c = net_D(real_img)
            loss_real = (- torch.mean(v) + MBCE(c, real_c)) * 0.5
            v, c = net_D(fake_img.detach())
            loss_fake = (torch.mean(v) + MBCE(c, fake_features.detach())) * 0.5
            gp = gradient_penalty(real_img.detach(), fake_img.detach(), net_D)
            loss_D = (loss_real + loss_fake) * 0.5 + lam_gp * gp  # total loss of D

            # Update D
            loss_D.backward()
            optimizer_D.step()

            # -----------------------------------------------------------
            # Update G network: maximize D(G(z)) , equal to minimize - D(G(z))  update A net
            if i % n_critic == 0:
                net_G.zero_grad()

                # Calculate G loss
                v, c = net_D(fake_img)

                loss_G = (- torch.mean(v) + MBCE(c, fake_features)) * 0.5

                # Update G
                loss_G.backward(retain_graph=True)
                optimizer_G.step()

                # Calculate A loss
                loss_A = loss_A_v + loss_A_tr
                loss_A.backward()
                optimizer_A.step()

            # -----------------------------------------------------------
            # Output training stats
            with torch.no_grad():
                list_loss_D.append(loss_D.item())

                if type(loss_G) == float:
                    list_loss_G.append(loss_G)
                else:
                    list_loss_G.append(loss_G.item())

                list_loss_A.append(loss_A.item())

                if i % 5 == 0 or is_load:
                    print(
                        '[%d/%d][%2d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_A: %.4f'
                        % (epoch, max_epochs, i, len(dataloader),
                           list_loss_D[-1], list_loss_G[-1], list_loss_A[-1]))

                # Check how the generator is doing by saving G's output on sample_noise
                if (iteration % 500 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    # with torch.no_grad():
                    #     sample = netG(fixed_noise).detach().cpu()
                    samples = net_G(sample_noise, sample_labels).cpu()
                    vutils.save_image(samples, os.path.join(samples_path, 'iteration_%d.jpg' % iteration), padding=2,
                                      normalize=True)
                    save_diagram(list_loss_D, list_loss_G, list_loss_A, name='loss.jpg')
                
                # Save model
                if (iteration % 10000 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    save_path = os.path.join(samples_path, 'checkpoint_iteration_%d.tar' % iteration)
                    torch.save({
                        'epoch': epoch,
                        'iteration': iteration,
                        'last_current_iteration': i,
                        'netD_state_dict': net_D.state_dict(),
                        'netG_state_dict': net_G.state_dict(),
                        'netA_state_dict': net_A.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_A.state_dict': optimizer_A.state_dict(),
                        'list_loss_D': list_loss_D,
                        'list_loss_G': list_loss_G,
                        'list_loss_A': list_loss_A,
                        'sample_noise': sample_noise
                    }, save_path)
            
            # iteration: total iteration, i: iteration of current epoch
            iteration += 1
            is_load = False
    return adj_A1

lambda_A = [0.]
lambda_A = torch.from_numpy(np.array(lambda_A))
c_A = [1.]
c_A = torch.from_numpy(np.array(c_A))
h_A_old = np.inf

for step_k in range(20):  # 为了找lambda和c而进行迭代
    while c_A < 1e+20:  # 当c_a小于1e+20时不断增大c_A
        adj_A1 = train(lambda_A, c_A, sample_noise, sample_labels)
        print("Optimization Finished!(one train done)")

        # update parameters
        A_new = adj_A1.data.clone()  # 此时的增广后的A_new
        h_A_new = _h_A(A_new, 5)  # A_new的h(A) 要修改对应维度
        if h_A_new.item() > 0.25 * h_A_old:  # 如果h(A)比上一个epoch的0.25倍要大，就把c_A乘以10  # 第一次肯定break出去不执行这一步
            c_A*=10
        else:
            break

    h_A_old = h_A_new.item()  # 更新h(A)
    lambda_A += c_A * h_A_new.item()  # 更新lambda_A（+= c_A*h(A)）

    if h_A_new.item() <= 1e-8:  # h(a)已经属于0就不用找lambda和c了
        break