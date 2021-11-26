# 试着实现cign中的acgan wgan-gp
import os
from PIL import Image
import torch
from torchvision.utils import make_grid
import numpy as np
import argparse
from utils.data import CelebA
from torch.autograd.variable import Variable
from torch.optim import Adam
import torch.utils.data as data
from formats import data_formats, loaders
from datasets import Dataset
from utils.categorical import load_variable_sizes_from_metadata
from utils.initialization import load_or_initialize
from utils.cuda import to_cuda_if_available, to_cpu_if_available
from utils.logger import Logger
from utils.wgan_gp import calculate_gradient_penalty_forACGAN
from utils.dag_gnn_utils import matrix_poly
from check_torchshape import encode_conti_onehot
from CIGN_module import CIGNdiscriminator, CIGNGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_data, val_data = 1, 1

#celebA数据
data_path = 'data/celebA/img_align_celeba'
attr_path = 'data/celebA/list_attr_celeba.txt'
attrs_default = [
   'No_Beard' ,'Mustache', 'Bald', 'Young', 'Male'
]

train_dataset = CelebA(data_path, attr_path, 64, 'train', attrs_default)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=64, num_workers=0,
    shuffle=True, drop_last=True
)

cigngenerator = CIGNGenerator()
load_or_initialize(cigngenerator, None)

cigndiscriminator = CIGNdiscriminator()
load_or_initialize(cigndiscriminator, None)

def train(generator,
          discriminator,
          train_data,
          val_data,
          output_gen_path,
          output_disc_path,
          output_loss_path,
          lambda_A=1,
          c_A=1,
          batch_size=64,
          start_epoch=0,
          num_epochs=100,
          num_disc_steps=2,
          num_gen_steps=1,
          noise_size=128,
          l2_regularization=0,
          learning_rate=0.001,
          penalty=10
          ):
    
    # print('接下来准备cuda模型')
    generator, discriminator = to_cuda_if_available(generator, discriminator)
    # g和d的优化器Adam
    optim_gen = Adam(generator.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    # 用来记录Loss的Logger
    logger = Logger(os.path.join(output_loss_path, 'loss.csv'), append=start_epoch > 0)


    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # train
        generator.train(mode=True)
        discriminator.train(mode=True)

        disc_losses = []
        gen_losses = []

        batch_num = -1
        more_batches = True

        it = iter(train_dataloader)

        while more_batches:
            # train discriminator
            for _ in range(num_disc_steps):
                # next batch
                try:
                    (imgs, labels) = next(it)
                    # print('img', imgs, imgs.shape)  # torch.Size([64, 3, 64, 64])
                    # print('labels', labels, labels.shape)  # torch.Size([64, 5])
                    batch_num += 1
                except StopIteration:
                    more_batches = False
                    break
                
                print('Epoch:%d.batch:%d,用于D'%(epoch_index, batch_num))

                optim_disc.zero_grad()

                # first train the discriminator only with real data
                real_features = Variable(imgs)
                real_features = to_cuda_if_available(real_features)
                # real_pred = discriminator(real_features)  # torch.Size([64])
                D_r_v, D_r_c = discriminator(real_features)
                # print('D_r_c的数据类型',D_r_c.type()) # torch.cuda.FloatTensor
                D_r_v_loss = - D_r_v.mean(0).view(1)
                D_r_v_loss.backward(retain_graph=True)
                # D对真数据的分类损失
                labels = to_cuda_if_available(labels.type(torch.FloatTensor))
                labels = Variable(labels)
                # print('labels数据类型', labels.type()) # 不指定的话就是torch.cuda.LongTensor
                D_r_c_loss = torch.nn.functional.binary_cross_entropy_with_logits(D_r_c, labels)
                D_r_c_loss.backward()

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(imgs.shape[0], noise_size).normal_())
                noise = to_cuda_if_available(noise)
                # 准备输入G的标签
                fake_features = generator(labels, noise)
                fake_features = fake_features.detach()  # do not propagate to the generator
                # fake_pred = discriminator(fake_features)
                D_f_v, D_f_c = discriminator(fake_features)
                D_f_v_loss = D_f_v.mean(0).view(1)
                D_f_v_loss.backward(retain_graph=True)
                #D对假数据的分类损失
                D_f_c_loss = torch.nn.functional.binary_cross_entropy_with_logits(D_f_c, labels)
                D_f_c_loss.backward()

                # this is the magic from WGAN-GP
                gradient_penalty = calculate_gradient_penalty_forACGAN(discriminator, penalty, real_features, fake_features)
                gradient_penalty.backward()

                # finally update the discriminator weights
                # using two separated batches is another trick to improve GAN training
                optim_disc.step()

                disc_loss = D_r_v_loss + D_r_c_loss + D_f_v_loss + D_f_c_loss + gradient_penalty
                print('Disc_loss:{:.4f}, D_r_v_loss:{:.4f}, D_r_c_loss:{:.4f}, D_f_v_loss:{:.4f}, D_f_c_loss:{:.4f}, gradient_penalty:{:.4f}'.format(
                    disc_loss.item(), D_r_v_loss.item(), D_r_c_loss.item(), D_f_v_loss.item(), D_f_c_loss.item(), gradient_penalty.item()))
                disc_loss = to_cpu_if_available(disc_loss)
                disc_losses.append(disc_loss.data.numpy())

                del disc_loss
                del gradient_penalty
                del D_r_v_loss
                del D_r_c_loss
                del D_f_v_loss
                del D_f_c_loss

                #采样图片看看G的效果 'No_Beard' ,'Mustache', 'Bald', 'Young', 'Male'
                if batch_num % 400 == 0:
                    sample_labels = np.array([[1, 0, 1, 1, 1], [1, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 1, 1]])
                    sample_labels = Variable(to_cuda_if_available(torch.FloatTensor(sample_labels)))
                    sample_noise = Variable(torch.FloatTensor(4, 128).normal_())
                    sample_noise = to_cuda_if_available(sample_noise)
                    sample_imgs = generator(sample_labels, sample_noise)
                    grid = make_grid(sample_imgs.data, nrow=2, padding=2, normalize=True)
                    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    im = Image.fromarray(ndarr)
                    im.save('output/cign_imgs/E:{:d}_B:{:d}.png'.format(epoch_index, batch_num))  
            
            # train generator
            for _ in range(num_gen_steps):
                print('Epoch:%d.batch:%d,用于G'%(epoch_index, batch_num))
                optim_gen.zero_grad()

                noise = Variable(torch.FloatTensor(labels.shape[0], noise_size).normal_())
                noise = to_cuda_if_available(noise)
                gen_features = generator(labels, noise)
                D_f_v, D_f_c = discriminator(gen_features)
                D_f_v_loss = - D_f_v.mean(0).view(1)
                D_f_v_loss.backward(retain_graph=True)
                #D对假数据的分类损失
                D_f_c_loss = torch.nn.functional.binary_cross_entropy_with_logits(D_f_c, labels)
                D_f_c_loss.backward()

                optim_gen.step()

                fake_loss = D_f_v_loss + D_f_c_loss
                print('Gen_loss:{:.4f}, D_f_v_loss:{:.4f}, D_f_c_loss:{:.4f}'.format(fake_loss.item(), D_f_v_loss.item(), D_f_c_loss.item()))
                fake_loss = to_cpu_if_available(fake_loss)
                gen_losses.append(fake_loss.data.numpy())

                del fake_loss
                del D_f_v_loss
                del D_f_c_loss

        # log epoch metrics for current class
        logger.log(epoch_index, num_epochs, "discriminator", "train_mean_loss", np.mean(disc_losses))
        logger.log(epoch_index, num_epochs, "generator", "train_mean_loss", np.mean(gen_losses))     

        # save models for the epoch
        # with DelayedKeyboardInterrupt():
        torch.save(generator.state_dict(), os.path.join(output_gen_path, 'epoch:{:d}_generator.pth'.format(epoch_index)))
        torch.save(discriminator.state_dict(), os.path.join(output_disc_path, 'epoch:{:d}_discriminator.pth'.format(epoch_index)))
        logger.flush()
    
    logger.close()

train(cigngenerator, cigndiscriminator, train_data, val_data, 'output/models/cign_models', 'output/models/cign_models', 'output/models/cign_models')