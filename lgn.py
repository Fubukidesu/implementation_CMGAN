# 原始的wgan-gp multi_label
import torch
import numpy as np
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
from utils.wgan_gp import calculate_gradient_penalty
from check_torchshape import encode_conti_onehot
from lgn_module import LGNGenerator, LGNDiscriminator

# features = loaders['sparse']('data/synthetic/fixed_2/synthetic.features.npz')
# # print('features:', features, features.shape)  # (10000, 20)
# data = Dataset(features)
# train_data, val_data = data.split(1.0 - .1)
# # variable_sizes = load_variable_sizes_from_metadata('data/synthetic/fixed_2/metadata.json')  # [2, 2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2]

train_data, val_data = 1, 1
#celebA数据
data_path = 'data/celebA/img_align_celeba'
attr_path = 'data/celebA/list_attr_celeba.txt'
attrs_default = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows' ,'Attractive' ,'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair'
]

train_dataset = CelebA(data_path, attr_path, 32, 'train', attrs_default)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=64, num_workers=0,
    shuffle=True, drop_last=True
)

lgngenerator = LGNGenerator()
load_or_initialize(lgngenerator, None)

lgndiscriminator = LGNDiscriminator()
load_or_initialize(lgndiscriminator, None)

# lgngenerator = to_cuda_if_available(lgngenerator)
# lgndiscriminator = to_cuda_if_available(lgndiscriminator)

# print(lgndiscriminator)

def train(generator,
          discriminator,
          train_data,
          val_data,
          output_gen_path,
          output_disc_path,
          output_loss_path,
          batch_size=64,
          start_epoch=0,
          num_epochs=100,
          num_disc_steps=2,
          num_gen_steps=1,
          noise_size=10,
          l2_regularization=0,
          learning_rate=0.001,
          penalty=10
          ):
    generator, discriminator = to_cuda_if_available(generator, discriminator)

    optim_gen = Adam(generator.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    logger = Logger(output_loss_path, append=start_epoch > 0)

    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # train
        generator.train(mode=True)
        discriminator.train(mode=True)

        disc_losses = []
        gen_losses = []

        batch_num = -1
        more_batches = True
        # train_data_iterator = train_data.batch_iterator(batch_size)
        it = iter(train_dataloader)

        while more_batches:
            # train discriminator
            for _ in range(num_disc_steps):
                # next batch
                try:
                    (imgs, labels) = next(it)
                    labels = encode_conti_onehot(labels.numpy())
                    batch = labels
                    batch_num += 1
                    # print('batch', labels, labels.shape)  # 自己的celebA数据
                    # print('batch', batch, batch.shape) # noise为10时：(64,20)
                except StopIteration:
                    more_batches = False
                    break
                
                print('Epoch:%d.batch:%d,用于D'%(epoch_index, batch_num))

                optim_disc.zero_grad()

                # first train the discriminator only with real data
                real_features = Variable(torch.from_numpy(batch))
                real_features = to_cuda_if_available(real_features)
                real_pred = discriminator(real_features)  # torch.Size([64])
                real_loss = - real_pred.mean(0).view(1)
                real_loss.backward()

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_features = generator(noise, training=True)
                fake_features = fake_features.detach()  # do not propagate to the generator
                fake_pred = discriminator(fake_features)
                fake_loss = fake_pred.mean(0).view(1)
                fake_loss.backward()

                # this is the magic from WGAN-GP
                gradient_penalty = calculate_gradient_penalty(discriminator, penalty, real_features, fake_features)
                gradient_penalty.backward()

                # finally update the discriminator weights
                # using two separated batches is another trick to improve GAN training
                optim_disc.step()

                disc_loss = real_loss + fake_loss + gradient_penalty
                disc_loss = to_cpu_if_available(disc_loss)
                disc_losses.append(disc_loss.data.numpy())

                del disc_loss
                del gradient_penalty
                del fake_loss
                del real_loss

            # train generator
            for _ in range(num_gen_steps):
                print('Epoch:%d.batch:%d,用于G'%(epoch_index, batch_num))
                optim_gen.zero_grad()

                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                gen_features = generator(noise, training=True)
                fake_pred = discriminator(gen_features)
                fake_loss = - fake_pred.mean(0).view(1)
                fake_loss.backward()

                optim_gen.step()

                fake_loss = to_cpu_if_available(fake_loss)
                gen_losses.append(fake_loss.data.numpy())

                del fake_loss

        # log epoch metrics for current class
        logger.log(epoch_index, num_epochs, "discriminator", "train_mean_loss", np.mean(disc_losses))
        logger.log(epoch_index, num_epochs, "generator", "train_mean_loss", np.mean(gen_losses))

        # save models for the epoch
        # with DelayedKeyboardInterrupt():
        torch.save(generator.state_dict(), output_gen_path)
        torch.save(discriminator.state_dict(), output_disc_path)
        logger.flush()

    logger.close()

train(lgngenerator, lgndiscriminator, train_data, val_data, 'output/models/generator.pth', 'output/models/discriminator.pth', 'output/models/loss.csv')