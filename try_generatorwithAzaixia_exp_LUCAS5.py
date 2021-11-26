# lgn网络，其中G中矩阵在下部
# 为了做对比实验，对celebA的9个特征进行
# 'Bald', 'Eyeglasses', 'Male', 'Mustache', 'Mouth_Slightly_Open', 'Young', 'Wearing_Lipstick', 'Smiling', 'Narrow_Eyes'
import torch
import numpy as np
import argparse
from utils.data import CelebA
from torch.autograd.variable import Variable
from torch.optim import Adam
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
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
from lgn_module_Azaixia import LGNGenerator_Azaixia_5, LGNDiscriminator_Azaixia_5
from utils.get_LUCAS import LUCAS


# parser = argparse.ArgumentParser()
# parser.add_argument('--lambda_A',  type = torch.cuda.DoubleTensor, default= 0.,
#                     help='coefficient for DAG constraint h(A).')
# parser.add_argument('--c_A',  type = torch.cuda.DoubleTensor, default= 1,
#                     help='coefficient for absolute value h(A).')
# args = parser.parse_args()

# features = loaders['sparse']('data/synthetic/fixed_2/synthetic.features.npz')
# # print('features:', features, features.shape)  # (10000, 20)
# data = Dataset(features)
# train_data, val_data = data.split(1.0 - .1)
# # variable_sizes = load_variable_sizes_from_metadata('data/synthetic/fixed_2/metadata.json')  # [2, 2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2]

train_data, val_data = 1, 1
#celebA数据
# data_path = 'data/celebA/img_align_celeba'
# attr_path = 'data/celebA/list_attr_celeba.txt'
# attrs_default = [
#     '5_o_Clock_Shadow', 'Arched_Eyebrows' ,'No_Beard' ,'Mustache', 'Bald', 'Pale_Skin', 'Young', 'Male', 'Black_Hair', 'Blond_Hair'
# ]
# attrs_default = [
#    'Bald', 'Eyeglasses', 'Male', 'Mustache', 'Mouth_Slightly_Open', 'Young', 'Wearing_Lipstick', 'Smiling', 'Narrow_Eyes'
# ]

dataset = LUCAS(csv_file='data/LUCAS/lucas0_train_5.csv')    
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# 加矩阵
adj_A = np.zeros((5, 5))
lgngenerator = LGNGenerator_Azaixia_5(adj_A = adj_A)  # 改输出个数
load_or_initialize(lgngenerator, None)

lgndiscriminator = LGNDiscriminator_Azaixia_5()  # 这里面要对应改D的输入维度
load_or_initialize(lgndiscriminator, None)

# lgngenerator = to_cuda_if_available(lgngenerator)
# lgndiscriminator = to_cuda_if_available(lgndiscriminator)

# print(lgndiscriminator)

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)  # (I+(A*A)/m)^m
    h_A = torch.trace(expm_A) - m  # tr[(I+(A*A)/m)^m] - m，即h_a
    return h_A

def train(generator,
          discriminator,
          train_data,
          val_data,
          output_gen_path,
          output_disc_path,
          output_loss_path,
          lambda_A,
          c_A,
          batch_size=64,
          start_epoch=0,
          num_epochs=20,
          num_disc_steps=2,
          num_gen_steps=1,
          noise_size=10,
          l2_regularization=0,
          learning_rate=0.001,
          penalty=10
          ):
    lambda_A = lambda_A.cuda()
    c_A = c_A.cuda()
    
    # print('接下来准备cuda模型')
    generator, discriminator = to_cuda_if_available(generator, discriminator)
    # print(generator)
    # for name in generator.state_dict():
    #     print(name)
    for name, item in generator.named_parameters():
        # print('name is:', name)
        # print('parameter is:', item)
        if name == 'adj_A':
            print(item)

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
        it = iter(dataloader)

        while more_batches:
            # train discriminator
            for _ in range(num_disc_steps):
                # next batch
                try:
                    labels = next(it)
                    # labels = encode_conti_onehot(labels.numpy())
                    batch = labels.numpy().astype(np.float32)
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
                fake_features, adj_A1, output_10, Wa = generator(noise, training=True)
                fake_features = fake_features.detach()  # do not propagate to the generator
                
                # # 对此时G产生的10维标签数据，转换成20维，再送给D
                # fake_features = encode_conti_onehot(fake_features.numpy())
                # fake_features = to_cuda_if_available(Variable(torch.from_numpy(fake_features)))

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
                print('disc_loss:', disc_loss.item())
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
                gen_features, adj_A1, output_10, Wa = generator(noise, training=True)
                fake_pred = discriminator(gen_features)
                fake_loss = - fake_pred.mean(0).view(1)
                

                # compute h(A)
                # adj_A1 = adj_A1.double()
                h_A = _h_A(adj_A1, 5).type(torch.cuda.FloatTensor)  # 修改对应特征维度
                # print(h_A)
                lambda_A = lambda_A.type(torch.cuda.FloatTensor)
                # print(lambda_A)
                c_A = c_A.type(torch.cuda.FloatTensor)
                # print(c_A)
                # erfenyi = torch.from_numpy(np.array([0.5], dtype=np.float64))
                # yibai = torch.from_numpy(np.array([100.], dtype=np.float64))
                # fake_loss += lambda_A * h_A + erfenyi * c_A * h_A * h_A + yibai * torch.trace(adj_A1*adj_A1).double()
                fake_loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(adj_A1*adj_A1).type(torch.cuda.FloatTensor)
                print('gen_loss:', fake_loss.item())

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
    return adj_A1
lambda_A = [0.]
lambda_A = torch.from_numpy(np.array(lambda_A))
c_A = [1.]
c_A = torch.from_numpy(np.array(c_A))
h_A_old = np.inf
for step_k in range(20):  # 为了找lambda和c而进行迭代
    while c_A < 1e+20:  # 当c_a小于1e+20时不断增大c_A
        # adj_A1 = train(lgngenerator, lgndiscriminator, train_data, val_data, 'output/models/%s_generator_Azaixia_sig.pth'%step_k, 'output/models/%s_discriminator_Azaixia_sig.pth'%step_k, 'output/models/%s_loss1_sig.csv'%step_k, lambda_A, c_A)
        adj_A1 = train(lgngenerator, lgndiscriminator, train_data, val_data, 'output/exp_models/lgn_model_LUCAS5/%s_generator.pth'%step_k, 'output/exp_models/lgn_model_LUCAS5/%s_discriminator.pth'%step_k, 'output/exp_models/lgn_model_LUCAS5/%s_loss.csv'%step_k, lambda_A, c_A)
        print("Optimization Finished!")

        # update parameters
        A_new = adj_A1.data.clone()  # 此时的增广后的A_new
        h_A_new = _h_A(A_new, 5)  # A_new的h(A) 要修改对应维度
        if h_A_new.item() > 0.25 * h_A_old:  # 如果h(A)比上一个epoch的0.25倍要大，就把c_A乘以10  # 第一次肯定break出去不执行这一步
            c_A*=10
        else:
            break

        # update parameters
        # h_A, adj_A are computed in loss anyway, so no need to store
    h_A_old = h_A_new.item()  # 更新h(A)
    lambda_A += c_A * h_A_new.item()  # 更新lambda_A（+= c_A*h(A)）

    if h_A_new.item() <= 1e-8:  # h(a)已经属于0就不用找lambda和c了
        break