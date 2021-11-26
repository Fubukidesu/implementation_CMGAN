# 先得到celebA的Dtest(19904) 看看有多少条 以及形式
# 再生成相同数目的Dsample
# 最后调用mseM1来计算数值
from utils.data import CelebA
import torch.utils.data as data
import numpy as np
import torch
from torch.autograd.variable import Variable
from lgn_module_Azaixia import LGNGenerator_Azaixia
from utils.initialization import load_or_initialize
from utils.cuda import to_cuda_if_available
from utils.dag_gnn_utils import preprocess_adj_new1

def get_celebAtest():
    # 得到(19904, 9)的celebA测试数据
    data_path = 'data/celebA/img_align_celeba'
    attr_path = 'data/celebA/list_attr_celeba.txt'
    attrs_default = [
    'Bald', 'Eyeglasses', 'Male', 'Mustache', 'Mouth_Slightly_Open', 'Young', 'Wearing_Lipstick', 'Smiling', 'Narrow_Eyes'
    ]

    train_dataset = CelebA(data_path, attr_path, 32, 'test', attrs_default)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=64, num_workers=0,
        shuffle=True, drop_last=True
    )

    batch_num = -1
    more_batches = True
    it = iter(train_dataloader)
    Dtest = np.empty((0, 9), dtype=np.float32)
    while more_batches:
        # train discriminator
        try:
            (imgs, labels) = next(it)
            # labels = encode_conti_onehot(labels.numpy())
            batch = labels.numpy().astype(np.float32)
            batch_num += 1
            # print('labels', labels, labels.shape)  # tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0], torch.Size([64, 9])
            # print('batch', batch, batch.shape)  # [[0. 0. 0. 0. 1. 1. 1. 1. 0.] (64, 9)
            Dtest = np.append(Dtest, batch, axis=0)
            print('已加入第%d个batch的数据'%batch_num)
        except StopIteration:
            more_batches = False
            break
    print('Dtest：', Dtest, Dtest.shape) # [0. 0. 1. ... 0. 1. 0.]] (19904, 9)
    np.save('data/celebA_exp/Dtest.npy', Dtest)

# get_celebAtest()

def get_celebAsample():
    # 利用训练好的LGN生成(19904, 9)样本 作为Dsample
    adj_A = np.zeros((9, 9))
    lgngenerator = LGNGenerator_Azaixia(adj_A = adj_A)
    load_or_initialize(lgngenerator, 'output/exp_models/lgn_model_celebA9fromCAN/9_generator.pth')
    lgngenerator = to_cuda_if_available(lgngenerator)
    # print(lgngenerator)
    lgngenerator.load_state_dict(torch.load('output/exp_models/lgn_model_celebA9fromCAN/9_generator.pth'))
    noise = Variable(torch.FloatTensor(19904, 10).normal_())  # 选择生成多少条样本
    noise = to_cuda_if_available(noise)
    gen_features, adj_A1, output_10, Wa = lgngenerator(noise, training=True)
    # 要不要去阈值计算？
    adj_A1 = lgngenerator.state_dict()['adj_A']
    print('adj_A', adj_A1)
    # adj_A1[torch.abs(adj_A1) < 0.006] = 0
    # print('去阈值后的adj_A1', adj_A1)
    adj_A1 = torch.sinh(3.*adj_A1)
    # print('去阈值且增幅后的adj_A1', adj_A1)
    # adj_A1 = 100.*adj_A1
    # print('去阈值且大增幅后的adj_A1', adj_A1)
    # 开始额外矩阵计算
    adj_A_new1 = preprocess_adj_new1(adj_A1)  # adj_A_new1 = (I-A^T)^(-1)
    # 增加一个维度 [bs d] -> [bs d 1]
    output_10 = output_10.unsqueeze(2).double()
    final_output = torch.matmul(adj_A_new1, output_10 + Wa) - Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
    #维度变回来
    gen_features = final_output.squeeze(2).float()
    gen_features = gen_features.cpu().detach().numpy().astype(np.float32)
    # print(gen_features)
    gen_features = np.where(gen_features<0.5, 0., 1.).astype(np.float32)
    # print(gen_features, gen_features.shape, gen_features.dtype) [0. 0. 0. 0. 0. 0. 1. 0. 1.]] (10, 9) float32
    np.save('data/celebA_exp/Dsample_19904_noL_A.npy', gen_features)

    # np.savetxt('try_LGN/output/syn_data/Azaixia/originA_gen_sig_100_clear.txt', gen_features.cpu().detach().numpy(), fmt='%.2f', encoding='utf-8')

get_celebAsample()
    

# testb = np.array([[1, 1, 1], [1, 1, 1]])
# testb = testb.astype(np.float32)
# Dtest = np.empty((1, 3), dtype=np.float32)
# print(Dtest, Dtest.dtype)
# print(np.concatenate((Dtest, testb), axis=0))


# a=np.empty((0,3))
# b = np.array([[1,2,3],[4,5,6]])
# c=[[7,8,9]]
 
# print(a, a.shape, a.dtype)
# print(b, b.shape, b.dtype)
 
# a = np.append(a, b, axis=0)
# a = np.append(a, c, axis=0)
 
# print(a, a.shape, a.dtype)
# print(b, b.shape, b.dtype)

# a = np.array([[-0.5, 0.4], [0.5, 1.5]]).astype(np.float32)
# b = np.where(a < 0.5, 0, 1)
# print(b)
