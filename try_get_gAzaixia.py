import torch
import numpy as np
from lgn_module_Azaixia import LGNGenerator_Azaixia
from torch.autograd.variable import Variable
from utils.cuda import to_cuda_if_available
from utils.dag_gnn_utils import preprocess_adj_new1

random_seed = 123
torch.manual_seed(random_seed)

adj_A = np.zeros((5, 5))
attrs_default = [
   'No_Beard' ,'Mustache', 'Bald', 'Young', 'Male'
]

def get_A():
    # 得到A的数值
    generator = LGNGenerator_Azaixia(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/6_generator_Azaixia_sig.pth'))
    # for name in generator.state_dict():
    #     print(name)
    for name, item in generator.named_parameters():
        if name == 'adj_A':
            item = item.cpu().detach().numpy()
            print('adj_A', np.around(item, decimals=4))
#  [[ 0.     -0.0022  0.0083 -0.0001  0.0074]
#  [-0.002   0.      0.0004 -0.0022 -0.0134]
#  [-0.0006 -0.0007 -0.     -0.0056  0.0003]
#  [ 0.0172 -0.0029 -0.0005  0.     -0.0052]
#  [ 0.0003 -0.0001  0.012   0.0003 -0.    ]] 7_generator_Azaixia.pth
# [[-0.     -0.0028  0.0037 -0.0002 -0.    ]
#  [ 0.0001 -0.      0.0001 -0.0005 -0.0124]
#  [ 0.0001 -0.0041  0.     -0.0006  0.0038]
#  [ 0.001  -0.0012 -0.0018 -0.      0.0001]
#  [-0.0083 -0.0001 -0.0002  0.0084 -0.    ]] 6_generator_Azaixia_sig.pth
# get_A()

def modi_A():
     # 设置为0的阈值，方便画图
    threshold = 0.004
    generator = LGNGenerator_Azaixia(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/6_generator_Azaixia_sig.pth'))
    for name, item in generator.named_parameters():
        if name == 'adj_A':
            # print(item)
            A = item.cpu().detach().numpy()
    # print(A)
    B = A
    B[np.abs(B) < threshold] = 0
    print(threshold)
    print(B)
# modi_A()

def originA_gen():
    #得到原始的adj_A(产生数据的过程增幅了)下产生的数据
    generator = LGNGenerator_Azaixia(adj_A).cuda()
    generator.load_state_dict(torch.load('try_LGN/output/models/lgn_models/6_generator_Azaixia_sig.pth'))
    
    noise = Variable(torch.FloatTensor(100, 10).normal_())
    noise = to_cuda_if_available(noise)
    gen_features, adj_A1, output_10, Wa = generator(noise, training=True)
    # print('不去阈值增幅后的adj_A1', adj_A1)
    # tensor([[ 2.8484e-06, -6.6638e-03,  2.4778e-02, -2.1283e-04,  2.2261e-02],
    #     [-5.9605e-03,  4.6124e-07,  1.3329e-03, -6.5927e-03, -4.0254e-02],
    #     [-1.8879e-03, -2.1917e-03, -1.0186e-08, -1.6871e-02,  9.8516e-04],
    #     [ 5.1497e-02, -8.7291e-03, -1.4700e-03,  1.3276e-06, -1.5580e-02],
    #     [ 1.0449e-03, -3.0092e-04,  3.5911e-02,  8.3354e-04, -3.1373e-07]],
    #    device='cuda:0', dtype=torch.float64, grad_fn=<SinhBackward>)
    # print('fake_features', gen_features)
    print('output10', output_10)

    # 要不要去阈值计算？
    adj_A1 = generator.state_dict()['adj_A']
    print('adj_A', adj_A1)
    adj_A1[torch.abs(adj_A1) < 0.004] = 0
    print('去阈值后的adj_A1', adj_A1)
    adj_A1 = torch.sinh(3.*adj_A1)
    print('去阈值且增幅后的adj_A1', adj_A1)
    # adj_A1 = 100.*adj_A1
    # print('去阈值且大增幅后的adj_A1', adj_A1)
    # 开始额外矩阵计算
    adj_A_new1 = preprocess_adj_new1(adj_A1)  # adj_A_new1 = (I-A^T)^(-1)
    # 增加一个维度 [bs d] -> [bs d 1]
    output_10 = output_10.unsqueeze(2).double()
    final_output = torch.matmul(adj_A_new1, output_10 + Wa) - Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
    #维度变回来
    gen_features = final_output.squeeze(2).float()

    np.savetxt('try_LGN/output/syn_data/Azaixia/originA_gen_sig_100_clear.txt', gen_features.cpu().detach().numpy(), fmt='%.2f', encoding='utf-8')
# originA_gen()

def changeA_gen(value):
    # 改变adj_A
    generator = LGNGenerator_Azaixia(adj_A).cuda()
    generator.load_state_dict(torch.load('try_LGN/output/models/lgn_models/6_generator_Azaixia_sig.pth'))

    # print(generator.state_dict()['adj_A'])
    # generator.state_dict()['adj_A'] = torch.sinh(3.*generator.state_dict()['adj_A'])
    generator.state_dict()['adj_A'][:, 1:2] = 0.  # 干预第一特征，消除所有指向第一特征的边
    # generator.state_dict()['adj_A'][:, 4:5] = 0.  # 干预第五特征，消除所有指向第五特征的边
    adj_A1 = generator.state_dict()['adj_A']
    print('只修改了列后的adj_A', adj_A1)

    noise = Variable(torch.FloatTensor(100, 10).normal_())
    # noise = torch.full((10,10), 0.1)
    # noise = Variable(torch.FloatTensor(noise))
    noise = to_cuda_if_available(noise)
    
    adj_A1[torch.abs(adj_A1) < 0.004] = 0
    print('修改列、去阈值后的adj_A1', adj_A1)
    gen_features, adj_A_, output_10, Wa = generator(noise, training=True)  # 得到矩阵前结果output_10，以及增幅修改后的矩阵
    # adj_A1 = torch.sinh(3.*adj_A1)
    # print('修改列、去阈值且增幅后的adj_A1', adj_A1)
    adj_A1 = 100.*adj_A1
    print('修改列、去阈值且大增幅后的adj_A1', adj_A1)
    # # 去不去阈值？
    # adj_A1[torch.abs(adj_A1) < 0.003] = 0
    # print('再阈值后的adj_A1', adj_A1)
    output_10[:, 1:2] = value  # 干预第一特征，设定第一特征某个初值
    print('output_10', output_10)
    # print('WA', Wa)

    # 开始额外矩阵计算
    adj_A_new1 = preprocess_adj_new1(adj_A1)  # adj_A_new1 = (I-A^T)^(-1)
    # 增加一个维度 [bs d] -> [bs d 1]
    output_10 = output_10.unsqueeze(2).double()
    final_output = torch.matmul(adj_A_new1, output_10 + Wa) - Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
    #维度变回来
    final_output = final_output.squeeze(2).float()

    # print('final_output', final_output, final_output.shape)  # device='cuda:0', grad_fn=<CopyBackwards>) torch.Size([100, 5])

    np.savetxt('try_LGN/output/syn_data/Azaixia/changeA_gen_sig_inter2_1_clear_amp.txt', final_output.cpu().detach().numpy(), fmt='%.2f', encoding='utf-8')
# changeA_gen(1.)

def get_somecol():
    with open('try_LGN/output/syn_data/Azaixia/changeA_gen_sig_inter2_1_clear_amp.txt', 'r') as f:
        list1 = []
        list2 = []
        num1, num2 = 0, 0
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            a = line.split()
            b = a[2:3]  # 这是选取需要读取的位数 0是No_beard，第一特征，第四特征指向了第一特征 origin:80 change:0
            c = a[4:5]  # 这是选取需要读取的位数 3是Young，第四特征 origin:81 change:78
            list1.append(b) # 将其添加在列表
            list2.append(c)
        # print(np.array(list1))
        for row in list1:
            # print('count1:', row.count('1'))
            row = list(map(float,row))  # 把list内的元素变成float型
            if row[0] > 0.5:
                num1 += 1
        for row in list2:
            row = list(map(float,row))  # 把list内的元素变成float型
            if row[0] > 0.5:
                num2 += 1
        print('total1:%d, total2:%d'%(num1, num2))
        return num1, num2          
# get_somecol()

