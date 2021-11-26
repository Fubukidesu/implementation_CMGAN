import torch
import numpy as np
import torch.nn.functional as F
from lgn_module import LGNGenerator
from lgn_module_Azaixia import LGNGenerator_Azaixia
from torch.autograd.variable import Variable
from utils.cuda import to_cuda_if_available, to_cpu_if_available
from utils.dag_gnn_utils import preprocess_adj_new1
from multi_categorical import MultiCategorical

adj_A = np.zeros((10, 10))

def get_A():
    generator = LGNGenerator(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/8_generator1.pth'))
    # print(generator)
    # print('.......')
    # for name in generator.state_dict():
    #     print(name)
# adj_A
# Wa
# hidden_layers.0.weight
# hidden_layers.0.bias
# hidden_layers.2.weight
# hidden_layers.2.bias
# hidden_layers.3.weight
# hidden_layers.3.bias
# hidden_layers.3.running_mean
# hidden_layers.3.running_var
# hidden_layers.3.num_batches_tracked
# hidden_layers.5.weight
# hidden_layers.5.bias
# hidden_layers.6.weight
# hidden_layers.6.bias
# hidden_layers.6.running_mean
# hidden_layers.6.running_var
# hidden_layers.6.num_batches_tracked
# hidden_layers.8.weight
# hidden_layers.8.bias
# hidden_layers.9.weight
# hidden_layers.9.bias
# hidden_layers.9.running_mean
# hidden_layers.9.running_var
# hidden_layers.9.num_batches_tracked
# output.output_layers.0.weight
# output.output_layers.0.bias
# output.output_layers.1.weight
# output.output_layers.1.bias
# output.output_layers.2.weight
# output.output_layers.2.bias
# output.output_layers.3.weight
# output.output_layers.3.bias
# output.output_layers.4.weight
# output.output_layers.4.bias
# output.output_layers.5.weight
# output.output_layers.5.bias
# output.output_layers.6.weight
# output.output_layers.6.bias
# output.output_layers.7.weight
# output.output_layers.7.bias
# output.output_layers.8.weight
# output.output_layers.8.bias
# output.output_layers.9.weight
# output.output_layers.9.bias
    for name, item in generator.named_parameters():
        if name == 'adj_A':
            # print(item)
            item = item.cpu().detach().numpy()
            print('adj_A', np.around(item, decimals=4))
# [[ 0.      0.0001  0.0088  0.0195 -0.0017  0.0063 -0.0029  0.007  -0.0006
#   -0.0002]
#  [-0.0018 -0.     -0.0003  0.012  -0.0019  0.0005  0.0004  0.0007  0.0016
#    0.0039]
#  [ 0.0001  0.0023  0.      0.0014  0.0007 -0.0053 -0.0002  0.0026 -0.0023
#   -0.    ]
#  [-0.      0.0001  0.0029 -0.      0.009  -0.0002  0.0065  0.0003 -0.0003
#    0.0014]
#  [-0.0002  0.0001 -0.0009 -0.0002  0.      0.      0.0048  0.0001 -0.0043
#    0.    ]
#  [-0.0002 -0.0011  0.      0.0005 -0.0121 -0.      0.      0.     -0.0002
#    0.    ]
#  [-0.0001  0.0028 -0.0059  0.      0.0002  0.001   0.     -0.0092 -0.0052
#   -0.0075]
#  [-0.0001  0.0002 -0.0006 -0.0018  0.0234 -0.0067  0.     -0.     -0.0001
#   -0.0001]
#  [ 0.0021  0.0038 -0.      0.003  -0.0004  0.0138  0.0001 -0.0052  0.
#    0.0002]
#  [ 0.0041 -0.0002 -0.0271  0.0012 -0.0367  0.005  -0.      0.002   0.0021
#    0.    ]]

# get_A()

def modi_A():
    # 设置为0的阈值，方便画图
    threshold = 0.005

    generator = LGNGenerator(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/8_generator1.pth'))
    for name, item in generator.named_parameters():
        if name == 'adj_A':
            # print(item)
            A = item.cpu().detach().numpy()
    # print(A)
    B = A
    B[np.abs(B) < threshold] = 0
    print(B)

# modi_A()

def originA_gen():
    #得到原始的adj_A下产生的数据
    generator = LGNGenerator(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/8_generator1.pth'))
    
    noise = Variable(torch.FloatTensor(100, 10).normal_())
    noise = to_cuda_if_available(noise)
    gen_features, adj_A1 = generator(noise, training=True)
    print('增幅后的adj_A1', adj_A1)
    print('fake_features', gen_features)

    np.savetxt('output/syn_data/originA_gen.txt', gen_features.cpu().detach().numpy(), fmt='%.f', encoding='utf-8')


# originA_gen()

def changeA_gen(value):
    # 改变adj_A
    generator = LGNGenerator(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/8_generator1.pth'))
    # for name, item in generator.named_parameters():
    #     if name == 'adj_A':
    #         print('原先的adj_A', item)
    #         #对某一特征干预，就是去掉指向该特征的边，也就是消除该列。
    #         item[:, 4:5] = 0.
    #         print('干预后的adj_A', item)
    # print('修改前的adj_A：', generator.state_dict()['adj_A'])
    generator.state_dict()['adj_A'] = torch.sinh(3.*generator.state_dict()['adj_A'])
    generator.state_dict()['adj_A'][:, 4:5] = 0.  # 干预第五特征，消除所有指向第五特征的边
    # generator.state_dict()['adj_A'][:, 7:8] = 0.  # 干预第八特征，消除所有指向第八特征的边
    # print('修改后的adj_A', generator.state_dict()['adj_A'])

    # noise = Variable(torch.FloatTensor(100, 10).normal_())
    noise = torch.full((10,10), 0.1)
    noise = Variable(torch.FloatTensor(noise))
    noise = to_cuda_if_available(noise)
    noise[5:, 4:5] = value  # 干预第五特征，设定第五特征某个初值
    # noise[:, 7:8] = value  # 干预第八特征，设定第八特征某个初值
    gen_features, adj_A1 = generator(noise, training=True)
    # print('增幅后的adj_A1', adj_A1)
    # print('fake_features', gen_features, gen_features.shape)  # torch.Size([100, 20])

    np.savetxt('output/syn_data/changeA_gen.txt', gen_features.cpu().detach().numpy(), fmt='%.f', encoding='utf-8')

# changeA_gen(0.5)

def testz():
    noise = Variable(torch.FloatTensor(100, 10).normal_())
    noise = to_cuda_if_available(noise)
    print(noise)
    noise[:, 4:5] = 1.
    print(noise)

# testz()

def get_somecol():
    with open('output/syn_data/changeA_gen.txt', 'r') as f:
        list1 = []
        list2 = []
        num1, num2 = 0, 0
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            a = line.split()
            b = a[15:16]  # 这是选取需要读取的位数 15是Male，第八特征，第八特征指向了第五特征 origin:43 change:43
            c = a[9:10]  # 这是选取需要读取的位数 9是Bald，第五特征 origin:5 change:0
            list1.append(b) # 将其添加在列表
            list2.append(c)
        # print(np.array(list1))
        for row in list1:
            # print('count1:', row.count('1'))
            num1 += row.count('1')
        for row in list2:
            num2 += row.count('1')
        print('total1:%d, total2:%d'%(num1, num2))
        return num1, num2
            
# get_somecol()

def zhaovalue():
    max1, max2 = 0, 0
    for value in np.linspace(-1., 1., 100):
        changeA_gen(value)
        num1, num2 = get_somecol()
        if num1 > max1:
            max1 = num1
        if num2 > max2:
            max2 = num2
    print('max1:%d, manx2:%d'%(max1, max2))
# zhaovalue()

def try_Azaixia():
    # 尝试将A放在下层
    generator = LGNGenerator_Azaixia(adj_A).cuda()
    noise = Variable(torch.FloatTensor(5, 10).normal_())
    noise = to_cuda_if_available(noise)
    gen_features, adj_A1 = generator(noise, training=True)
    print('gen_features', gen_features)

try_Azaixia()

def matrix_multi():
    # 矩阵乘法原理
    # noise = Variable(torch.FloatTensor(1, 5).normal_())
    noise = torch.tensor([[0.1, 0.1, 0.5, 0.1, 0.1]])
    print('noise:', noise)
    # 把noise增加一个维度 [bs d] -> [bs d 1]
    noise = noise.unsqueeze(2)

    # adj_A = torch.tensor([[0., 0., 0., 0., 6.], [9., 0., 3., 0., 0.], [2., 0., 0., 5., 0.], [0., 0., 0., 0., 1.], [0., 0., 0., 0., 0.,]]).double().cuda()
    # 1 2 0 3 4 拓扑排序
    adj_A = torch.tensor([[0., 3., 9., 0., 0.], [0., 0., 2., 5., 0.], [0., 0., 0., 0., 6.], [0., 0., 0., 0., 1.], [0., 0., 0., 0., 0.,]]).double().cuda()
    # print('adj_A', adj_A)
    # print('经过干预<第0特征>矩阵计算')
    adj_A_new1 = preprocess_adj_new1(adj_A)
    adj_A_new1 = adj_A_new1.cpu().float()
    # print('adj_A_new1:', adj_A_new1)
    # Wa = 0.1
    # Wa = torch.tensor(0.1)
    Wa = torch.tensor([0.])
    # print('Wa_origin', Wa_origin)
    # print('Wa', Wa)
    # print('noise + Wa', noise + Wa)
    # print('noise + Wa_noise', noise + Wa_origin)

    mat_z = torch.matmul(adj_A_new1, noise + Wa) - Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
    #维度变回来
    noise = mat_z.squeeze(2).float()
    print('result:', noise)
# matrix_multi()

def matrix_multi_8_generator():
    # 矩阵乘法原理 具体实践
    threshold = 0.005
    noise = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    noise = noise.unsqueeze(2)
    Wa = torch.tensor([0.])
    print('noise:', noise)
    generator = LGNGenerator(adj_A).cuda()
    generator.load_state_dict(torch.load('output/models/8_generator1.pth'))
    adj_A_new1 = np.zeros((10, 10))
    for name, item in generator.named_parameters():
        if name == 'adj_A':
            # print(item)
            # A = item.cpu().detach().numpy()
            # print('原始矩阵:')  #全0.1时： [[0.1004, 0.1008, 0.0977, 0.1036, 0.0981, 0.1014, 0.1009, 0.0998, 0.0991, 0.0998]]

            item[:, 4:5] = 0.  # 干预第五特征，消除所有指向第五特征的边
            print('干预后矩阵')
            # print(item)

            adj_A_new1 = preprocess_adj_new1(item)
            adj_A_new1 = adj_A_new1.cpu().float()
    # print(adj_A_new1)
    # print('不干预')
    mat_z = torch.matmul(adj_A_new1, noise + Wa) - Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
    noise = mat_z.squeeze(2).float()
    print('result:', noise)
# matrix_multi_8_generator()

def test_softmax():
    # 测试softmax
    outputs = []
    logits1 = torch.tensor([[0.000000005, 900000.], [1., 3.]])
    output1 = F.softmax(logits1, dim=1)
    print(output1)
    outputs.append(output1)

    logits2 = torch.tensor([[1., 2.], [1., 2.]])
    output2 = F.softmax(logits2, dim=1)
    print(output2)
    outputs.append(output2)
    print(torch.cat(outputs, dim=1))
# test_softmax()

def check_MultiCategorical():
    output_size = [1, 1, 1, 1 ,1 ,1 ,1 ,1 ,1 ,1]
    output = MultiCategorical(100, output_size)
    noise = Variable(torch.FloatTensor(5, 100).normal_())
    result = output(noise, training=True, temperature=None)
    print(result)
# check_MultiCategorical()