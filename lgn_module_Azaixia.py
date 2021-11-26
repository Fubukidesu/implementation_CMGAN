import torch
from torch import nn
from multi_categorical import MultiCategorical
from utils.dag_gnn_utils import preprocess_adj_new1
from torch.autograd.variable import Variable


class LGNGenerator(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[2, 2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9):
        super(LGNGenerator, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=False, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # 把noise增加一个维度 [bs d] -> [bs d 1]
        noise = noise.unsqueeze(2).double()
        mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        # output_9 = self.output(hidden, training=training, temperature=temperature)

        return self.output(hidden, training=training, temperature=temperature), adj_A1  # torch.Size([100, 20])


class LGNDiscriminator(nn.Module):

    def __init__(self, input_size=20, hidden_sizes=[100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)

class LGNGenerator_Azaixia(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[1, 1, 1, 1, 1, 1, 1, 1, 1], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9): # 对应修改恩正维度
        super(LGNGenerator_Azaixia, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])

class LGNDiscriminator_Azaixia(nn.Module):

    def __init__(self, input_size=9, hidden_sizes=[100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator_Azaixia, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)


class LGNGenerator_Azaixia_5(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[1, 1, 1, 1, 1], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9): # 对应修改恩正维度
        super(LGNGenerator_Azaixia_5, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])

class LGNGenerator_Azaixia_5_c(nn.Module):

    def __init__(self, adj_A, noise_size=50, c_size=5, output_size=[1, 1, 1, 1, 1], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9): # 对应修改恩正维度
        super(LGNGenerator_Azaixia_5_c, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0
        self.exp = nn.Linear(c_size, c_size)

        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size + c_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # pre -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, c, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            
            hidden = self.hidden_layers(torch.cat((noise, self.exp(c)), 1))
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])

class LGNDiscriminator_Azaixia_5(nn.Module):

    def __init__(self, input_size=5, hidden_sizes=[100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator_Azaixia_5, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)

class LGNGenerator_Azaixia_7(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[1, 1, 1, 1, 1, 1 ,1], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9): # 对应修改恩正维度
        super(LGNGenerator_Azaixia_7, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])

class LGNDiscriminator_Azaixia_7(nn.Module):

    def __init__(self, input_size=7, hidden_sizes=[100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator_Azaixia_7, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)

class LGNGenerator_Azaixia_36(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[1, 1, 1, 1, 1, 1 ,1 ,1 ,1 ,1,1, 1, 1, 1, 1, 1 ,1 ,1 ,1 ,1, 1, 1, 1, 1, 1, 1 ,1 ,1 ,1,1, 1, 1, 1, 1, 1 ,1], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9): # 对应修改恩正维度
        super(LGNGenerator_Azaixia_36, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])

class LGNDiscriminator_Azaixia_36(nn.Module):

    def __init__(self, input_size=36, hidden_sizes=[100, 100, 100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator_Azaixia_36, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)

class LGNGenerator_Azaixia_18(nn.Module):

    def __init__(self, adj_A, noise_size=10, output_size=[1, 1 ,1 ,1 ,1,1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1], hidden_sizes=[100, 100, 100, 100], bn_decay=0.9): # 对应修改恩正维度
        super(LGNGenerator_Azaixia_18, self).__init__()

        #加矩阵
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))  # [d,d]全0，adj_A
        self.Wa = nn.Parameter(torch.zeros(1).double(), requires_grad=True)  # [z]全0


        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):  # 3层
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))  # fc noise -> 100
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))  # bn
            hidden_layers.append(hidden_activation)  # ReLU
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            # self.output = SingleOutput(previous_layer_size, output_size)
            print('不要单独的输出部分！')
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=True, temperature=None):
        # 加矩阵
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(adj_A1)
        # # 把noise增加一个维度 [bs d] -> [bs d 1]
        # noise = noise.unsqueeze(2).double()
        # mat_z = torch.matmul(adj_A_new1, noise + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        # #维度变回来
        # noise = mat_z.squeeze(2).float()
        
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
            # print('hidden:', hidden, hidden.shape)  # device='cuda:0', grad_fn=<ReluBackward0>) torch.Size([64, 100]

        output_10 = self.output(hidden, training=training, temperature=temperature)

        # 把noise增加一个维度 [bs d] -> [bs d 1]
        input_10 = output_10.unsqueeze(2).double()
        final_output = torch.matmul(adj_A_new1, input_10 + self.Wa) - self.Wa  # adj_A_new1[d,d] matmul (input_z[bs*d*z]+wa[z]) -> [bs,d,z] 减wa[z] -> [bs,d,z] 即mat_z
        #维度变回来
        final_output = final_output.squeeze(2).float()

        return final_output, adj_A1, output_10, self.Wa  # torch.Size([100, 20])

class LGNDiscriminator_Azaixia_18(nn.Module):

    def __init__(self, input_size=18, hidden_sizes=[100, 100, 100, 100, 100], bn_decay=0, critic=True):
        super(LGNDiscriminator_Azaixia_18, self).__init__()

        hidden_activation = nn.LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)