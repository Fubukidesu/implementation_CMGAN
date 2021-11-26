
import torch
from torch.autograd.variable import Variable

random_seed = 123
torch.manual_seed(random_seed)
print(Variable(torch.FloatTensor(5, 5).uniform_()))
print(Variable(torch.FloatTensor(5, 5).normal_()))