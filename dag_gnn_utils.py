import torch

def preprocess_adj_new1(adj):  # (I-A^T)^(-1)
    adj_normalized = torch.inverse(torch.eye(adj.shape[0], device='cuda').double()-adj.transpose(0,1))
    return adj_normalized

def matrix_poly(matrix, d):
    x = torch.eye(d, device='cuda').double()+ torch.div(matrix, d).double()  # I + matrix/[结点数]，即x
    return torch.matrix_power(x, d)  # x的d次方 (I+M/d)^d