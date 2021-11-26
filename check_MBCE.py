import torch
def MBCE(input, target, esp=1e-19):
    loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
        (1 - target) * torch.log((1 - input).clamp_min(esp)))
    return loss

A = torch.tensor([0, 1, 1, 1, 1,], dtype=torch.float32)
B = torch.tensor([0, 1, 0, 1, 0,], dtype=torch.float32)
C = MBCE(A, B)
print('A', A)
# print('input.clamp_min(esp)', A.clamp_min(1e-19))
print('1 - target', 1 - B)
print('B', B)
print('C', C)

# fake_c = torch.randint(2, (3, 5), dtype=torch.float32)
# print(fake_c)