from utils.data import CelebA
import torch.utils.data as data
import torch
from check_torchshape import encode_conti_onehot

data_path = 'data/celebA/img_align_celeba'
attr_path = 'data/celebA/list_attr_celeba.txt'
attrs_default = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows' ,'Attractive' ,'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair'
]

train_dataset = CelebA(data_path, attr_path, 32, 'valid', attrs_default)


train_dataloader = data.DataLoader(
    train_dataset, batch_size=100, num_workers=0,
    shuffle=False, drop_last=True
)


# for i, (a, b) in enumerate(train_dataloader):
#     a = a.cuda()
#     b = encode_conti_onehot(b)
#     b = torch.tensor(b).cuda()
#     print(i)
#     # print('A', a, a.shape)  # device='cuda:0') torch.Size([2, 3, 32, 32])
#     print('B', b, b.shape)  # device='cuda:0') torch.Size([2, 4])


# for epoch_index in range(0, 2):

#     more_batches = True
#     batch_num = -1

#     it = iter(train_dataloader)

#     while more_batches:
#         # train discriminator
#         for _ in range(2):
#             try:
#                 # next batch
#                 (imgs, labels) = next(it)
#                 batch_num += 1
#             except StopIteration:
#                 more_batches = False
#                 break

#             labels = encode_conti_onehot(labels.numpy())
#             # print('iter结果', imgs.shape, labels.shape)
#             print('epoch:%d, batch:%d, D'%(epoch_index, batch_num))
        
#         for _ in range (1):
#             print('epoch:%d, batch:%d, G'%(epoch_index, batch_num))
        

