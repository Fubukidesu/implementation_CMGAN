# 用Dtrain(181952)训练处LRtrain，利用Dtest(19904)算出f1分数，得到ftrain
# 用Dsample训练LRsample，利用Dtest算出f1分数，得到fsample
import numpy as np
from utils.data import CelebA
import torch.utils.data as data

def get_celebAtrain():
    # 得到celebA的训练数据 [0. 0. 0. ... 1. 0. 0.]] (181952, 9)
    data_path = 'data/celebA/img_align_celeba'
    attr_path = 'data/celebA/list_attr_celeba.txt'
    attrs_default = [
    'Bald', 'Eyeglasses', 'Male', 'Mustache', 'Mouth_Slightly_Open', 'Young', 'Wearing_Lipstick', 'Smiling', 'Narrow_Eyes'
    ]

    train_dataset = CelebA(data_path, attr_path, 32, 'train', attrs_default)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=64, num_workers=0,
        shuffle=True, drop_last=True
    )

    batch_num = -1
    more_batches = True
    it = iter(train_dataloader)
    Dtrain = np.empty((0, 9), dtype=np.float32)
    while more_batches:
        # train discriminator
        try:
            (imgs, labels) = next(it)
            # labels = encode_conti_onehot(labels.numpy())
            batch = labels.numpy().astype(np.float32)
            batch_num += 1
            Dtrain = np.append(Dtrain, batch, axis=0)
            print('已加入第%d个batch的数据'%batch_num)
        except StopIteration:
            more_batches = False
            break
    print('Dtrain：', Dtrain, Dtrain.shape) # [0. 0. 1. ... 0. 1. 0.]] (19904, 9)
    np.save('data/celebA_exp/Dtrain_180000.npy', Dtrain)

# get_celebAtrain()

# 看看方法本身有没有用
testtrain = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]).astype(np.float32)
testsamples = np.array([[0,0,0,0], [1,1,1,1]]).astype(np.float32)
testtest = np.array([[1,1,1,1], [1,1,1,1]]).astype(np.float32)

np.save('data/testforexp/testtrain.npy', testtrain)
np.save('data/testforexp/testsamples.npy', testsamples)
np.save('data/testforexp/testtest.npy', testtest)
