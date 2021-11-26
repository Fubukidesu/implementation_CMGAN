import numpy as np
from torch.autograd.variable import Variable
from utils.data import CelebA
import torch.utils.data as data
from utils.cuda import to_cuda_if_available, to_cpu_if_available
from torchvision.utils import make_grid
from utils.initialization import load_or_initialize
from PIL import Image
import torch
from CIGN_module import CIGNdiscriminator, CIGNGenerator

#celebA数据
data_path = 'data/celebA/img_align_celeba'
attr_path = 'data/celebA/list_attr_celeba.txt'
attrs_default = [
   'No_Beard' ,'Mustache', 'Bald', 'Young', 'Male'
]

train_dataset = CelebA(data_path, attr_path, 64, 'train', attrs_default)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=64, num_workers=0,
    shuffle=False, drop_last=True
)

generator = CIGNGenerator()
load_or_initialize(generator, 'output/models/cign_models/epoch:99_generator.pth')
generator = to_cuda_if_available(generator)
print('生成器初始化完毕')

it = iter(train_dataloader)
(imgs, labels) = next(it)
# print('img', imgs, imgs.shape)  # torch.Size([64, 3, 64, 64]) [-1,1]
print(labels)

# G产生图片也没问题
# sample_labels = np.array([[1, 0, 1, 1, 1], [1, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 1, 1]])
# sample_labels = Variable(to_cuda_if_available(torch.FloatTensor(sample_labels)))
# sample_noise = Variable(torch.FloatTensor(4, 128).normal_())
# sample_noise = to_cuda_if_available(sample_noise)
# sample_imgs = generator(sample_labels, sample_noise)
# print('样本产生完毕')
# grid = make_grid(sample_imgs.data, nrow=2, padding=2, normalize=True)
# ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
# im = Image.fromarray(ndarr)
# im.save('output/test_cign_imgs.png')  

# celebA的图片能正常显示
# img = imgs[0]
# print('img', img, img.shape)  # 3 64 64
# grid = make_grid(img.data, nrow=1, padding=2, normalize=True)
# ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
# im = Image.fromarray(ndarr)
# im.save('output/test.png')  