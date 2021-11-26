import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import data_loader
from torchvision.transforms import transforms
img_path = 'data/celebA/img_align_celeba'
attr_path = 'data/celebA/list_attr_celeba.txt'
image_size = 128
batch_size = 2
#celebA数据
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = data_loader.CelebA_Slim(img_path=img_path,
                                    attr_path=attr_path,
                                    transform=transform,
                                    slice=[0, -1])  # CelebA_Slim的attr参数列表选择了若干特征

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

classes = dataset.idx2attr
n_classes = len(classes)
# print('classes:', classes)
def check_dataloader():
    for i, (real_img, real_c) in enumerate(dataloader, 0):
        print(real_img, real_img.shape)  # torch.Size([2, 3, 128, 128])
        print(real_c)  # torch.Size([2, 5])
        break
check_dataloader()