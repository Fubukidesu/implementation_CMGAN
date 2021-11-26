import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import data_loader
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def check_dataset():
    # Root directory for dataset
    img_path = 'data/celebA/img_align_celeba'
    attr_path = 'data/celebA/list_attr_celeba.txt'
    samples_path = 'check/ACGAN+_Samples'
    os.makedirs(samples_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the dataset
    dataset = data_loader.CelebA_Slim(img_path=img_path,
                                        attr_path=attr_path,
                                        transform=transform,
                                        slice=[0, -1])  # CelebA_Slim的attr参数列表选择了若干特征

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                shuffle=False, num_workers=0)

    classes = dataset.idx2attr
    n_classes = len(classes)

    print('classes:', classes)  # classes: {0: 'Eyeglasses', 1: 'Male', 2: 'Smiling', 3: 'Young'}
    print('n_classes:', n_classes)

    for i, (real_img, real_c) in enumerate(dataloader, 0):

            # -----------------------------------------------------------
            # Initial batch
            real_img, real_c = real_img.to(device), real_c.to(device)
            # print('real_img', real_img, real_img.shape)  # 【-1,1】[1, 3, 128, 128]
            # print('real_c', real_c, real_c.shape)  # tensor([[0., 0., 1., 1.]], device='cuda:0') torch.Size([1, 4]) 对应的
            real_batch_size = real_img.size(0)
            noise = torch.randn(real_batch_size, 100, device=device)
            # random label for computer loss
            fake_c = torch.randint(2, (real_batch_size, n_classes), dtype=torch.float32, device=device)
            # print('noise', noise, noise.shape)
            # print('fake_c', fake_c, fake_c.shape)  # fake_c tensor([[0., 1., 0., 1.]], device='cuda:0') torch.Size([1, 4])
            # fake_img = net_G(noise, fake_c)
            break
# check_dataset()
# --------------------------------------------------------------------------------------------------
def check_sample():
    # sample_noise
    n_sample = 64  # 生成图片数量
    n_classes = 5
    manualSeed = 999
    torch.manual_seed(manualSeed)
    # manual_seed的作用期很短
    ngpu = 1

    n_condition = 2 ** n_classes

    sample_noise = torch.randn(n_sample, 100, device=device)
    sample_labels = torch.zeros(n_sample, n_classes, dtype=torch.float32, device=device)

    # 生成噪声相同，但标签不同的示例噪声序列
    for i in range(n_sample):
        sample_noise[i] = sample_noise[i - i % n_condition]
        # print('i % n_condition', i % n_condition)
        bi = bin(i % n_condition)[2:].zfill(n_classes)
        # print('bi', bi)
        for j in range(len(bi)):
            if bi[j] == '1':
                sample_labels[i][j] = 1
    print('sample_noise', sample_noise)
    # print('sample_labels', sample_labels)

check_sample()
