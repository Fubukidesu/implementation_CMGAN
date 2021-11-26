import torch
# --------------------------------------------------------------------------------------------------
# sample_noise 按照特征的数量，按照二进制顺序产生噪声一致但标签不同的sample对。
n_sample = 16  # 生成图片数量
manualSeed = 999
torch.manual_seed(manualSeed)
latent_dim = 10
# manual_seed的作用期很短

n_classes = 3
n_condition = 2 ** n_classes

sample_noise = torch.randn(n_sample, latent_dim)
sample_labels = torch.zeros(n_sample, n_classes, dtype=torch.float32)

# 生成噪声相同，但标签不同的示例噪声序列
for i in range(n_sample):
    sample_noise[i] = sample_noise[i - i % n_condition]
    # print(i % n_condition)
    bi = bin(i % n_condition)[2:].zfill(n_classes)
    # print(bi)
    for j in range(len(bi)):
        if bi[j] == '1':
            sample_labels[i][j] = 1
print(sample_noise)
print(sample_labels)