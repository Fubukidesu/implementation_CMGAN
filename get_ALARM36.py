## custom datasets for LUCAS
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class ALARM36(Dataset):    
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        m_data = self.data_frame.iloc[idx, :].values
        m_data = m_data.astype(float)
        # sample = {'y': np.array([m_data[2]]), 'x': np.delete(m_data, 2)}
        sample = torch.from_numpy(m_data)
        sample = sample[:, ]

        if self.transform:
            sample = self.transform(sample)

        return sample

# dataset = ALARM36(csv_file='data/ALARM/ALARM10k_num.csv')    
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# batch_num = -1
# more_batches = True
# it = iter(dataloader)
# # Dtest = np.empty((0, 9), dtype=np.float32)
# while more_batches:
#     # train discriminator
#     try:
#         labels = next(it)
#         # labels = encode_conti_onehot(labels.numpy())
#         batch = labels.numpy().astype(np.float32)
#         batch_num += 1
#         # print('labels', labels, labels.shape)  # tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0], torch.Size([64, 9])
#         print('batch', batch, batch.shape)  # 
#         # Dtest = np.append(Dtest, batch, axis=0)
#         # print('已加入第%d个batch的数据'%batch_num)
#         break
#     except StopIteration:
#         more_batches = False
#         break