import numpy as np
import torch
from torch.autograd.variable import Variable
from formats import data_formats, loaders
from datasets import Dataset
features = np.load('data/synthetic/fixed_2/synthetic.features.npz')
# Min: 0
# Max: 1
# Rows: 10000
# Columns: 20
# Mean ones per row: 10.0
# Mean ones per column: 5000.0
# Total ones: 100000
# Total positions: 200000
# Total ratio of ones: 0.5
# Empty rows: 0
# Full rows: 0
# Empty columns: 0
# Full columns: 0

# files = features.files  # ['indices', 'indptr', 'format', 'shape', 'data']
# print(features['data'], features['data'].size)  # [1 1 1 ... 1 1 1] 100000
# print(features['indices'], features['indices'].size)  # [ 1  2  5 ... 14 17 19] 100000
# print(features['shape'])  # [10000    20]
# print(features['indptr'], features['indptr'].size)
# print(features['format'], features['format'].size)

# features = loaders['sparse']('data/synthetic/fixed_2/synthetic.features.npz')
# # print('features:', features, features.shape)  # numpy.ndarray (10000, 20) float32
# data = Dataset(features)
# train_data, val_data = data.split(1.0 - .1)
# # variable_sizes = load_variable_sizes_from_metadata('data/synthetic/fixed_2/metadata.json')  # [2, 2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2]
# train_data_iterator = train_data.batch_iterator(2)
# batch = next(train_data_iterator)
# real_features = Variable(torch.from_numpy(batch))
# print('batch', batch, batch.shape)  # batch [[0. 1. 0. 1. 0. 1. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 0. 1. 0. 1.]
# #  [1. 0. 1. 0. 0. 1. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 0. 1. 1. 0.]] (2, 20)
# print('real_features', real_features, real_features.shape)  # real_features tensor([[0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1.,
#         #  0., 1.],
#         # [1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1.,
#         #  1., 0.]]) torch.Size([2, 20])

# 用np.array 的float32来生成简单的.npy文件，测试标签的对比评估方法。
# testa = np.array([[1, 0, 1], [1, 0, 1]])
# # print(testa, testa.dtype) # int64
# testa = testa.astype(np.float32)
# # print(testa, testa.dtype) # float32
# testb = np.array([[1, 0, 1], [1, 0, 1]])
# testb = testb.astype(np.float32)
# np.save('data/testforexp/testa.npy', testa)
# np.save('data/testforexp/testb.npy', testb)

# data = np.array([[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]]).astype(np.float32)
# variable_sizes = [1,1,1,1]
# for selected_index, variable_size in enumerate(variable_sizes):
#     print('_________________')
#     left_size = sum(variable_sizes[:selected_index])
#     print('left_size', left_size)
#     left = data[:, :left_size]
#     print('left', left)
#     labels = data[:, left_size:left_size + variable_sizes[selected_index]]
#     print('variable_sizes[selected_index]', variable_sizes[selected_index])
#     print('labels', labels)
#     right = data[:, left_size + variable_sizes[selected_index]:]
#     print('right', right)
#     features = np.concatenate((left, right), axis=1)
#     print('features', features)