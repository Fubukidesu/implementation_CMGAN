import torch
import numpy as np
# A = torch.tensor([1,2,3])
# print(A.shape)
# B = torch.tensor([[1],[2]])
# print(B.shape)
# C = torch.randn(1,)
# print(C, C.shape)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# print(encode_onehot([1,2,3]))

# A = np.array([[0, 1, 1, 0], [0, 0, 0, 1]])
# print(A, A.dtype)
# [[1, 0, 0, 1, 0, 1, 1, 0], []]
def encode_conti_onehot(labels):
    size = labels.shape
    # print(size)  # (2, 4)
    new_labels = np.zeros([size[0], size[1]*2], dtype = float)
    # print(new_labels)
    for i in range(size[0]):
        for j in range(size[1]):
            if labels[i][j] == 0.:
                new_labels[i][j*2] = 1.
                new_labels[i][j*2 + 1] = 0.
            if labels[i][j] == 1.:
                new_labels[i][j*2] = 0.
                new_labels[i][j*2 + 1] = 1.
    return new_labels.astype(np.float32)
        

# print(encode_conti_onehot(A), encode_conti_onehot(A).dtype)

