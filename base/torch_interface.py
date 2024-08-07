import torch
import numpy as np

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(i, v, coo.shape)