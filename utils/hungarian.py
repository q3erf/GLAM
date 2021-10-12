import torch
import scipy.optimize as opt
import numpy as np


def hungarian(s: torch.Tensor, n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat


def hungarian_from_single(s: torch.Tensor, n1=None, n2=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    device = s.device
    perm_mat = s.cpu().detach().numpy() * -1
    n1b = perm_mat.shape[0] if n1 is None else n1
    n2b = perm_mat.shape[1] if n2 is None else n2
    row, col = opt.linear_sum_assignment(perm_mat[:n1b, :n2b])
    perm_mat = np.zeros_like(perm_mat)
    perm_mat[row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat

if __name__ == '__main__':
    a = torch.rand(8,8)
    pad = torch.nn.ZeroPad2d((0,4,0,4))
    b = pad(a)
    print(hungarian_from_single(a))