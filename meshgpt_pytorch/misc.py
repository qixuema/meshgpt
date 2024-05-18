import os
import torch.nn.functional as F


# helper functions

def identity(t):
    return t

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

def get_file_list(dir_path):
    file_path_list = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    file_path_list.sort()
    return file_path_list