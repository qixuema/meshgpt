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


def find_files_with_extension(folder_path, extension):
    """
    在给定的文件夹及其子文件夹中查找所有指定扩展名的文件。

    参数:
    folder_path (str): 要搜索的文件夹路径。
    extension (str): 要搜索的文件扩展名，以点开头（例如 '.ply'）。

    返回:
    list: 找到的所有指定扩展名文件的路径列表。
    """
    files_with_extension = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(root, file)
                files_with_extension.append(full_path)

    return files_with_extension