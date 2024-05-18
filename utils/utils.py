from pathlib import Path
import numpy as np
import torch
from einops import rearrange
import random
import json

def save_static_dict_keys(static_dict, file_path='static_dict_keys.json'):
    # 展平嵌套的键
    checkpoint_keys  = extract_keys(static_dict)

    # 保存键到文本文件
    with open(file_path, 'w') as f:
        for key in checkpoint_keys:
            f.write(f"{key}\n")

def extract_keys(d, parent_key=''):
    keys_list = []
    for k, v in d.items():
        # 构建新的键路径
        new_key = f"{parent_key}/{k}" if parent_key else k
        # 如果值是字典，递归调用
        if isinstance(v, dict):
            keys_list.extend(extract_keys(v, new_key))
        else:
            keys_list.append(new_key)
    return keys_list

def load_ae(model, load_path, device='cpu'):
    autoencoder_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = autoencoder_ckpt['ema_model']
    # 创建新的状态字典，移除'model.'前缀
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('ema_model.'):
            new_key = key.replace('ema_model.', '')
            new_model_state_dict[new_key] = value

    load_status = model.load_state_dict(new_model_state_dict)
    # print(device)
    # print(f"Autoencoder loaded from {load_path}")
    
    return model
            
def load_transformer(model, load_path):
    transformer_ckpt = torch.load(load_path)
    model_state_dict = transformer_ckpt['model']
    # 创建新的状态字典，移除'model.'前缀
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('model.'):
            new_key = key.replace('model.', '')
            key = new_key
        new_model_state_dict[key] = value
            

    load_status = model.load_state_dict(new_model_state_dict)
    
    return model


def line_list_to_house_layout(line_batch_list):
    
    Path('generate_lines/2d').mkdir(parents=True, exist_ok=True)

    for i, line_list in enumerate(line_batch_list):

        vertices = line_list['vertices'].detach().cpu().numpy()

        # 创建线索引
        lines = np.asarray(line_list['lines'])

        file_name = f'generate_lines/2d/{i}_generate_lines.npz'
        
        np.savez(file_name, vertices=vertices, lines=lines)
    


def line_coords_to_file(line_coords):
    vertices = rearrange(line_coords, 'nl nlv c -> (nl nlv) c')
    lines = [[i, i + 1] for i in range(0, vertices.size(0), 2)]

    return dict(
        vertices = vertices,
        lines = lines
    )
    
def mask_variable_data(data):
    """对 data 进行 padding"""
    length = data.size(0)
    # 遍历每个batch
    for i in range(length):
        # 对于每个batch，随机选择n的值
        n = random.randint(1, int(data.size(1) * 0.4) + 1)  # 假设n可以是1到64之间的任何值
        # 设置该batch的最后n个样本的所有元素为-1
        data[i, -n:, :] = -1

    return data