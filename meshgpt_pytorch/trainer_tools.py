import os
import torch
from pathlib import Path

def save_checkpoint(state, filename, max_keep=5):
    """ 保存模型并删除旧的模型 """    
    torch.save(state, filename)
    # checkpoint_dir = Path(filename).parent

    # 保留最近的 max_keep 个checkpoints
    # existing_checkpoints = [c for c in os.listdir(checkpoint_dir) if c.endswith('.pt')]
    # if len(existing_checkpoints) > max_keep:
    #     existing_checkpoints.sort()
    #     os.remove(os.path.join(checkpoint_dir, existing_checkpoints[0]))