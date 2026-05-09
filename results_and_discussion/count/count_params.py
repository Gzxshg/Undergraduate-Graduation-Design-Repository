from architecture import model_generator
import torch
import json
import sys
import numpy as np
from hsi_dataset import ValidDataset
from torch.utils.data import DataLoader

pretrained_model_path = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/results_and_discussion/model/origin_pruned_finetuned.pth"
method='mst_plus_plus'
model = model_generator(method,pretrained_model_path)
total = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total}")