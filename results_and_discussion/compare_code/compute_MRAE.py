import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import json
import numpy as np
import hdf5storage
import os
import io
import sys

from hsi_dataset import ValidDataset
from architecture import model_generator
from torch.utils.data import DataLoader

# Define the ground truth directory
gt_dir='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec'

def compute_mrae(pred: np.ndarray, gt: np.ndarray):
    """
    Compute Mean Relative Absolute Error (MRAE).
    """
    relative_error = np.abs(pred - gt) / (np.abs(gt) + 1e-8)
    return np.mean(relative_error)

def evaluate_inference_metric(model, data_loader, device="cuda"):
    """
    Calculate MRAE for the model on the validation dataset.
    """
    mrae_results = []

    # Load the model
    with open(f'/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/split_txt/valid_list.txt', 'r') as fin:
        hyper_list = [line.replace('.jpg\n', '.mat') for line in fin]
    model = model.to(device)
    model.eval()

    with open('/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/split_txt/valid_list.txt', 'r') as file:
        lines = file.readlines()
    array = [line.strip().replace('.jpg', '.mat') for line in lines]

    for i, (rgb) in enumerate(data_loader):
        rgb = rgb.to(device)
        output = model(rgb)
        result = output.cpu().detach().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)

        gt_path = os.path.join(gt_dir, array[i])
        gt_mat = hdf5storage.loadmat(gt_path)
        gt_arr = gt_mat['cube']

        # Compute MRAE
        mrae_value = compute_mrae(result, gt_arr)
        mrae_results.append(mrae_value)

    avg_mrae = np.mean(mrae_results)
    return avg_mrae

if __name__ == "__main__":
    device = 'cuda'

    pretrained_model_path = '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/results_and_discussion/model/origin.pth'
    method = 'mst_plus_plus'

    test_model = model_generator(method, pretrained_model_path).cuda()

    valid_data = ValidDataset(data_root='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset', bgr2rgb=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)

    baseline_mrae = evaluate_inference_metric(test_model, valid_loader, device)

    print(f"MRAE: {baseline_mrae:.4f}, method: {method}")