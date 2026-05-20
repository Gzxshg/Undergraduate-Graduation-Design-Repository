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


gt_dir='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec'

def MRAE(gt, pred):
    """
    Calculate Mean Relative Absolute Error (MRAE) between ground truth and prediction.

    Args:
        gt (numpy.ndarray): Ground truth array.
        pred (numpy.ndarray): Predicted array.

    Returns:
        float: MRAE value.
    """
    assert gt.shape == pred.shape, "Ground truth and prediction must have the same shape."
    mask = gt == 0
    if mask.any():
        gt_wo_zero = gt.copy()
        gt_wo_zero[mask] = 1e-8
    else:
        gt_wo_zero = gt
    error = np.abs(pred - gt) / gt_wo_zero
    mrae = np.mean(error)
    return mrae

def evaluate_inference_metric(model, data_loader, device="cuda"):
    """
    计算模型当前在验证集上的核心指标（如 mAP, PSNR 或 Accuracy）。
    """
    MRAE_results = []

    #加载模型
    with open(f'/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/split_txt/valid_list.txt', 'r') as fin:
        hyper_list = [line.replace('.jpg\n', '.mat') for line in fin]
    model=model.to(device)
    model.eval()
    with open('/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/split_txt/valid_list.txt', 'r') as file:
        lines = file.readlines()
    array = [line.strip().replace('.jpg', '.mat') for line in lines]
    for i, (rgb) in enumerate(data_loader):
        rgb=rgb.to(device)
        output = model(rgb)
        result=output.cpu().detach().numpy() * 1.0
        result=np.transpose(np.squeeze(result), [1, 2, 0])
        result=np.minimum(result, 1.0)
        result=np.maximum(result, 0)
        gt_path = os.path.join(gt_dir, array[i])
        gt_mat = hdf5storage.loadmat(gt_path)
        gt_arr = gt_mat['cube']
        mrae_value=MRAE(gt_arr, result)
        MRAE_results.append(mrae_value)
    avg_mrae = np.mean(MRAE_results)
    return avg_mrae

if __name__ == "__main__":
    device='cuda'

    pretrained_model_path = '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/results_and_discussion/compare_model/hinet.pth'
    method='hinet'

    test_model = model_generator(method, pretrained_model_path).cuda()


    valid_data = ValidDataset(data_root='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset', bgr2rgb=True)
    valid_loader=DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)

    baseline_mrae = evaluate_inference_metric(test_model, valid_loader,device)

    print(f"MRAE: {baseline_mrae:.4f}, method: {method}")