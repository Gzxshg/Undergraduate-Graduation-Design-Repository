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
from skimage.metrics import structural_similarity as ssim

gt_dir='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec'

def psnr_np(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)

def evaluate_inference_metric(model, data_loader, device="cuda"):
    """
    计算模型当前在验证集上的核心指标（如 mAP, PSNR 或 Accuracy）。
    """
    ssim_results = []
    psnr_results = []

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
        ssim_value = ssim(gt_arr, result, data_range=1.0)  # 计算 SSIM 值
        psnr_value = psnr_np(gt_arr, result, max_val=1.0)  # 计算 PSNR 值
        ssim_results.append(ssim_value)
        psnr_results.append(psnr_value)
    avg_ssim = np.mean(ssim_results)
    avg_psnr = np.mean(psnr_results)
    return avg_ssim, avg_psnr 

if __name__ == "__main__":
    device='cuda'

    pretrained_model_path = '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/results_and_discussion/compare_model/hinet.pth'
    method='hinet'

    test_model = model_generator(method, pretrained_model_path).cuda()


    valid_data = ValidDataset(data_root='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset', bgr2rgb=True)
    valid_loader=DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)

    baseline_ssim,baseline_psnr = evaluate_inference_metric(test_model, valid_loader,device)

    print(f"SSIM: {baseline_ssim:.4f}, PSNR: {baseline_psnr:.2f} dB, method: {method}")