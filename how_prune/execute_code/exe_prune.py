import os
import torch.nn.utils.prune as prune
import hdf5storage
import torch
import torch.nn.utils.prune as prune
import json
import numpy as np
import torch.nn as nn
import sys

from architecture import model_generator
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from hsi_dataset import ValidDataset


class DoubleOutput:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 绑定到输出
sys.stdout = DoubleOutput("/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/execute_code/pruned_model/04212106/log.txt")


with open("/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/execute_code/pruning_plan.txt", "r", encoding="utf-8") as f:
    pruning_plan = json.load(f)

pretrained_model_path = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/model_zoo/net_300epoch.pth"
method='mst_plus_plus'
model = model_generator(method,pretrained_model_path)

# 2. 执行真正的网络剪枝
for name, module in model.named_modules():
    if name in pruning_plan:
        ratio = pruning_plan[name]
        if ratio > 0:
            # 执行剪枝 (这里以 L1 非结构化为例)
            prune.l1_unstructured(module, name='weight', amount=ratio)
            print(f"已对 {name} 执行剪枝，裁剪率: {ratio*100}%",flush=True)

# 3. 验证即时性能
# 此时评估模型，PSNR 和 SSIM 必然会出现一定程度的下降，这是正常的。
gt_dir='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec'
#calculate the psnr value between two images, the input images should be in the range of [0, 1]
def psnr_np(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)
def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def evaluate_inference_metric(model, data_loader, device="cuda"):
    """
    计算模型当前在验证集上的核心指标（如 mAP, PSNR 或 Accuracy）。
    """
    ssim_results = []
    psnr_results = []

    #加载模型
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
valid_data = ValidDataset(data_root='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset', bgr2rgb=True)
valid_loader=DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)
avg_ssim, avg_psnr = evaluate_inference_metric(model, valid_loader, device="cuda")
print(f"剪枝后模型的平均 SSIM: {avg_ssim:.4f}, 平均 PSNR: {avg_psnr:.2f} dB",flush=True)

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        try:
            prune.remove(module, 'weight')
        except ValueError:
            pass

torch.save(model.state_dict(), '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/execute_code/pruned_model/04212106/pruned_clean_model.pth')