import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import json
import numpy as np
import hdf5storage
import os
import io

from hsi_dataset import ValidDataset
from architecture import model_generator
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

gt_dir='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec'
rec_dir='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/evaluate_code/mats'
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


# ==========================================
# 1. 严格的算子白名单定义
# ==========================================
# 仅允许对以下类型的网络层进行剪枝测试，确保符合底层算子支持限制
SUPPORTED_OPERATORS_WHITELIST = (
    nn.Conv2d, 
    nn.ConvTranspose2d, 
    nn.Linear
)

# ==========================================
# 2. 模拟评估函数 (需替换为真实业务逻辑)
# ==========================================
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
        mat_dir = os.path.join('/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/evaluate_code/mats', array[i])
        save_matv73(mat_dir, 'res', result)
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".mat")])
    rec_files = sorted([f for f in os.listdir(rec_dir) if f.endswith(".mat")])
    common = sorted(list(set(gt_files) & set(rec_files)))
    for name in common:
        gt_path = os.path.join(gt_dir, name)
        rec_path = os.path.join(rec_dir, name)
        gt_mat = hdf5storage.loadmat(gt_path)
        rec_mat = hdf5storage.loadmat(rec_path)
        gt_arr = gt_mat['cube']
        rec_arr = rec_mat['res']
        ssim_value = ssim(gt_arr, rec_arr, data_range=1.0)  # 计算 SSIM 值
        psnr_value = psnr_np(gt_arr, rec_arr, max_val=1.0)  # 计算 PSNR 值
        ssim_results.append(ssim_value)
        psnr_results.append(psnr_value)
    avg_ssim = np.mean(ssim_results)
    avg_psnr = np.mean(psnr_results)
    return avg_ssim, avg_psnr 


# ==========================================
# 3. 核心：带白名单限制的模块级敏感度分析
# ==========================================
def run_strict_sensitivity_analysis(model, prune_ratio=0.2, device="cuda"):
    print(">>> 启动受限架构下的模块级敏感度分析...")
    valid_data = ValidDataset(data_root='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset', bgr2rgb=True)
    valid_loader=DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)

    model = model.to(device)
    
    # 获取完全未剪枝时的基线性能
    baseline_ssim,baseline_psnr = evaluate_inference_metric(model, valid_loader,device)
    print(f"[Baseline] 初始模型性能指标: SSIM={baseline_ssim:.4f}, PSNR={baseline_psnr:.4f}\n")
    
    # 物理级状态备份：深拷贝整个 state_dict，确保后续变量绝对单一
    original_state_dict = copy.deepcopy(model.state_dict())
    
    sensitivity_results = {}
    tested_layers_count = 0

    # 遍历网络层
    for name, module in model.named_modules():
        # 核心约束检查：模块类型是否在白名单内
        if isinstance(module, SUPPORTED_OPERATORS_WHITELIST):
            print(f"正在分析受支持层: {name} | 类型: {str(type(module).__name__)}")
            
            try:
                prune.l1_unstructured(module, name='weight', amount=prune_ratio)
                
                # 测试掉点
                pruned_ssim,pruned_psnr = evaluate_inference_metric(model,valid_loader,device)
                drop_rate_ssim = baseline_ssim - pruned_ssim
                drop_rate_psnr = baseline_psnr - pruned_psnr

                sensitivity_results[name] = {
                    "layer_type": type(module).__name__,
                    "pruned_ssim": round(pruned_ssim, 4),
                    "pruned_psnr": round(pruned_psnr, 4),
                    "metric_drop_ssim": round(drop_rate_ssim, 4),
                    "metric_drop_psnr": round(drop_rate_psnr, 4),
                    "metric_drop": round(drop_rate_psnr+drop_rate_ssim, 4),
                }
                
                print(f"  -> 掉点幅度: SSIM={drop_rate_ssim:.4f}, PSNR={drop_rate_psnr:.4f}\n")
                tested_layers_count += 1
                
            except Exception as e:
                print(f"  [!] 处理层 {name} 时发生异常: {e}")
            
            finally:
                # 强制状态重置：直接覆盖内存中的参数字典，抹除掩码残留
                model.load_state_dict(original_state_dict)

    print(f"分析完成。共测试了 {tested_layers_count} 个符合白名单规范的网络层。")

    # 按照掉点幅度从小到大排序 (掉点越小 -> 冗余度越高 -> 越能剪)
    sorted_results = sorted(
        sensitivity_results.items(), 
        key=lambda item: item[1]['metric_drop']
    )
    
    return (baseline_ssim, baseline_psnr), sorted_results

# ==========================================
# 4. 执行与规范化配置导出
# ==========================================
if __name__ == "__main__":
    pretrained_model_path = '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/test_code/modelzoo/net_300epoch.pth'
    method='mst_plus_plus'
    test_model = model_generator(method, pretrained_model_path).cuda()
    test_prune_ratio = 0.20
    
    # 运行分析
    baseline_ssim, baseline_psnr, results = run_strict_sensitivity_analysis(test_model, prune_ratio=test_prune_ratio, device="cpu")
    
    # 输出结果
    print(">>> 各层敏感度排名 (建议从顶部开始分配较高剪枝率)：")
    for rank, (layer_name, metrics) in enumerate(results, 1):
        print(f" {rank}. {layer_name} ({metrics['layer_type']}) | 掉点: {metrics['metric_drop']:.4f}")
        
    # 导出受限配置字典
    # 包含确切的层类型和层级路径，方便底层解析引擎进行内存对齐预分配
    config_output = {
        "metadata": {
            "baseline_ssim": baseline_ssim,
            "baseline_psnr": baseline_psnr,
            "test_prune_ratio": test_prune_ratio,
            "compliance_checked": True
        },
        "pruning_whitelist_ranking": dict(results)
    }
    
    with open("mst_pruning_blueprint.json", "w") as f:
        json.dump(config_output, f, indent=4)
        
    print("\n已导出规范化蓝图配置文件：mst_pruning_blueprint.json")