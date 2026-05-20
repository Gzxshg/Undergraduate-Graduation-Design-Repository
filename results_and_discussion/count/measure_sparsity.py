import torch
import torch.nn as nn
from architecture import model_generator
pretrained_model_path = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/results_and_discussion/model/origin_pruned_finetuned_pruned_finetuned_pruned.pth"
method='mst_plus_plus'
model = model_generator(method,pretrained_model_path)

def measure_sparsity(model):
    total_params = 0
    zero_params = 0
    
    # 遍历模型中所有的 module
    for name, module in model.named_modules():
        # 通常只统计卷积层和全连接层
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 获取当前层的权重
            tensor = module.weight.data
            
            # 计算当前层的参数总数
            num_elements = tensor.numel()
            # 计算当前层数值为 0 的参数总数
            num_zeros = torch.sum(tensor == 0).item()
            
            total_params += num_elements
            zero_params += num_zeros
            
            # 可选：打印每一层的稀疏度，用于敏感度分析
            layer_sparsity = 100. * num_zeros / num_elements
            print(f"Layer: {name} | Sparsity: {layer_sparsity:.2f}%")

    global_sparsity = 100. * zero_params / total_params
    print(f"\n[Global] Total Sparsity: {global_sparsity:.2f}%")
    return global_sparsity

if __name__ == "__main__":
    measure_sparsity(model)
