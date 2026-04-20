import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import json

# ==========================================
# 1. 定义评估函数 (需要根据你的业务逻辑替换)
# ==========================================
def evaluate_model(model, dataloader, device="cuda"):
    """
    在验证集上评估模型性能。
    请将此处的逻辑替换为你实际的准确率或 Loss 计算代码。
    """
    model.eval()
    correct = 0
    total = 0
    
    # 假设这是一个简单的分类任务评估逻辑
    # with torch.no_grad():
    #     for inputs, targets in dataloader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += targets.size(0)
    #         correct += (predicted == targets).sum().item()
    # return correct / total

    # 这里为了演示，返回一个模拟的准确率
    return 0.8542 

# ==========================================
# 2. 核心：模块级敏感度分析逻辑
# ==========================================
def run_sensitivity_analysis(model, dataloader, prune_ratio=0.2, device="cuda"):
    print(">>> 启动模块级敏感度分析...")
    model = model.to(device)
    
    # 获取基准线
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"[Baseline] 剪枝前基线准确率: {baseline_acc:.4f}\n")
    
    # 深拷贝一份原始的 state_dict，确保每次测试后能完美且严格地恢复结构
    original_state_dict = copy.deepcopy(model.state_dict())
    
    sensitivity_results = {}

    # 遍历模型中所有受支持的层
    for name, module in model.named_modules():
        # 这里以卷积层和全连接层为例
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"正在测试层: {name} (裁剪比例: {prune_ratio*100}%)")
            
            # 对当前层执行 L1 非结构化剪枝
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            
            # 评估剪枝后的模型
            pruned_acc = evaluate_model(model, dataloader, device)
            
            # 计算掉点 (Drop Rate)，掉点越小说明该层冗余度越高
            drop_rate = baseline_acc - pruned_acc
            
            sensitivity_results[name] = {
                "pruned_accuracy": round(pruned_acc, 4),
                "accuracy_drop": round(drop_rate, 4)
            }
            
            print(f"  -> 掉点幅度: {drop_rate:.4f}\n")
            
            # 关键步骤：严格恢复为未剪枝的原始状态
            # 注意：不能使用 prune.remove，因为那是永久固化 mask。
            # 直接加载之前备份的 state_dict 是最安全彻底的做法。
            model.load_state_dict(original_state_dict)

    # 按照掉点幅度从低到高排序
    # 排在前面的层，意味着剪掉 20% 后对模型影响极小，是后续重点剪枝对象
    sorted_results = sorted(
        sensitivity_results.items(), 
        key=lambda item: item[1]['accuracy_drop']
    )
    
    return sorted_results

# ==========================================
# 3. 执行脚本与白名单配置导出
# ==========================================
if __name__ == "__main__":
    # 实例化你的模型和 DataLoader
    # my_model = MyNetwork()
    # val_loader = get_val_dataloader()
    
    # 测试代码 (使用伪模型替代)
    my_model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    val_loader = [] # 替换为真实 DataLoader
    
    # 统一设定 20% 的测试剪枝率
    test_prune_ratio = 0.20 
    
    # 运行分析
    results = run_sensitivity_analysis(my_model, val_loader, test_prune_ratio)
    
    print(">>> 敏感度分析完成！各层排名如下 (按抗压能力从强到弱)：")
    for rank, (layer_name, metrics) in enumerate(results, 1):
        print(f" {rank}. {layer_name} | 掉点: {metrics['accuracy_drop']:.4f}")
        
    # 导出为 JSON 格式的剪枝白名单配置表
    # 这份规范的配置文件将用于指导后续 C/C++ 环境中的模型重构与推理
    config_output = {
        "baseline_accuracy": 0.8542,
        "test_prune_ratio": test_prune_ratio,
        "layer_sensitivity_ranking": dict(results)
    }
    
    with open("pruning_whitelist_config.json", "w") as f:
        json.dump(config_output, f, indent=4)
    print("\n已生成剪枝策略白名单配置文件：pruning_whitelist_config.json")