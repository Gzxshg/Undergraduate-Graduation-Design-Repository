import re
import json

def generate_pruning_plan(input_filepath, output_filepath):
    pruning_plan = {}
    
    # 定义正则表达式，匹配格式如: " 1. body.2.embedding (Conv2d) | 掉点: -0.1177"
    # 提取 group(1): 层名 (例如 body.2.embedding)
    # 提取 group(2): 掉点数值 (例如 -0.1177)
    pattern = re.compile(r"\d+\.\s+([\w\.]+)\s+\([^)]+\)\s+\|\s+掉点:\s+(-?\d+\.\d+)")
    
    # 统计各梯队数量
    stats = {"0%": 0, "10%": 0, "30%": 0, "50%": 0}

    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 逐行或全局匹配
        matches = pattern.finditer(content)
        
        for match in matches:
            layer_name = match.group(1)
            drop_value = float(match.group(2))
            
            # 规则 1：掉点大于 0.1 的层，不进行剪枝 (0%)
            if drop_value > 0.1:
                ratio = 0.0
                stats["0%"] += 1
            # 规则 2：掉点在 0.02 到 0.1 之间的层，剪枝率 10%
            elif 0.02 < drop_value <= 0.1:
                ratio = 0.10
                stats["10%"] += 1
            # 规则 3：掉点在 0 到 0.02 之间的层，剪枝率 30%
            elif 0.0 <= drop_value <= 0.02:
                ratio = 0.30
                stats["30%"] += 1
            # 规则 4：掉点为负数区，剪枝率 50%
            elif drop_value < 0.0:
                ratio = 0.50
                stats["50%"] += 1
                
            pruning_plan[layer_name] = ratio

    # 将生成的配置字典写入新的 txt 文件（以规范的 JSON 格式写入，方便后续读取）
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(pruning_plan, f, indent=4)
        
    print(f">>> 剪枝策略生成完毕！已保存至: {output_filepath}")
    print("各梯队层数统计:")
    print(f"  - 绝对禁区 (0% 剪枝, 掉点 > 0.1): {stats['0%']} 层")
    print(f"  - 保守区   (10% 剪枝, 掉点 0.02~0.1): {stats['10%']} 层")
    print(f"  - 激进区   (30% 剪枝, 掉点 0~0.02): {stats['30%']} 层")
    print(f"  - 免疫区   (50% 剪枝, 掉点 < 0): {stats['50%']} 层")

if __name__ == "__main__":
    # 输入你现有的敏感度分析文本文件
    INPUT_FILE = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/evaluate_code/sensitive_analysis.txt"
    # 输出可供后续 PyTorch 脚本直接加载的配置文件
    OUTPUT_FILE = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/evaluate_code/pruning_plan.txt"

    generate_pruning_plan(INPUT_FILE, OUTPUT_FILE)