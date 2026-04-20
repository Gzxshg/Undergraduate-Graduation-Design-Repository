import torch
from architecture import model_generator
pretrained_model_path = '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/test_code/modelzoo/net_300epoch.pth'  # 替换为你的模型路径
method = 'your_method'  # 替换为你的方法名称
model = model_generator(method, pretrained_model_path).cuda()

with open("/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/how_prune/evaluate_code/layer_count.txt", "w", encoding="utf-8") as f:
        for name,module in model.named_modules():
            f.write(f"Layer: {name}, Type: {type(module)}, Number of parameters: {sum(p.numel() for p in module.parameters())}\n")