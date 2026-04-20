#calculate the inference latency
import torch
import time
from architecture import model_generator

pretrained_model_path = '/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/test_code/modelzoo/net_300epoch.pth'  # 替换为你的模型路径
method = 'your_method'  # 替换为你的方法名称
model = model_generator(method, pretrained_model_path).cuda()
# 1. 准备模型和输入数据（确保都放在相同的设备上）
model.eval()
dummy_input = torch.randn(1, 3, 482, 512).cuda()

# 2. 预热 (Warm-up) - 极其重要！
# GPU 刚启动时需要分配显存和初始化上下文，前几次推理会非常慢。
with torch.no_grad():
    for _ in range(50):
        _ = model(dummy_input)

# 3. 正式测速
iterations = 100
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = []

with torch.no_grad():
    for _ in range(iterations):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        
        # 强制 CPU 等待 GPU 完成当前任务
        torch.cuda.synchronize()
        
        # 计算时间（毫秒）
        curr_time = starter.elapsed_time(ender)
        timings.append(curr_time)

# 4. 计算平均延迟
avg_latency = sum(timings) / iterations
print(f"平均推理延迟: {avg_latency:.2f} ms")