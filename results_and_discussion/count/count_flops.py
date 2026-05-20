import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from architecture import model_generator
import cv2
from torch.autograd import Variable
import numpy as np
pretrained_model_path = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/results_and_discussion/model/origin.pth"
method='mst_plus_plus'
model = model_generator(method,pretrained_model_path)
model.eval()

bgr_path='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_RGB/ARAD_1K_0901.jpg'
bgr = cv2.imread(bgr_path) # 需要真实的 batch 维度
bgr = np.float32(bgr)
bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
bgr = np.transpose(bgr, [2, 0, 1])
bgr=torch.from_numpy(bgr).unsqueeze(0) # 添加 batch 维度
images = bgr.cuda()
images = Variable(images)

'''
flops = FlopCountAnalysis(model, images)
total_flops = flops.total()
total_params = parameter_count(model)['']

print(f"FLOPs: {total_flops/1e9:.2f} G")  # 输出单位就是 FLOPs
print(f"Params: {total_params/1e6:.2f} M")
print(f"Model: {pretrained_model_path}")
'''
from ptflops import get_model_complexity_info


macs, params = get_model_complexity_info(
    model, (3, 482, 512),
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True
)
print(f"FLOPs: {macs}")
print(f"Params: {params}")

