import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import time2file_name
import os
import datetime
import argparse
from PIL import Image

parser= argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='bisrnet_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/test_code/modelzoo/net_300epoch.pth')
parser.add_argument('--patch_size', type=int, default=128, help='the patch size of input RGB images')
parser.add_argument('--outf', type=str, default='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/exp/MST_Plus_Plus/', help='path log files')
parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset', help='the path of dataset')
parser.add_argument('--gpu_id', type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'#指定GPU按照PCI_BUS_ID顺序排序
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from hsi_dataset import TestDataset

with open('/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/split_txt/valid_list.txt', 'r') as file:
    lines = file.readlines()

# 去除每行末尾的换行符并去掉.jpg后缀
array = [line.strip().replace('.jpg', '') for line in lines]
print("\nloading dataset ...")
test_data = TestDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=False)
print(f"Test set samples: {len(test_data)}")
from architecture import model_generator
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(pretrained_model_path).cuda()
print('parameters number is ', sum(param.numel() for param in model.parameters()))

datetime = str(datetime.datetime.now())
date_time = time2file_name(datetime)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)


model.eval()
with torch.no_grad():
    test_loader=DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    for i, (rgb) in enumerate(test_loader):
        rgb=rgb.cuda()
        output = model(rgb)
        result = output
        result=result.clamp(min=0., max=1.)
        res=res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
        save_path = opt.outf + '/' + array[i] + '.mat'
        from scipy.io import savemat
        savemat(save_path, {'result': res})
        
            