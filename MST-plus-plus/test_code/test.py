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
from thop import profile

parser= argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='bisrnet_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='/root/autodl-tmp/exp/bisrnet_plus_plus/26000Epoch_PTH/net_250epoch.pth')
parser.add_argument('--patch_size', type=int, default=128, help='the patch size of input RGB images')
parser.add_argument('--outf', type=str, default='./exp/bisrnet_plus_plus/', help='path log files')
parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/BiSRNet-Plus-Plus/datasets')
parser.add_argument('--gpu_id', type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'#指定GPU按照PCI_BUS_ID顺序排序
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from hsi_dataset import TestDataset

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
        
        
        if output.size(1) == 31:
            for j in range(31):
                hyper_spec=output[:,j,:,:].squeeze(1)
                hyper_spec=hyper_spec.squeeze().cpu()
                single_channel = np.nan_to_num(hyper_spec)  # 处理NaN/Inf
                min_val = np.min(single_channel)
                max_val = np.max(single_channel)
                if max_val != min_val:
                    img_normalized = (single_channel - min_val) / (max_val - min_val) * 255
                else:
                    img_normalized = np.zeros_like(single_channel)
                img_8bit = img_normalized.astype(np.uint8)
                img = Image.fromarray(img_8bit)
                img.save(os.path.join(opt.outf, f'{i}_{j}.png'))
        else:
            raise NotImplementedError('The output channel of the model is not 31, please check the model and modify the code accordingly.') 

            