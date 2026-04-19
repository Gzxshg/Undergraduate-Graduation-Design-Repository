from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py

class TestDataset(Dataset):
    def __init__(self,data_root,crop_size,arg=True,bgr2rgb=True,stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.rgbs = []
        self.arg = arg
        self.stride = stride
        h,w = 482,512
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        bgr_data_path = f'{data_root}/Valid_RGB/'

        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            bgr_list = [line.replace('mat','jpg') for line in fin]

        bgr_list.sort()
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(bgr_list)):
            bgr_path = bgr_data_path + bgr_list[i]
            bgr = cv2.imread(bgr_path.strip())
            if bgr2rgb:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.float32(rgb)
            rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
            rgb = np.transpose(rgb, [2, 0, 1])  # [3,482,512]
            self.hypers.append(rgb)
            self.rgbs.append(rgb)

    def __getitem__(self, index):
        rgb = self.rgbs[index]
        return np.ascontiguousarray(rgb)
    
    def __len__(self):
        return len(self.rgbs)