import os
import numpy as np
from scipy.io import loadmat
import hdf5storage
from skimage.metrics import structural_similarity as ssim



def compute_ssim_for_dirs(gt_dir, rec_dir, max_val=1.0):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".mat")])
    rec_files = sorted([f for f in os.listdir(rec_dir) if f.endswith(".mat")])

    common = sorted(list(set(gt_files) & set(rec_files)))
    if not common:
        raise RuntimeError("两个目录没有共同的 mat 文件名，请检查路径与文件名。")

    results = []
    for name in common:
        gt_path = os.path.join(gt_dir, name)
        rec_path = os.path.join(rec_dir, name)

        gt_mat = hdf5storage.loadmat(gt_path)
        rec_mat = hdf5storage.loadmat(rec_path)

        gt_arr = gt_mat['cube']
        rec_arr = rec_mat['cube']

        if gt_arr.shape != rec_arr.shape:
            raise ValueError(f"{name} 形状不一致：{gt_arr.shape} vs {rec_arr.shape}")
        ssim_value = ssim(gt_arr, rec_arr,data_range=max_val)  # 计算 SSIM 值

        results.append((name, ssim_value))

    return results

if __name__ == "__main__":
    gt_dir = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec"       # 真实图像 mat 文件目录
    rec_dir = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/exp/MST_Plus_Plus/202604202235"     # 预测/重建 mat 文件目录
    max_val = 1.0                             # 如果数据范围是 0~1；若是 0~255，就用 255

    all_ssim = compute_ssim_for_dirs(gt_dir, rec_dir, max_val=max_val)
    for fn, value in all_ssim:
        print(f"{fn}: SSIM = {value:.4f} ")
    avg = np.mean([v for _, v in all_ssim if np.isfinite(v)])
    print(f"平均 SSIM = {avg:.4f} ")