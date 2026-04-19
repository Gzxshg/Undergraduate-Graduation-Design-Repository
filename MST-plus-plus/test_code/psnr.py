import os
import numpy as np
from scipy.io import loadmat
import hdf5storage
from skimage.metrics import structural_similarity as ssim

pair_ssim_scores = []
def psnr_np(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)

def compute_psnr_for_dirs(gt_dir, rec_dir, max_val=1.0):
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

        psnr_value = psnr_np(gt_arr, rec_arr)

        results.append((name, psnr_value))

    return results

if __name__ == "__main__":
    gt_dir = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/dataset/Valid_Spec"       # 真实图像 mat 文件目录
    rec_dir = "/root/autodl-tmp/Undergraduate-Graduation-Design-Repository/MST-plus-plus/exp/MST_Plus_Plus/202604192019"     # 预测/重建 mat 文件目录
    max_val = 1.0                             # 如果数据范围是 0~1；若是 0~255，就用 255

    all_psnr = compute_psnr_for_dirs(gt_dir, rec_dir, max_val=max_val)
    for fn, value in all_psnr:
        print(f"{fn}: PSNR = {value:.4f} dB")
    avg = np.mean([v for _, v in all_psnr if np.isfinite(v)])
    print(f"平均 PSNR = {avg:.4f} dB")