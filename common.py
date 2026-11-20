# -*- coding: utf-8 -*-
"""公共函数：读写图像、重采样、nnU-Net 数据集格式、视网膜区域、评价指标、统计。一般不需要修改。"""
import os
import re
import json
import glob
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import config as C

p = C.p


# ============================================================ 读写
def read_gray(path):
    """读取 2D 灰度图（png/jpg/bmp/tif），返回 float32"""
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    return a.astype(np.float32)


def read_mask(path):
    """读取掩膜：支持 png/bmp/tif 以及 3D Slicer 导出的 .nii/.nii.gz/.nrrd（需 SimpleITK）"""
    low = path.lower()
    if low.endswith((".nii", ".nii.gz", ".nrrd", ".mha")):
        try:
            import SimpleITK as sitk
            a = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(path)))
        except ImportError:                   # 没装 SimpleITK 时，NIfTI 用自带的最小读取器（ITK-SNAP / 3D Slicer 导出的 2D 掩膜）
            if not low.endswith((".nii", ".nii.gz")):
                raise
            from s03b_itksnap import read_nifti
            a = read_nifti(path)
        if a.ndim == 3:                       # 万一导出成 1 层的 3D，取非空的那一层
            a = a[np.argmax(a.reshape(a.shape[0], -1).sum(1))]
    else:
        a = np.asarray(Image.open(path))
        if a.ndim == 3:
            a = a[..., 0]
    return (a > 0).astype(np.uint8)


def resample(arr, axial_um, lateral_um, is_mask=False):
    """把 (深度, 宽度) 的 B-scan 重采样到统一像素间距"""
    zy = axial_um / C.TARGET_AXIAL_UM
    zx = lateral_um / C.TARGET_LATERAL_UM
    if abs(zy - 1) < 1e-3 and abs(zx - 1) < 1e-3:
        return arr.astype(np.uint8) if is_mask else arr
    out = ndi.zoom(arr.astype(np.float32), (zy, zx), order=0 if is_mask else 1)
    return (out > 0.5).astype(np.uint8) if is_mask else out


def to_uint8(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def save_case(folder, case_id, img, mask=None, mask_folder=None):
    """按 nnU-Net v2 的 2D PNG 格式保存：图像 case_0000.png，掩膜 case.png（0/1）"""
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(to_uint8(img)).save(os.path.join(folder, f"{case_id}_0000.png"))
    if mask is not None:
        os.makedirs(mask_folder, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)).save(os.path.join(mask_folder, f"{case_id}.png"))


def write_dataset_json(ds_folder, n_train):
    js = {"channel_names": {"0": "OCT"}, "labels": {"background": 0, "IRF": 1},
          "numTraining": n_train, "file_ending": ".png"}
    with open(os.path.join(ds_folder, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2)


def ds_dir(ds_id, name):
    return os.path.join(C.NNUNET_RAW, f"Dataset{ds_id:03d}_{name}")


def preproc_dir(ds_id, name):
    return os.path.join(C.NNUNET_PREPROCESSED, f"Dataset{ds_id:03d}_{name}")


# ============================================================ 视网膜区域（用于不确定性取平均）
def retina_mask(img):
    """粗略视网膜区域：平滑后阈值化，每一列取最上与最下的组织像素之间（ILM 到 RPE/脉络膜）"""
    from skimage.filters import threshold_otsu
    sm = ndi.gaussian_filter(img.astype(np.float32), 3)
    fg = sm > threshold_otsu(sm)
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    if n == 0:
        return np.ones_like(img, bool)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    fg = lab == (np.argmax(sizes) + 1)
    m = np.zeros_like(fg)
    for x in range(fg.shape[1]):
        ys = np.where(fg[:, x])[0]
        if ys.size:
            m[ys[0]:ys[-1] + 1, x] = True
    return m


# ============================================================ 指标
def dice(gt, pr):
    gt, pr = gt.astype(bool), pr.astype(bool)
    s = gt.sum() + pr.sum()
    return np.nan if s == 0 else 2 * (gt & pr).sum() / s


def sensitivity(gt, pr):
    gt, pr = gt.astype(bool), pr.astype(bool)
    return np.nan if gt.sum() == 0 else (gt & pr).sum() / gt.sum()


def hd95(gt, pr, spacing=(C.TARGET_AXIAL_UM, C.TARGET_LATERAL_UM)):
    """95% Hausdorff 距离（μm）。
    参考标准为空 → NaN（不计算）；参考标准有积液而预测为空 → 记为图像对角线长度（最差值），
    避免漏检图像被静默剔除、使均值偏好。"""
    gt, pr = gt.astype(bool), pr.astype(bool)
    if gt.sum() == 0:
        return np.nan
    if pr.sum() == 0:
        h, w = gt.shape
        return float(np.hypot(h * spacing[0], w * spacing[1]))
    bg = gt ^ ndi.binary_erosion(gt)
    bp = pr ^ ndi.binary_erosion(pr)
    dg = ndi.distance_transform_edt(~bg, sampling=spacing)
    dp = ndi.distance_transform_edt(~bp, sampling=spacing)
    d = np.concatenate([dg[bp], dp[bg]])
    return float(np.percentile(d, 95))


def area_mm2(mask):
    return mask.astype(bool).sum() * C.TARGET_AXIAL_UM * C.TARGET_LATERAL_UM / 1e6


# ============================================================ 统计
def bootstrap_ci(values, stat=np.nanmean, n=C.N_BOOT, seed=C.GLOBAL_SEED):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(v, v.size, replace=True)) for _ in range(n)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
