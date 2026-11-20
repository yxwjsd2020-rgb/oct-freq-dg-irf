# -*- coding: utf-8 -*-
"""
逐图指标（Dice、灵敏度、HD95、面积等）——文章D 的 s12_freq_eval.py 调用本模块的 score()、meta_from() 和 ROWS。
公开版说明：本模块与作者另一项登记研究共用；原 main()（该研究的评估流程）已删除，其余函数未改动，
不影响文章D 的任何分析；完整版待该研究发表后公开。
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import config as C
import common as U

ROWS = []


EXTS = (".nii.gz", ".png", ".nii", ".nrrd")


def m(path):
    return U.read_mask(path)


def stem(f):
    b = os.path.basename(f)
    for e in EXTS:
        if b.lower().endswith(e):
            return b[:-len(e)]
    return os.path.splitext(b)[0]


def central_px(mask):
    """图像中央 CENTER_MM 宽度内的积液像素数。只对以中心凹为中心截取的本院图像有临床含义（中心凹受累）。"""
    half = int(round(C.CENTER_MM * 1000 / C.TARGET_LATERAL_UM / 2))
    c = mask.shape[1] // 2
    return int(mask[:, max(c - half, 0):c + half].astype(bool).sum())


UNC_DIR = {"local": "unc_fold{k}", "duke": "unc_duke_fold{k}", "topcon": "unc_topcon_fold{k}", "octdl": "unc_octdl_fold{k}"}


def ensemble_unc(set_name, case):
    """基础模型 5 个折模型的前景概率方差（视网膜内平均）——不需要任何标注，可在部署前计算"""
    if set_name not in UNC_DIR:
        return np.nan
    from s05_uncertainty import load_prob
    fs = [U.p("pred", UNC_DIR[set_name].format(k=k), f"{case}.npz") for k in range(5)]
    img_f = U.p(set_name, "images", f"{case}_0000.png")
    if not all(os.path.exists(f) for f in fs) or not os.path.exists(img_f):
        return np.nan
    probs = np.stack([load_prob(f) for f in fs])
    img = U.read_gray(img_f)
    if probs.shape[1:] != img.shape:
        return np.nan
    return float(probs.var(0)[U.retina_mask(img)].mean())


def score(set_name, pred_dir, gt_dir, meta=None, **cond):
    """逐幅比较 pred_dir 与 gt_dir 中同名的 png"""
    meta = meta if meta is not None else {}
    files = sorted(f for e in EXTS for f in glob.glob(os.path.join(pred_dir, "*" + e))
                   if not (e == ".nii" and f.lower().endswith(".nii.gz")))
    if not files:
        print(f"[跳过] {set_name} {cond}：{pred_dir} 中没有预测结果"); return 0
    for f in files:
        case = stem(f)
        g = os.path.join(gt_dir, f"{case}.png") if gt_dir else None
        pr = m(f)
        gt = m(g) if g and os.path.exists(g) else (np.zeros_like(pr) if gt_dir is None else None)
        if gt is None:
            continue
        ROWS.append({"set": set_name, **cond, "case": case, **meta.get(case, {}),
                     "has_irf": int(gt.any()), "dice": U.dice(gt, pr), "sens": U.sensitivity(gt, pr),
                     "hd95": U.hd95(gt, pr), "pred_area": U.area_mm2(pr), "gt_area": U.area_mm2(gt),
                     "tp": int((gt.astype(bool) & pr.astype(bool)).sum()), "n_gt": int(gt.sum()), "n_pr": int(pr.sum()),
                     "gt_c1": central_px(gt), "pr_c1": central_px(pr),
                     "unc": ensemble_unc(set_name, case) if cond.get("condition") == "zeroshot" else np.nan})
    return len(files)


def meta_from(table, key="case", cols=("patient", "fold", "vol", "eye")):
    p = U.p("tables", table)
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, dtype=str)
    keep = [c for c in cols if c in df.columns]
    return df.set_index(key)[keep].to_dict("index")
