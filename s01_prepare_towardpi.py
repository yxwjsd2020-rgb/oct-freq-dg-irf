# -*- coding: utf-8 -*-
"""
第 1 步：TowardPi DME 数据 → nnU-Net 训练集（Dataset101_TowardPi）
-----------------------------------------------------------------
用法：
  python s01_prepare_towardpi.py --inspect        # 先看数据：维度、掩膜取值、三个方向的预览图
  python s01_prepare_towardpi.py                  # 正式转换 + 生成按眼（或患者）分组的 5 折划分
  python s01_prepare_towardpi.py --pick-reannot 20  # 随机抽 20 幅含 IRF 的 B-scan 供本院医师重新勾画
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from sklearn.model_selection import GroupKFold
import config as C
import common as U


def read_volume(path):
    try:
        import tifffile
        return np.asarray(tifffile.imread(path))
    except ImportError:
        im = Image.open(path)
        return np.stack([np.asarray(f.convert("F") if f.mode not in ("L", "I;16", "F") else f)
                         for f in ImageSequence.Iterator(im)])


def list_volumes():
    inp = os.path.join(C.RAW_TOWARDPI, C.TP_INPUT_DIR)
    lab = os.path.join(C.RAW_TOWARDPI, C.TP_LABEL_DIR)
    files = sorted(glob.glob(os.path.join(inp, "*.tif*")))
    if not files:
        raise SystemExit(f"在 {inp} 下没有找到 .tif/.tiff 文件，请检查 config.RAW_TOWARDPI 和 TP_INPUT_DIR")
    out = []
    for f in files:
        stem = os.path.basename(f).split(".")[0]
        m = re.match(C.TP_NAME_REGEX, stem)
        if not m:
            print(f"[跳过] 文件名不符合规则：{stem}")
            continue
        cand = glob.glob(os.path.join(lab, stem + C.TP_MASK_SUFFIX + ".tif*"))
        if not cand:
            print(f"[跳过] 找不到掩膜：{stem}{C.TP_MASK_SUFFIX}.tiff")
            continue
        out.append({"vol": stem, "eye": m.group(1).upper() + m.group(2), "img": f, "mask": cand[0]})
    return pd.DataFrame(out)


def group_of(df):
    if C.TP_EYE_PATIENT_CSV:
        mp = pd.read_csv(C.TP_EYE_PATIENT_CSV, dtype=str).set_index("eye_id")["patient_id"].to_dict()
        return df["eye"].map(lambda e: mp.get(e, e))
    return df["eye"]


def vol_info(path):
    """不读入整个容积，只看页数、每页尺寸和数据类型"""
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        return len(t.pages), tuple(p.shape), str(p.dtype)


def fix_mask(msk, vol_shape):
    """掩膜常见变体：多一个通道维（取最大值）；其余形状不一致返回 None"""
    if msk.ndim == len(vol_shape) + 1:
        msk = msk.max(axis=-1)
    return msk if msk.shape == tuple(vol_shape) else None


def bscans(vol):
    """把容积按 B-scan 拆开，并统一为 (深度, 宽度)"""
    v = np.moveaxis(vol, C.TP_BSCAN_AXIS, 0)
    if C.TP_DEPTH_AXIS_IN_BSCAN == 1:
        v = np.swapaxes(v, 1, 2)
    return v


def inspect():
    df = list_volumes()
    print(f"共 {len(df)} 个带掩膜的 DME 容积，{df['eye'].nunique()} 只眼")
    try:
        rows = []
        for _, r in df.iterrows():
            ni, si, di = vol_info(r["img"]); nm, sm, dm = vol_info(r["mask"])
            rows.append({"vol": r["vol"], "图像页数": ni, "图像每页": si, "图像类型": di,
                         "掩膜页数": nm, "掩膜每页": sm, "掩膜类型": dm,
                         "一致": "是" if (ni, si) == (nm, sm[:len(si)]) else "否"})
        info = pd.DataFrame(rows)
        info.to_csv(U.p("inspect", "towardpi_volumes.csv"), index=False, encoding="utf-8-sig")
        print(f"页数分布：{info['图像页数'].value_counts().to_dict()}；掩膜类型：{info['掩膜类型'].value_counts().to_dict()}")
        bad = info[info["一致"] == "否"]
        if len(bad):
            print("  !! 以下容积的图像与掩膜尺寸不一致（转换时会单独处理或跳过）：\n" + bad.to_string(index=False))
        print(f"  逐个容积的信息：{U.p('inspect', 'towardpi_volumes.csv')}")
    except ImportError:
        print("（未安装 tifffile，跳过逐个容积检查：pip install tifffile）")
    r = df.iloc[0]
    vol, msk = read_volume(r["img"]), read_volume(r["mask"])
    print(f"示例 {r['vol']}：图像形状 {vol.shape} {vol.dtype}，灰度范围 {vol.min()}–{vol.max()}")
    print(f"           掩膜形状 {msk.shape}，取值 {np.unique(msk)[:10]}")
    if vol.shape != msk.shape:
        print("  !! 图像与掩膜形状不一致，请停止并联系负责人")
    if len(np.unique(msk)) > 2:
        print("  !! 掩膜取值超过 2 种（可能含多个类别，如 IRF/SRF），当前代码会把所有非零值都当作 IRF，"
              "请停止并联系负责人确认各取值含义")
    else:
        print("  掩膜为二值（0 与一个非零值），代码会把非零值当作 IRF，无需处理")
    outdir = U.p("inspect")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for nm in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if nm in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [nm]; break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for k in range(3):
        mid = vol.shape[k] // 2
        sl = np.take(vol, mid, axis=k); ml = np.take(msk, mid, axis=k)
        ax[0, k].imshow(sl, cmap="gray"); ax[0, k].set_title(f"axis={k} 第{mid}层（图像）")
        ax[1, k].imshow(sl, cmap="gray"); ax[1, k].contour(ml > 0, colors="r", linewidths=0.8)
        ax[1, k].set_title(f"axis={k}（红线=IRF掩膜）")
    for a in ax.ravel():
        a.axis("off")
    plt.suptitle("哪一列看起来像正常的 B-scan（视网膜横断面，上方玻璃体、下方脉络膜）？\n"
                 "那一列的 axis 就是 TP_BSCAN_AXIS；若图中视网膜是竖着的，TP_DEPTH_AXIS_IN_BSCAN 设为 1")
    f = os.path.join(outdir, "towardpi_axes.png")
    plt.tight_layout(); plt.savefig(f, dpi=110)
    print(f"预览图已保存：{f}\n请打开预览图确定 config.py 中的 TP_BSCAN_AXIS 和 TP_DEPTH_AXIS_IN_BSCAN。")


def convert():
    df = list_volumes()
    df["group"] = group_of(df)
    ds = U.ds_dir(C.DS_TOWARDPI, "TowardPi")
    rows, skipped = [], []
    for _, r in df.iterrows():
        v0, m0 = read_volume(r["img"]), read_volume(r["mask"])
        m0 = fix_mask(m0, v0.shape)
        if m0 is None:
            print(f"[跳过] {r['vol']}：掩膜尺寸与图像不一致（见 inspect/towardpi_volumes.csv），请告知负责人")
            skipped.append(r["vol"]); continue
        vol, msk = bscans(v0), bscans(m0)
        for k in range(0, vol.shape[0], C.TP_EVERY_NTH):
            img = U.resample(vol[k].astype(np.float32), C.TP_AXIAL_UM, C.TP_LATERAL_UM)
            m = U.resample((msk[k] > 0).astype(np.uint8), C.TP_AXIAL_UM, C.TP_LATERAL_UM, is_mask=True)
            cid = f"TP_{r['vol']}_{k:03d}"
            U.save_case(os.path.join(ds, "imagesTr"), cid, img, m, os.path.join(ds, "labelsTr"))
            rows.append({"case": cid, "vol": r["vol"], "eye": r["eye"], "group": r["group"],
                         "slice": k, "has_irf": int(m.any()), "irf_area_mm2": U.area_mm2(m)})
        print(f"{r['vol']}：已转换")
    cases = pd.DataFrame(rows)
    U.write_dataset_json(ds, len(cases))
    cases.to_csv(U.p("tables", "towardpi_cases.csv"), index=False)
    # 按组的 5 折（同一眼/患者的所有 B-scan 在同一折）
    gkf = GroupKFold(n_splits=5)
    groups = cases["group"].values
    splits = []
    for tr, va in gkf.split(cases, groups=groups):
        splits.append({"train": cases["case"].iloc[tr].tolist(), "val": cases["case"].iloc[va].tolist()})
    U.save_json(splits, U.p("splits", "towardpi_splits_final.json"))
    if skipped:
        print(f"\n注意：跳过 {len(skipped)} 个容积：{skipped}")
    print(f"\n完成：{len(df) - len(skipped)} 个容积 → {len(cases)} 幅 B-scan（含 IRF {cases['has_irf'].mean():.0%}）")
    print(f"分组方式：{'按患者' if C.TP_EYE_PATIENT_CSV else '按眼（文件名中无患者编号）'}；划分已保存。")


def pick_reannot(n):
    cases = pd.read_csv(U.p("tables", "towardpi_cases.csv"))
    pos = cases[cases["has_irf"] == 1]
    rng = np.random.default_rng(C.GLOBAL_SEED)
    eyes = rng.permutation(pos["eye"].unique())[:n]          # 每只眼抽 1 幅，尽量来自不同患者
    pick = pd.concat([pos[pos["eye"] == e].sample(1, random_state=C.GLOBAL_SEED) for e in eyes])
    out = U.p("towardpi_reannot", "to_annotate")
    src = os.path.join(U.ds_dir(C.DS_TOWARDPI, "TowardPi"), "imagesTr")
    for c in pick["case"]:
        Image.open(os.path.join(src, f"{c}_0000.png")).save(os.path.join(out, f"{c}.png"))
    pick.to_csv(U.p("tables", "towardpi_reannot_list.csv"), index=False)
    print(f"已导出 {len(pick)} 幅图像到 {out}，请本院医师按 SOP 勾画（不要查看原掩膜），\n"
          f"掩膜以相同文件名保存到 {U.p('towardpi_reannot', 'masks')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--pick-reannot", type=int, default=0)
    a = ap.parse_args()
    if a.inspect:
        inspect()
    elif a.pick_reannot:
        pick_reannot(a.pick_reannot)
    else:
        convert()
