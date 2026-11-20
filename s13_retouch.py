# -*- coding: utf-8 -*-
"""
第 13 步（文章D）：RETOUCH 训练集 → 外部测试集（只取 IRF 标签）
-----------------------------------------------------------------
数据：RETOUCH 挑战赛训练集（Bogunović et al., IEEE TMI 2019），三家设备：
    Cirrus 24 / Spectralis 24 / Topcon 22 个容积；标签 0=背景、1=IRF、2=SRF、3=PED。
预期解压结构（文件夹名只要含设备名即可，大小写不限）：
    D:\\公共数据集\\RETOUCH\\
        RETOUCH-TrainingSet-Cirrus\\TRAIN001\\oct.mhd + oct.raw、reference.mhd + reference.raw
        RETOUCH-TrainingSet-Spectralis\\TRAIN0xx\\...
        RETOUCH-TrainingSet-Topcon\\TRAIN0xx\\...
输出（与文章B 外部测试集格式一致，供 gpu/run_freq.sh predict 和 s12_freq_eval.py 使用）：
    work/retouch_cirrus|retouch_spectralis|retouch_topcon/
        images/<case>_0000.png   模型输入（已重采样到 3.9 × 11.7 μm）
        gt/<case>.png            IRF 掩膜（0/1）；SRF、PED 一律当背景
        qc/<case>.png            抽样质控图（绿 = IRF，蓝 = SRF，黄 = PED，便于核对方向和标签）
    work/tables/retouch_cases.csv  每幅 B-scan 一行：case, patient, vendor, volume, bscan, has_irf, has_srf, has_ped, ...
    work/tables/retouch_volumes.csv 每个容积一行：尺寸、像素间距、灰度处理方式
用法：
    python s13_retouch.py --inspect      只读取每家设备第 1 个容积的文件头并打印，不写文件（先跑这个核对）
    python s13_retouch.py                全部转换
    python s13_retouch.py --every 2      每隔 2 张取 1 张 B-scan（默认全部）
    python s13_retouch.py --spectralis-nodisp
                                         SAP 敏感性分析 S7：只把 Spectralis 再转换一份、不做 1/4 次方显示变换，输出到
                                         work/retouch_spectralis_nodisp/（表格 retouch_cases_spectralis_nodisp.csv）。
                                         须先做过全部转换；若 Spectralis 原始数据本来就不是线性强度（gray 全为 none），S7 不适用，直接退出
注意：
  1. 不需要 SimpleITK，自带 MetaImage(.mhd) 读取器。
  2. 外部测试集锁定：本脚本只做格式转换，不看任何模型结果。
  3. RETOUCH 疾病为 AMD/RVO 而非 DME，论文中需在局限中说明。
"""
import os
import re
import sys
import glob
import zlib
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import config as C
import common as U

RAW_RETOUCH = os.environ.get("RAW_RETOUCH", getattr(C, "RAW_RETOUCH", r"D:\公共数据集\RETOUCH"))
VENDORS = ("Cirrus", "Spectralis", "Topcon")

# 各设备标称像素间距（μm）：只在 .mhd 文件头缺失或明显不合理时使用，并在 retouch_volumes.csv 中标记 spacing_source=nominal
# Cirrus 512×1024×128，6×2×6 mm；Spectralis 512×496×49，约 6×1.9×6 mm；Topcon 512×885(或 650)×128，6×2.3×6 mm
NOMINAL_UM = {"Cirrus": (2000 / 1024, 6000 / 512), "Spectralis": (3.87, 11.3), "Topcon": (2300 / 885, 6000 / 512)}
AXIAL_OK = (1.0, 8.0)        # 合理的轴向间距范围（μm）
LATERAL_OK = (5.0, 25.0)     # 合理的横向间距范围（μm）
QC_EVERY = 8                 # 每个容积每隔几张 B-scan 存一张质控图

# Spectralis 原始数据可能是线性强度（高动态范围），需要像 Heidelberg 软件一样做 1/4 次方变换后才接近显示图。
# "auto"：16 位/浮点且分布极度偏斜时自动变换；"on"：总是变换；"off"：不变换（敏感性分析）
SPECTRALIS_DISPLAY = os.environ.get("RETOUCH_SPECTRALIS_DISPLAY", "auto")

MET_TYPES = {"MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
             "MET_UINT": np.uint32, "MET_INT": np.int32, "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64}


# ============================================================ MetaImage 读取
def read_mhd(path, header_only=False):
    """返回 (数组 (z, y, x) = (B-scan, 深度, A-scan), 文件头 dict)。支持未压缩和 zlib 压缩（CompressedData = True）。"""
    hdr = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                hdr[k.strip()] = v.strip()
    dims = [int(v) for v in hdr["DimSize"].split()]
    if header_only:
        return None, hdr
    dtype = np.dtype(MET_TYPES[hdr["ElementType"]])
    if hdr.get("BinaryDataByteOrderMSB", hdr.get("ElementByteOrderMSB", "False")).lower() == "true":
        dtype = dtype.newbyteorder(">")
    ch = int(hdr.get("ElementNumberOfChannels", "1"))
    datafile = hdr["ElementDataFile"]
    if datafile == "LOCAL":
        raise ValueError(f"{path}：数据嵌在 .mhd 内（LOCAL），本读取器未支持，请改用 SimpleITK")
    raw = open(os.path.join(os.path.dirname(path), datafile), "rb").read()
    if hdr.get("CompressedData", "False").lower() == "true":
        raw = zlib.decompress(raw)
    n = int(np.prod(dims)) * ch
    a = np.frombuffer(raw, dtype=dtype, count=n)
    shape = dims[::-1] + ([ch] if ch > 1 else [])        # 文件按 x 变化最快存储 → numpy (z, y, x)
    a = a.reshape(shape)
    if ch > 1:
        a = a[..., 0]
    return a, hdr


def spacing_um(hdr, vendor):
    """返回 (轴向 μm, 横向 μm, 来源)。ElementSpacing 顺序为 x(A-scan) y(深度) z(B-scan)，单位 mm。"""
    sp = hdr.get("ElementSpacing") or hdr.get("ElementSize")
    if sp:
        v = [float(t) for t in sp.split()]
        lat, ax = v[0] * 1000, v[1] * 1000
        if AXIAL_OK[0] <= ax <= AXIAL_OK[1] and LATERAL_OK[0] <= lat <= LATERAL_OK[1]:
            return ax, lat, "header"
        print(f"  [{vendor}] 文件头像素间距不合理（轴向 {ax:.2f} μm、横向 {lat:.2f} μm），改用标称值 {NOMINAL_UM[vendor]}")
    return NOMINAL_UM[vendor][0], NOMINAL_UM[vendor][1], "nominal"


# ============================================================ 灰度
def display_transform(vol, vendor):
    """返回 (float32 容积, 处理方式)。仅对 Spectralis 判断是否需要 1/4 次方变换。"""
    v = vol.astype(np.float32)
    if vendor != "Spectralis" or SPECTRALIS_DISPLAY == "off":
        return v, "none"
    if SPECTRALIS_DISPLAY == "on":
        return np.power(np.clip(v, 0, None), 0.25), "pow0.25"
    if vol.dtype == np.uint8:
        return v, "none"
    pos = v[v > 0]
    if pos.size and np.percentile(pos, 99.5) / max(np.median(pos), 1e-6) > 20:   # 线性强度：亮的 RPE 比中位数高几十倍以上
        return np.power(np.clip(v, 0, None), 0.25), "pow0.25(auto)"
    return v, "none"


def fill_padding_bscan(img):
    """Spectralis 配准后上下边缘的纯色填充块 → 替换为背景（沿用文章B 对 Duke 的处理）"""
    from s02_prepare_external import fill_padding
    return fill_padding(img)


# ============================================================ 文件查找
def find_volumes():
    out = {}
    for vendor in VENDORS:
        dirs = [d for d in glob.glob(os.path.join(glob.escape(RAW_RETOUCH), "*")) if os.path.isdir(d) and vendor.lower() in os.path.basename(d).lower()]
        cases = []
        for d in dirs:
            for oct_path in sorted(glob.glob(os.path.join(glob.escape(d), "**", "oct.mhd"), recursive=True)):
                ref = os.path.join(os.path.dirname(oct_path), "reference.mhd")
                cases.append((oct_path, ref if os.path.exists(ref) else None))
        out[vendor] = cases
    return out


def volume_id(oct_path):
    name = os.path.basename(os.path.dirname(oct_path))
    m = re.search(r"(\d+)", name)
    return f"{int(m.group(1)):03d}" if m else re.sub(r"\W", "", name)


# ============================================================ 主流程
def qc_image(img, lab):
    from scipy import ndimage as ndi
    g = U.to_uint8(img)
    rgb = np.stack([g] * 3, -1)
    for val, col in ((3, (255, 255, 0)), (2, (0, 128, 255)), (1, (0, 255, 0))):
        m = lab == val
        if m.any():
            rgb[m ^ ndi.binary_erosion(m)] = col
    return rgb


def resample_label(lab, ax, lat):
    """多类标签重采样（最近邻），保留 1/2/3 以便统计 SRF、PED"""
    from scipy import ndimage as ndi
    zy, zx = ax / C.TARGET_AXIAL_UM, lat / C.TARGET_LATERAL_UM
    if abs(zy - 1) < 1e-3 and abs(zx - 1) < 1e-3:
        return lab.astype(np.uint8)
    return ndi.zoom(lab.astype(np.uint8), (zy, zx), order=0)


def inspect(vols):
    print(f"RETOUCH 目录：{RAW_RETOUCH}")
    for vendor, cases in vols.items():
        print(f"\n[{vendor}] 找到 {len(cases)} 个容积（有标签 {sum(r is not None for _, r in cases)} 个）")
        if not cases:
            continue
        oct_path, ref = cases[0]
        a, hdr = read_mhd(oct_path)
        ax, lat, src = spacing_um(hdr, vendor)
        print(f"  示例 {oct_path}")
        print(f"  DimSize(x y z) = {hdr['DimSize']}；ElementType = {hdr['ElementType']}；ElementSpacing(mm) = {hdr.get('ElementSpacing')}")
        print(f"  → 轴向 {ax:.2f} μm、横向 {lat:.2f} μm（{src}）；灰度范围 {a.min()}–{a.max()}，中位数 {np.median(a):.1f}")
        v, how = display_transform(a, vendor)
        print(f"  灰度处理：{how}")
        top, bot = v[:, : v.shape[1] // 10].mean(), v[:, -v.shape[1] // 10:].mean()
        print(f"  顶部 10% 行均值 {top:.1f}，底部 10% 行均值 {bot:.1f}（两者都应偏暗；视网膜应在中间。务必看 qc 图确认没有上下颠倒）")
        if ref:
            r, _ = read_mhd(ref)
            vals, cnt = np.unique(r, return_counts=True)
            print(f"  标签取值：{dict(zip(vals.tolist(), cnt.tolist()))}（应为 0/1/2/3；1 = IRF）")
            if r.shape != a.shape:
                print(f"  ⚠ 标签尺寸 {r.shape} 与图像 {a.shape} 不一致")


def convert(vols, every=1, suffix="", table_suffix=""):
    rows, vrows = [], []
    for vendor, cases in vols.items():
        key = f"retouch_{vendor.lower()}{suffix}"
        img_dir, gt_dir, qc_dir = U.p(key, "images"), U.p(key, "gt"), U.p(key, "qc")
        if cases:                                   # 重新转换时先清空旧输出，避免 --every 不同造成新旧文件混在一起
            for d in (img_dir, gt_dir, qc_dir):
                for f in glob.glob(os.path.join(glob.escape(d), "*.png")):
                    os.remove(f)
        n_img = n_irf = 0
        for oct_path, ref in cases:
            if ref is None:
                print(f"  [{vendor}] {oct_path} 没有 reference.mhd，跳过"); continue
            vol, hdr = read_mhd(oct_path)
            lab, _ = read_mhd(ref)
            if lab.shape != vol.shape:
                raise SystemExit(f"{oct_path}：标签尺寸 {lab.shape} 与图像 {vol.shape} 不一致，请检查")
            ax, lat, src = spacing_um(hdr, vendor)
            v, how = display_transform(vol, vendor)
            vid = volume_id(oct_path)
            patient = f"RETOUCH_{vendor}_{vid}"
            vrows.append({"patient": patient, "vendor": vendor, "path": oct_path, "n_bscan": vol.shape[0],
                          "depth_px": vol.shape[1], "width_px": vol.shape[2], "axial_um": ax, "lateral_um": lat,
                          "spacing_source": src, "dtype": str(vol.dtype), "gray": how,
                          "irf_bscans": int((lab == 1).reshape(lab.shape[0], -1).any(1).sum())})
            for k in range(0, vol.shape[0], every):
                im = v[k]
                if vendor == "Spectralis":
                    im = fill_padding_bscan(im)
                im = U.resample(im, ax, lat)
                lb = resample_label(lab[k], ax, lat)
                irf = (lb == 1).astype(np.uint8)
                cid = f"{patient}_{k:03d}"
                U.save_case(img_dir, cid, im, irf, gt_dir)
                if k % QC_EVERY == 0 or (irf.any() and k % 2 == 0 and k % QC_EVERY == 2):
                    Image.fromarray(qc_image(im, lb)).save(os.path.join(qc_dir, f"{cid}.png"))
                rows.append({"case": cid, "patient": patient, "vendor": vendor, "volume": vid, "bscan": k,
                             "n_bscan": vol.shape[0], "has_irf": int(irf.any()), "has_srf": int((lb == 2).any()),
                             "has_ped": int((lb == 3).any()), "irf_mm2": U.area_mm2(irf),
                             "orig_h": vol.shape[1], "orig_w": vol.shape[2], "h": im.shape[0], "w": im.shape[1]})
                n_img += 1; n_irf += int(irf.any())
        print(f"[{vendor}] {len(cases)} 个容积 → {n_img} 幅 B-scan（含 IRF {n_irf} 幅）→ {img_dir}")
    t = pd.DataFrame(rows)
    cases_csv, vols_csv = U.p("tables", f"retouch_cases{table_suffix}.csv"), U.p("tables", f"retouch_volumes{table_suffix}.csv")
    t.to_csv(cases_csv, index=False)
    pd.DataFrame(vrows).to_csv(vols_csv, index=False)
    if len(t):
        s = t.groupby("vendor").agg(volumes=("patient", "nunique"), bscans=("case", "size"), irf_pos=("has_irf", "sum"),
                                    irf_volumes=("patient", lambda p: t.loc[p.index].groupby("patient").has_irf.max().sum()))
        print("\n汇总：\n" + s.to_string())
        print(f"\n→ {cases_csv}\n→ {vols_csv}")
        print("下一步：逐家设备打开 qc 文件夹看几张图，确认视网膜方向正确、绿色轮廓落在视网膜内的囊腔上。")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="只打印文件头和标签取值，不写文件")
    ap.add_argument("--every", type=int, default=1, help="每隔几张取 1 张 B-scan（默认 1 = 全部）")
    ap.add_argument("--spectralis-nodisp", action="store_true",
                    help="S7：Spectralis 另转一份、不做显示变换，输出到 retouch_spectralis_nodisp")
    a = ap.parse_args(argv)
    vols = find_volumes()
    if not any(vols.values()):
        raise SystemExit(f"在 {RAW_RETOUCH} 下没有找到任何 oct.mhd。请确认已解压，且子文件夹名含 Cirrus / Spectralis / Topcon。")
    if a.inspect:
        inspect(vols)
    elif a.spectralis_nodisp:
        return convert_nodisp(vols, a.every)
    else:
        convert(vols, a.every)


def convert_nodisp(vols, every=1):
    """SAP 敏感性分析 S7：Spectralis 不做 1/4 次方显示变换。只在主转换判定为线性强度（gray 以 pow0.25 开头）时才做。"""
    global SPECTRALIS_DISPLAY
    vt_path = U.p("tables", "retouch_volumes.csv")
    if not os.path.exists(vt_path):
        raise SystemExit("请先运行一次全部转换（python s13_retouch.py），再做 --spectralis-nodisp")
    vt = pd.read_csv(vt_path)
    spe = vt[vt.vendor == "Spectralis"]
    if spe.empty or not spe.gray.astype(str).str.startswith("pow0.25").any():
        print("S7 不适用：主转换中 Spectralis 没有做显示变换（原始数据不是线性强度），无需另转。")
        return None
    old = SPECTRALIS_DISPLAY
    SPECTRALIS_DISPLAY = "off"
    try:
        print("S7：Spectralis 不做显示变换，另存到 retouch_spectralis_nodisp（主转换结果不变）")
        return convert({"Spectralis": vols.get("Spectralis", [])}, every, suffix="_nodisp", table_suffix="_spectralis_nodisp")
    finally:
        SPECTRALIS_DISPLAY = old


if __name__ == "__main__":
    main()
