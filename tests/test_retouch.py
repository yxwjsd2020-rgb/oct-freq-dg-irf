# -*- coding: utf-8 -*-
"""s13_retouch.py 自测：用模拟的三家设备 .mhd 容积跑一遍 --inspect 和转换，检查输出。不需要真实数据。
用法：python tests/test_retouch.py   （应显示“全部通过”）"""
import os
import sys
import zlib
import tempfile
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = tempfile.mkdtemp(prefix="retouch_test_")
os.environ["RAW_RETOUCH"] = os.path.join(TMP, "raw")
os.environ["DME_WORK"] = os.path.join(TMP, "work")
sys.path.insert(0, ROOT)


def write_mhd(path, arr, spacing_mm, etype, compress=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = np.ascontiguousarray(arr).tobytes()
    if compress:
        raw = zlib.compress(raw)
    rawname = os.path.basename(path).replace(".mhd", ".zraw" if compress else ".raw")
    open(os.path.join(os.path.dirname(path), rawname), "wb").write(raw)
    z, y, x = arr.shape
    hdr = ["ObjectType = Image", "NDims = 3", "BinaryData = True", "BinaryDataByteOrderMSB = False",
           f"CompressedData = {'True' if compress else 'False'}",
           f"DimSize = {x} {y} {z}", f"ElementSpacing = {' '.join(str(s) for s in spacing_mm)}",
           f"ElementType = {etype}", f"ElementDataFile = {rawname}"]
    open(path, "w").write("\n".join(hdr) + "\n")


def fake_volume(z, y, x, rng, linear=False):
    vol = rng.normal(30, 5, (z, y, x))
    top, bot = int(y * 0.3), int(y * 0.6)
    vol[:, top:bot] += 80                            # “视网膜”
    vol[:, bot - 4:bot] += 120                       # “RPE”
    lab = np.zeros((z, y, x), np.uint8)
    for k in range(z // 3, 2 * z // 3):              # 中间三分之一的 B-scan 有 IRF
        cy, cx = (top + bot) // 2, x // 2
        lab[k, cy - y // 20: cy + y // 20, cx - x // 10: cx + x // 10] = 1
        vol[k][lab[k] == 1] = 35
    lab[z // 2, bot - 8: bot - 4, : x // 8] = 2      # SRF
    lab[z // 2, bot: bot + 4, -x // 8:] = 3          # PED
    vol = np.clip(vol, 1, 255)
    if linear:
        vol = (vol / 255.0) ** 4 * 60000             # 模拟线性强度（显示前需 1/4 次方）
    return vol, lab


def main():
    rng = np.random.default_rng(0)
    specs = {   # 设备: (容积数, z, y, x, 间距 mm (x y z), 类型, 线性强度, 压缩)
        "Cirrus": (2, 8, 256, 128, (0.01172, 0.00195, 0.047), "MET_UCHAR", False, False),
        "Spectralis": (2, 6, 124, 128, (0.0113, 0.00387, 0.12), "MET_USHORT", True, True),
        "Topcon": (1, 8, 221, 128, (0.01172, 0.0026, 0.047), "MET_UCHAR", False, False),
    }
    for vendor, (n, z, y, x, sp, et, lin, cmp_) in specs.items():
        for i in range(n):
            vol, lab = fake_volume(z, y, x, rng, lin)
            d = os.path.join(os.environ["RAW_RETOUCH"], f"RETOUCH-TrainingSet-{vendor}", f"TRAIN{i + 1 + 10 * list(specs).index(vendor):03d}")
            write_mhd(os.path.join(d, "oct.mhd"), vol.astype(np.uint16 if et == "MET_USHORT" else np.uint8), sp, et, cmp_)
            write_mhd(os.path.join(d, "reference.mhd"), lab, sp, "MET_UCHAR")
    # 第 1 个 Topcon 容积额外放一个文件头间距错误的副本 → 应回退到标称值
    vol, lab = fake_volume(8, 221, 128, rng)
    d = os.path.join(os.environ["RAW_RETOUCH"], "RETOUCH-TrainingSet-Topcon", "TRAIN099")
    write_mhd(os.path.join(d, "oct.mhd"), vol.astype(np.uint8), (1, 1, 1), "MET_UCHAR")
    write_mhd(os.path.join(d, "reference.mhd"), lab, (1, 1, 1), "MET_UCHAR")

    import s13_retouch as R
    import config as C
    import pandas as pd
    from PIL import Image

    vols = R.find_volumes()
    assert [len(vols[v]) for v in R.VENDORS] == [2, 2, 2], vols
    a, hdr = R.read_mhd(vols["Spectralis"][0][0])
    assert a.shape == (6, 124, 128) and a.dtype == np.uint16, (a.shape, a.dtype)
    R.inspect(vols)
    t = R.convert(vols)

    # 1. 数量与命名
    assert len(t) == 2 * 8 + 2 * 6 + 2 * 8, len(t)
    assert set(t.vendor) == set(R.VENDORS)
    assert t.case.str.match(r"^RETOUCH_(Cirrus|Spectralis|Topcon)_\d{3}_\d{3}$").all()
    # 2. 病人编号能被 s12_freq_eval 的正则正确解析
    import re
    pat = re.compile(r"(DUKE_Subject_\d+|TP_DME\d+|RETOUCH_\w+?_\d+)")
    assert all(pat.match(c).group(1) == p for c, p in zip(t.case, t.patient)), "病人编号解析错误"
    # 3. 只保留 IRF：每个容积中间三分之一有 IRF；SRF/PED 记录在表里但不进掩膜
    for p, g in t.groupby("patient"):
        z = g.n_bscan.iloc[0]
        exp = [(z // 3) <= k < (2 * z // 3) for k in g.bscan]
        assert (g.has_irf.astype(bool).tolist() == exp), (p, g.has_irf.tolist(), exp)
    assert t.has_srf.sum() == 6 and t.has_ped.sum() == 6
    # 4. 文件存在，掩膜只含 0/1，图像与掩膜同尺寸，且已重采样到 3.9 × 11.7 μm
    vt = pd.read_csv(os.path.join(os.environ["DME_WORK"], "tables", "retouch_volumes.csv"))
    for _, r in t.iterrows():
        key = f"retouch_{r.vendor.lower()}"
        im = np.asarray(Image.open(os.path.join(C.WORK, key, "images", f"{r.case}_0000.png")))
        gt = np.asarray(Image.open(os.path.join(C.WORK, key, "gt", f"{r.case}.png")))
        assert im.shape == gt.shape and set(np.unique(gt)) <= {0, 1}
        v = vt[vt.patient == r.patient].iloc[0]
        assert abs(im.shape[0] - r.orig_h * v.axial_um / C.TARGET_AXIAL_UM) <= 1
        assert abs(im.shape[1] - r.orig_w * v.lateral_um / C.TARGET_LATERAL_UM) <= 1
        if r.has_irf:
            assert gt.sum() > 0
    # 5. 间距与灰度处理
    assert (vt[vt.vendor == "Cirrus"].spacing_source == "header").all()
    bad = vt[vt.path.str.contains("TRAIN099")].iloc[0]
    assert bad.spacing_source == "nominal" and abs(bad.axial_um - 2300 / 885) < 1e-6
    assert vt[vt.vendor == "Spectralis"].gray.str.startswith("pow0.25").all(), vt.gray.tolist()
    assert (vt[vt.vendor != "Spectralis"].gray == "none").all()
    # 6. 质控图
    for v in R.VENDORS:
        assert len(os.listdir(os.path.join(C.WORK, f"retouch_{v.lower()}", "qc"))) > 0
    # 7. --every 2
    t2 = R.convert(vols, every=2)
    assert len(t2) == 2 * 4 + 2 * 3 + 2 * 4
    n_files = sum(len(os.listdir(os.path.join(C.WORK, f"retouch_{v.lower()}", "images"))) for v in R.VENDORS)
    assert n_files == len(t2), ("旧文件未清空", n_files, len(t2))
    # 8. S7：Spectralis 不做显示变换，另存一份；主转换输出和表格不变
    main_imgs = sorted(os.listdir(os.path.join(C.WORK, "retouch_spectralis", "images")))
    main_tab = pd.read_csv(os.path.join(C.WORK, "tables", "retouch_cases.csv"))
    t3 = R.main(["--spectralis-nodisp"])
    assert t3 is not None and len(t3) == 2 * 6 and set(t3.vendor) == {"Spectralis"}, None if t3 is None else len(t3)
    nd = os.path.join(C.WORK, "retouch_spectralis_nodisp")
    assert len(os.listdir(os.path.join(nd, "images"))) == 12 and len(os.listdir(os.path.join(nd, "gt"))) == 12
    vnd = pd.read_csv(os.path.join(C.WORK, "tables", "retouch_volumes_spectralis_nodisp.csv"))
    assert (vnd.gray == "none").all(), vnd.gray.tolist()
    assert sorted(os.listdir(os.path.join(C.WORK, "retouch_spectralis", "images"))) == main_imgs
    assert len(pd.read_csv(os.path.join(C.WORK, "tables", "retouch_cases.csv"))) == len(main_tab)
    assert R.SPECTRALIS_DISPLAY != "off" or os.environ.get("RETOUCH_SPECTRALIS_DISPLAY") == "off"   # 全局开关已恢复
    a1 = np.asarray(Image.open(os.path.join(C.WORK, "retouch_spectralis", "images", main_imgs[0])), float)
    a2 = np.asarray(Image.open(os.path.join(nd, "images", main_imgs[0])), float)
    assert a1.shape == a2.shape and np.abs(a1 - a2).mean() > 1, "nodisp 图像应与显示变换后的图像不同"
    # 9. Spectralis 若本来不是线性强度，S7 不适用
    vt2 = vt.copy(); vt2["gray"] = "none"
    vt2.to_csv(os.path.join(C.WORK, "tables", "retouch_volumes.csv"), index=False)
    assert R.main(["--spectralis-nodisp"]) is None
    print("\n全部通过")


if __name__ == "__main__":
    main()
