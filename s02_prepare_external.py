# -*- coding: utf-8 -*-
"""
第 2 步：外部测试数据 → 推理用输入
-----------------------------------------------------------------
  Duke DME（Chiu 2015，Subject_XX.mat）：取两位专家都标注过的 B-scan，裁剪到标注覆盖的列，
      输出 work/duke/images（模型输入）、gt1（专家1，参考标准）、gt2（专家2，用于人类一致性）
  OCTDL 正常图像：输出 work/octdl/images（只用于计算假阳性率）
用法：python s02_prepare_external.py
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import config as C
import common as U


def loadmat(path):
    try:
        from scipy.io import loadmat as lm
        return lm(path)
    except NotImplementedError:              # v7.3 格式
        import h5py
        f = h5py.File(path, "r")
        return {k: np.array(f[k]).T for k in f.keys()}


def fill_padding(img):
    """Spectralis 配准后图像上下边缘的纯白（或纯黑）填充块 → 用玻璃体背景灰度替换，避免模型把它当成结构"""
    a = img.astype(np.float32).copy()
    for val in (a.max(),):                             # 只处理纯白填充（顶部纯黑多为正常玻璃体）
        sat = a == val
        pad = np.cumprod(sat, 0).astype(bool) | np.cumprod(sat[::-1], 0)[::-1].astype(bool)
        if pad.any() and (~pad).any():
            bg = np.median(a[: max(5, a.shape[0] // 10)][~pad[: max(5, a.shape[0] // 10)]]) if (~pad[: max(5, a.shape[0] // 10)]).any() else np.median(a[~pad])
            a[pad] = bg
    return a


def duke():
    files = sorted(glob.glob(os.path.join(glob.escape(C.RAW_DUKE), "*.mat")))
    if not files:
        print(f"[Duke] 在 {C.RAW_DUKE} 没有找到 .mat 文件，跳过"); return
    lat = C.DUKE_LATERAL_UM or 11.3
    if not C.DUKE_LATERAL_UM:
        print("[Duke] config 中未填写 DUKE_LATERAL_UM，暂按 11.3 μm（Spectralis 30° 扫描 768 A-scan 的常见值），请在方法中说明")
    rows = []; fl_all = fl_below = 0
    for f in files:
        d = loadmat(f)
        img, g1, g2 = d["images"], d["manualFluid1"].astype(float), d["manualFluid2"].astype(float)
        lay = d.get("manualLayers1")
        subj = os.path.splitext(os.path.basename(f))[0]
        for k in range(img.shape[2]):
            v = ~np.isnan(g1[:, :, k]) & ~np.isnan(g2[:, :, k])
            if not v.any():
                continue
            cols = np.where(v.any(0))[0]
            sl = slice(cols.min(), cols.max() + 1)
            m1raw = np.nan_to_num(g1[:, sl, k]) > 0
            if lay is not None:                           # 统计位于 ONL/IS 边界（第 6 条手工分层线，≈ELM）以下的积液比例（≈SRF）
                l6 = lay[5, sl, k].astype(float)
                rr = np.arange(m1raw.shape[0])[:, None]
                below = m1raw & (rr > np.where(np.isnan(l6), 1e9, l6 - 1)[None, :])
                fl_all += m1raw.sum(); fl_below += below.sum()
            im = U.resample(fill_padding(img[:, sl, k]), C.DUKE_AXIAL_UM, lat)
            m1 = U.resample(m1raw.astype(np.uint8), C.DUKE_AXIAL_UM, lat, True)
            m2 = U.resample((np.nan_to_num(g2[:, sl, k]) > 0).astype(np.uint8), C.DUKE_AXIAL_UM, lat, True)
            cid = f"DUKE_{subj}_{k:02d}"
            U.save_case(U.p("duke", "images"), cid, im, m1, U.p("duke", "gt1"))
            Image.fromarray(m2).save(os.path.join(U.p("duke", "gt2"), f"{cid}.png"))
            qc(U.p("duke", "qc"), cid, im, m1, m2)
            rows.append({"case": cid, "patient": subj, "has_irf": int(m1.any())})
    t = pd.DataFrame(rows)
    t.to_csv(U.p("tables", "duke_cases.csv"), index=False)
    print(f"[Duke] {len(files)} 例，{len(t)} 幅 B-scan（专家1 标注含积液 {t.has_irf.sum()} 幅、不含 {len(t) - t.has_irf.sum()} 幅）")
    if fl_all:
        print(f"[Duke] 专家1 积液像素中位于 ELM 以下（≈视网膜下液）的比例：{fl_below / fl_all:.1%}（很小则可直接作为 IRF 参考标准）")


def qc(folder, cid, im, m1, m2=None):
    """质控图：绿 = 专家1（参考标准），红 = 专家2"""
    from scipy import ndimage as ndi
    g = U.to_uint8(im)
    rgb = np.stack([g] * 3, -1)
    for m, col in ((m2, (255, 0, 0)), (m1, (0, 255, 0))):
        if m is not None and m.any():
            mb = m.astype(bool)
            rgb[mb ^ ndi.binary_erosion(mb)] = col
    Image.fromarray(rgb).save(os.path.join(folder, f"{cid}.png"))


def octdl():
    """OCTDL 正常眼：JPG 为设备导出的截图，扫描长度不固定（原文：dynamic scan length），像素间距未知。
    OCTDL_RESAMPLE = "thickness"（默认）：用自动方法测视网膜 ILM→RPE 厚度（去掉中心凹附近），按正常值
        OCTDL_NORMAL_THICK_UM 推算本图的像素间距，整图等比例缩放（保持 JPG 的长宽比），再与其他数据一样重采样；
    OCTDL_RESAMPLE = "none"：不缩放（敏感性分析）。"""
    d = os.path.join(C.RAW_OCTDL, C.OCTDL_NORMAL_DIR)
    files = sorted(sum([glob.glob(os.path.join(glob.escape(d), "**", e), recursive=True) for e in ("*.jpg", "*.jpeg", "*.png")], []))
    if not files:
        print(f"[OCTDL] 在 {d} 没有找到图像，跳过"); return
    mode = getattr(C, "OCTDL_RESAMPLE", "thickness")
    if mode == "thickness":
        from s01b_axial_check import boundaries
    rows = []; skipped = []
    for f in files:
        im = U.read_gray(f)
        cid = "OCTDL_" + os.path.splitext(os.path.basename(f))[0].replace(" ", "_")
        info = {"case": cid, "patient": os.path.basename(f).split("_")[1] if "_" in os.path.basename(f) else cid,
                "has_irf": 0, "orig_h": im.shape[0], "orig_w": im.shape[1]}
        if mode == "thickness":
            ilm, rpe = boundaries(im)
            th = rpe - ilm
            ok = ~np.isnan(th)
            tpx0 = float(np.nanmedian(th)) if ok.any() else np.nan
            plausible = ok.sum() > im.shape[1] * 0.3 and 2.0 <= C.OCTDL_NORMAL_THICK_UM / tpx0 <= 15.0   # 推算间距须在 2–15 μm/像素
            if plausible:
                fov = int(np.nanargmin(np.where(ok, th, np.nan)))       # 最薄处 ≈ 中心凹
                w = im.shape[1]
                keep = ok & (np.abs(np.arange(w) - fov) > 0.15 * w)
                tpx = float(np.median(th[keep])) if keep.sum() > 20 else float(np.median(th[ok]))
                um = C.OCTDL_NORMAL_THICK_UM / tpx                       # 推算的像素间距（μm/像素，横纵相同）
                im = U.resample(im, um, um * C.OCTDL_LATERAL_RATIO)
                info.update({"thick_px": tpx, "est_um_per_px": um})
            else:
                print(f"[OCTDL] {cid}：未能可靠识别视网膜，无法确定尺度，不纳入（请在结果中报告排除数）")
                skipped.append(cid)
                continue
        elif C.OCTDL_SCAN_MM and C.OCTDL_AXIAL_UM:
            im = U.resample(im, C.OCTDL_AXIAL_UM, C.OCTDL_SCAN_MM * 1000 / im.shape[1])
        U.save_case(U.p("octdl", "images"), cid, im)
        rows.append(info)
    t = pd.DataFrame(rows, columns=["case", "patient", "has_irf", "orig_h", "orig_w", "thick_px", "est_um_per_px"] if mode == "thickness" else None)
    t.to_csv(U.p("tables", "octdl_cases.csv"), index=False)
    msg = f"[OCTDL] 正常图像 {len(t)} 幅，{t.patient.nunique()} 人" + (f"（另有 {len(skipped)} 幅无法确定尺度未纳入，名单见 tables/octdl_excluded.txt）" if skipped else "")
    if skipped:
        with open(U.p("tables", "octdl_excluded.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(skipped))
    if "est_um_per_px" in t:
        msg += f"；按厚度推算的像素间距中位数 {t.est_um_per_px.median():.2f} μm（四分位 {t.est_um_per_px.quantile(.25):.2f}–{t.est_um_per_px.quantile(.75):.2f}）"
    print(msg)


if __name__ == "__main__":
    duke()
    octdl()
