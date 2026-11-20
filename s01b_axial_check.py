# -*- coding: utf-8 -*-
"""
第 1b 步（可选，作者未回复时用）：用视网膜厚度粗略核对 TowardPi 的轴向像素间距
-----------------------------------------------------------------
思路：同一种自动方法，在三套数据的“中心凹旁 1.5–2.8 mm、无积液”的位置测量 ILM → RPE 的厚度。
  · Duke（Spectralis，轴向 3.87 μm 已知）和本院 RTVue（3.09 μm 已知）→ 厚度可换算成 μm，作为参照
  · TowardPi（轴向间距未知）→ 只能得到像素数
  · 推算的 TowardPi 轴向间距 ≈ 参照组厚度中位数（μm） ÷ TowardPi 厚度中位数（像素）
  · Duke 有专家手工分层，可顺便检验自动方法是否靠谱（自动 vs 手工）
注意：这只是“合理性核对”，人群、病情、设备不同都会带来误差（约 ±15%）；
     它能区分 4.16 μm 和 5.75 μm（相差 38%），但不能替代作者给出的数值。论文中写作敏感性分析的依据。

用法（在 dme_code 文件夹里）：
  python s01b_axial_check.py
结果：work/axial_check/summary.csv、各图测量明细 measurements.csv、质控图 work/axial_check/qc/*.png
     （质控图上绿线 = 自动找到的 ILM，红线 = RPE，黄色竖线之间 = 测量区，灰色 = 有积液不测）
"""
import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import config as C
import common as U

OUT = U.p("axial_check")
QC = U.p("axial_check", "qc")
RING_MM = (1.5, 2.8)        # 测量区：距中心凹 1.5–2.8 mm（避开中心凹本身和扫描边缘）
NEAR_FOVEA_MM = 0.6         # 只用距中心凹 ±0.6 mm 以内的 B-scan
DUKE_LAT_FALLBACK = 11.3    # Duke 横向间距未填写时的近似值（只用于确定测量区位置，对结果影响很小）
CANDIDATES = {"5.75（6 mm/1044，按空气中算）": 6000 / 1044, "4.16（按组织折射率 1.38 换算）": 6000 / 1.38 / 1044}
rows = []
qc_count = {}


# ------------------------------------------------------------ 自动找 ILM 与 RPE
def boundaries(img):
    """返回每一列的 ILM 行号、RPE 行号（像素）；找不到为 nan"""
    a = img.astype(np.float32)
    sat = a >= a.max() - 0.5                                              # Spectralis 配准后上下边缘用纯白填充：从顶/底连续的纯白像素当作背景
    pad = np.cumprod(sat, 0).astype(bool) | np.cumprod(sat[::-1], 0)[::-1].astype(bool)
    if pad.any():
        a[pad] = np.median(a[~pad]) if (~pad).any() else 0
    lo, hi = np.percentile(a, [1, 99.8])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    g = ndi.gaussian_filter(a, (2, 4))
    h, w = g.shape
    top = int(h * 0.05)
    bg = np.median(g[:max(top, 5)])                                       # 顶部玻璃体背景
    colmax = g[top:].max(0)
    # 1) ILM：从上往下第一个连续 3 个像素超过“背景 + 30%×(该列最亮 − 背景)”的位置
    above = g > (bg + 0.3 * (colmax - bg))[None, :]
    above[:top] = False
    run = above[:-2] & above[1:-1] & above[2:]
    ilm = np.where(run.any(0), run.argmax(0), np.nan).astype(float)
    ok = ~np.isnan(ilm)
    if ok.sum() > 31:
        ilm[ok] = ndi.median_filter(ilm[ok], size=31, mode="nearest")
    # 2) RPE：先估计这张图视网膜的典型厚度 T（ILM 到最亮处距离的中位数），
    #    再在 ILM 下方 0.6T–2.5T 范围内找最亮处（避开比 RPE 还亮的神经纤维层；水肿处视网膜可达 2 倍以上厚）
    rows = np.arange(h)[:, None]
    il = np.nan_to_num(ilm, nan=h).astype(int)
    d0 = np.where(rows > il[None, :] + int(0.06 * h), g, -1).argmax(0) - il    # 跳过 ILM 下方紧邻的神经纤维层（图高的 6%）
    good = ok & (d0 > 0)
    T = np.median(d0[good]) if good.any() else h / 4
    win = (rows >= (il + 0.6 * T)[None, :]) & (rows <= (il + 2.5 * T)[None, :])
    rpe = np.where(win, g, -1).argmax(0).astype(float)
    rpe = ndi.median_filter(rpe, size=31, mode="nearest")
    bad = ~ok | (rpe - ilm < 5)
    ilm[bad] = np.nan
    return ilm, rpe


def measure(dataset, case, img, axial_um, lateral_um, center_px, fluid=None, manual=None):
    """在测量区内、无积液的列上测厚度（像素），记录中位数；manual=(ilm, bm) 为手工分层（行号）"""
    ilm, rpe = boundaries(img)
    w = img.shape[1]
    ecc = np.abs(np.arange(w) - center_px) * lateral_um / 1000
    use = (ecc >= RING_MM[0]) & (ecc <= RING_MM[1]) & ~np.isnan(ilm)
    if fluid is not None:
        wet = ndi.binary_dilation(fluid.any(0), iterations=10)           # 该列任何位置有积液 → 整列不测（两侧再让 10 列）
        use &= ~wet
    else:
        wet = np.zeros(w, bool)
    if use.sum() < 20:
        return
    th = rpe[use] - ilm[use]
    r = {"dataset": dataset, "case": case, "axial_um": axial_um, "n_cols": int(use.sum()),
         "thick_px": float(np.median(th)), "thick_um": float(np.median(th) * axial_um) if axial_um else np.nan}
    if manual is not None:
        mi, mb = manual
        m = use & ~np.isnan(mi) & ~np.isnan(mb)
        if m.sum() >= 20:
            r["manual_um"] = float(np.median(mb[m] - mi[m]) * axial_um)
            r["auto_same_cols_um"] = float(np.median(rpe[m] - ilm[m]) * axial_um)
    rows.append(r)
    subj = re.sub(r"_\d+$", "", case)
    if subj not in qc_count and len(qc_count) < 80:                       # 每只眼/每例存 1 张质控图
        qc_count[subj] = 1
        g = U.to_uint8(img)
        rgb = np.stack([g] * 3, -1)
        rgb[:, wet] = (rgb[:, wet] * 0.4).astype(np.uint8)
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        for arr, col in ((ilm, (0, 255, 0)), (rpe, (255, 0, 0))):
            pts = [(x, float(arr[x])) for x in range(w) if not np.isnan(arr[x])]
            d.point(pts, fill=col)
        for s in (-1, 1):
            for e in RING_MM:
                x = center_px + s * e * 1000 / lateral_um
                if 0 <= x < w:
                    d.line([(x, 0), (x, img.shape[0])], fill=(255, 220, 0))
        im.save(os.path.join(QC, f"{dataset}_{case}.png"))


# ------------------------------------------------------------ TowardPi
def towardpi():
    try:
        import tifffile
    except ImportError:
        print("[TowardPi] 需要 tifffile：pip install tifffile"); return
    inp = os.path.join(C.RAW_TOWARDPI, C.TP_INPUT_DIR)
    lab = os.path.join(C.RAW_TOWARDPI, C.TP_LABEL_DIR)
    files = sorted(glob.glob(os.path.join(glob.escape(inp), "*.tif*")))
    if not files:
        print(f"[TowardPi] 在 {inp} 没有找到 .tif，跳过"); return
    n = 0
    for f in files:
        stem = os.path.basename(f).split(".")[0]
        mf = glob.glob(os.path.join(glob.escape(lab), stem + C.TP_MASK_SUFFIX + ".tif*"))
        with tifffile.TiffFile(f) as t:
            npg = len(t.pages)
            if C.TP_BSCAN_AXIS != 0:
                print("[TowardPi] 本脚本只支持 TP_BSCAN_AXIS = 0（每页一张 B-scan）"); return
            half = int(NEAR_FOVEA_MM * 1000 / C.TP_LATERAL_UM)            # B-scan 间距与横向相同（6 mm/512）
            ks = range(max(0, npg // 2 - half), min(npg, npg // 2 + half + 1), 10)
            imgs = {k: t.pages[k].asarray() for k in ks}
        masks = {}
        if mf:
            with tifffile.TiffFile(mf[0]) as t:
                if len(t.pages) == npg:
                    masks = {k: t.pages[k].asarray() > 0 for k in ks}
        for k, im in imgs.items():
            if im.ndim == 3:
                im = im[..., 0]
            m = masks.get(k)
            if m is not None and m.ndim == 3:
                m = m.any(-1)
            if C.TP_DEPTH_AXIS_IN_BSCAN == 1:
                im = im.T; m = None if m is None else m.T
            measure("TowardPi", f"{stem}_{k:03d}", im, None, C.TP_LATERAL_UM, im.shape[1] / 2, fluid=m)
        n += 1
    print(f"[TowardPi] {n} 个容积")


# ------------------------------------------------------------ Duke
def duke():
    from s02_prepare_external import loadmat
    files = sorted(glob.glob(os.path.join(glob.escape(C.RAW_DUKE), "*.mat")))
    if not files:
        print(f"[Duke] 在 {C.RAW_DUKE} 没有找到 .mat，跳过（没有 Duke 时只用本院数据作参照）"); return
    lat = C.DUKE_LATERAL_UM or DUKE_LAT_FALLBACK
    for f in files:
        d = loadmat(f)
        img = d["images"]
        lay = d.get("manualLayers1")
        flu = d.get("manualFluid1")
        nb = img.shape[2]
        c = nb // 2
        bs_um = 120.0                                                     # Spectralis 61 张 B-scan 间距约 120 μm
        half = int(NEAR_FOVEA_MM * 1000 / bs_um)
        for k in range(c - half, c + half + 1):
            manual = None
            if lay is not None and not np.all(np.isnan(lay[:, :, k].astype(float))):
                L = lay[:, :, k].astype(float)
                L[L <= 0] = np.nan
                manual = (L[0] - 1, L[-1] - 1)                            # 第 1 条 = ILM，最后 1 条 = Bruch 膜（MATLAB 行号从 1 开始）
            fl = None
            if flu is not None:
                fk = flu[:, :, k].astype(float)
                if np.all(np.isnan(fk)):                                  # 只用专家标注过积液的 B-scan（每例 11 张）
                    continue
                fl = np.nan_to_num(fk) > 0
                fl[:, np.isnan(fk).all(0)] = True                         # 未标注的列当作“不确定”，不测
            measure("Duke", f"{os.path.splitext(os.path.basename(f))[0]}_{k:02d}", img[:, :, k].astype(np.float32),
                    C.DUKE_AXIAL_UM, lat, img.shape[1] / 2, fluid=fl, manual=manual)
    print(f"[Duke] {len(files)} 例")


# ------------------------------------------------------------ 本院 Raster（已选好中心凹的 B-scan）
def local():
    sel_f = U.p("raster_select.xlsx")
    if not os.path.exists(sel_f):
        print("[本院] 没有 raster_select.xlsx，跳过"); return
    t = pd.read_excel(sel_f, dtype=str).fillna("")
    t = t[(t["exclude"] != "1") & (t["fovea_slice"] != "")]
    n = 0
    for _, r in t.iterrows():
        k = int(float(r["fovea_slice"])) - 1
        cand = glob.glob(os.path.join(glob.escape(r["dir"]), f"B{k:02d}*.png"))
        if not cand:
            continue
        im = U.read_gray(cand[0])
        width_mm = float(r.get("width_mm") or C.LOCAL_RASTER_MM)
        lat = width_mm * 1000 / im.shape[1]
        fx = float(r["fovea_x_mm"]) if r["fovea_x_mm"] else width_mm / 2
        measure("本院RTVue", r["study_id"], im, C.LOCAL_AXIAL_UM, lat, fx * 1000 / lat)   # 本院图无积液掩膜：囊腔、增殖膜落在测量区会使厚度偏大，质控图上请自己看
        n += 1
    print(f"[本院] {n} 只眼")


def main():
    towardpi(); duke(); local()
    if not rows:
        raise SystemExit("没有测到任何数据，请检查 config 中的路径")
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, "measurements.csv"), index=False, encoding="utf-8-sig")
    # 先在每只眼/每例内取中位数，再在数据集间比较（避免某一例 B-scan 多而权重大）
    m["subject"] = m["case"].str.replace(r"_\d+$", "", regex=True)
    per = m.groupby(["dataset", "subject"]).agg(thick_px=("thick_px", "median"), thick_um=("thick_um", "median")).reset_index()
    summ = per.groupby("dataset").agg(n=("subject", "size"), px_median=("thick_px", "median"),
                                      px_q1=("thick_px", lambda s: s.quantile(.25)), px_q3=("thick_px", lambda s: s.quantile(.75)),
                                      um_median=("thick_um", "median")).reset_index()
    lines = ["", "=== ILM→RPE 厚度（中心凹旁 1.5–2.8 mm，无积液处；先按眼取中位数）==="]
    for _, s in summ.iterrows():
        um = f"{s.um_median:.0f} μm" if not np.isnan(s.um_median) else "μm 未知"
        lines.append(f"  {s.dataset:<10} n={s.n:<3} 中位数 {s.px_median:.1f} 像素（四分位 {s.px_q1:.1f}–{s.px_q3:.1f}）→ {um}")
    if "manual_um" in m:
        v = m.dropna(subset=["manual_um"])
        if len(v):
            diff = v["auto_same_cols_um"] - v["manual_um"]
            lines.append(f"\n  自动方法检验（Duke {len(v)} 张，同一批列）：自动 − 手工 = {diff.median():+.0f} μm（中位数），"
                         f"相关系数 r = {np.corrcoef(v['auto_same_cols_um'], v['manual_um'])[0, 1]:.2f}")
            lines.append("  （自动方法量到 RPE 最亮处，手工量到 Bruch 膜，所以自动略小是正常的；两套参照都用自动方法，偏差相互抵消）")
    tp = summ[summ.dataset == "TowardPi"]
    refs = summ[summ.dataset != "TowardPi"].dropna(subset=["um_median"])
    if len(tp) and len(refs):
        tpx = float(tp.px_median.iloc[0])
        lines.append("\n=== 推算 TowardPi 轴向间距 = 参照厚度(μm) ÷ TowardPi 厚度(像素) ===")
        for _, s in refs.iterrows():
            lines.append(f"  以 {s.dataset} 为参照（n={s.n}）：{s.um_median:.0f} ÷ {tpx:.1f} = {s.um_median / tpx:.2f} μm/像素")
        refs = refs.assign(_duke=(refs.dataset == "Duke").astype(int))
        best = refs.sort_values(["_duke", "n"]).iloc[-1] if (refs.dataset == "Duke").any() else refs.sort_values("n").iloc[-1]   # 有 Duke 时以 Duke 为主
        est = best.um_median / tpx
        lines.append("\n  两个候选值下 TowardPi 的厚度：" + "；".join(f"{k} → {tpx * v:.0f} μm" for k, v in CANDIDATES.items()))
        near = min(CANDIDATES, key=lambda k: abs(CANDIDATES[k] - est))
        if min(abs(v - est) / v for v in CANDIDATES.values()) > 0.15:
            lines.append(f"  → 以 {best.dataset} 推算的 {est:.2f} μm/像素，与两个候选值都相差 15% 以上："
                         f"提示原始 B-scan 的深度范围可能不是 6 mm（例如约 {est * 1044 / 1000:.1f} mm），请把结果发给负责人判断")
        else:
            lines.append(f"  → 以 {best.dataset} 推算的 {est:.2f} μm/像素，与候选值 {near} 更接近")
        if best.n < 5:
            lines.append(f"  [注意] 参照组只有 {best.n} 例，结论很不稳定")
    lines.append(f"\n质控图：{QC}（请抽看几张：绿线应贴在视网膜内表面，红线应在 RPE 亮带上）")
    txt = "\n".join(lines)
    print(txt)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
