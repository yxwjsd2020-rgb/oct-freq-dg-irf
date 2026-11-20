# -*- coding: utf-8 -*-
"""
第 12 步（文章D）：频域模块各配置的内部验证与外部测试分析 —— v0.3（按 SAP v0.1 实现）
---------------------------------------------------------------------------------------
SAP：Claude Docs《文章D 预注册与统计分析计划 v0.1》https://claude.ai/artifact/Mjj4DVaQF8P6S4zsbanMEE
实现内容：
  主要  H1  RETOUCH 容积 IRF Dice（种子平均），RAS vs 基线，配对 Wilcoxon（双侧）+ 聚类 bootstrap 95% CI + Hodges–Lehmann
  次要  H2  容积静默失败率（含 IRF 图预测为空的比例），同法；H1 成立才判定
        H3  TowardPi 第 0 折内部眼 IRF Dice 非劣效（t 分布 95% CI 下限 > −界值），另报以眼聚类的 bootstrap；H2 成立才判定
  探索  其余频域配置 vs 基线（Holm 校正）、RAS_WTConv vs RAS、PDU-Net（开题网络重新实现）vs 基线 / RAS / 内部（不进 Holm 族）、分设备与设备级报警、按 SRF/PED/IRF 面积分层、
        Duke / 本院 / OCTDL 假阳性（Wilson CI）
  敏感  S1 逐幅等权 · S2 纳入不含 IRF 图 · S3 IRF 面积 ≥ 阈值 · S4 线性混合模型（statsmodels，缺失则跳过）·
        S5 逐种子 3×3 配对 · S6 基线只用 FreqBase 种子 2、3 · S7 Spectralis 不做显示变换（retouch_spectralis_nodisp）

尚待杨医生确认的 SAP 待定项都做成参数（见 DEFAULTS），默认值 = SAP v0.1 草稿值。改参数的三种方式（优先级从低到高）：
  1. 代码里的 DEFAULTS；2. work/tables/freq_sap_params.json；3. 命令行 --set 键=值（可多次）
  例：python s12_freq_eval.py --set NI_MARGIN=0.05 --set UNIT=bscan
实际使用的参数写入 results/freq/params_used.json，供审计。

输入：work/nnUNet_results*/.../fold_0/validation（内部）、work/pred/freq/<训练器>_s<种子>/<测试集>（外部）
输出：work/results/freq/  metrics_long.csv、hypotheses.csv、exploratory.csv、devices.csv、strata.csv、
                          external_other.csv、sensitivity.csv、s5_seed_pairs.csv、params_used.json、report.md
用法：python s12_freq_eval.py [--set 键=值 ...]
"""
import os
import re
import sys
import json
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import config as C
import common as U
import s06_evaluate as E

VERSION = "0.3.1"
DEFAULTS = {
    # —— SAP 待定项（杨医生确认后改这里或 freq_sap_params.json）——
    "UNIT": "volume",          # 待定 1：主要分析单位。volume = 按容积/患者平均；bscan = 每幅 B-scan 为单位（以容积聚类 bootstrap）
    "MCID": 0.05,              # 待定 2：最小有意义 Dice 差值（只报告是否达到，不影响 H1 判定）
    "NI_MARGIN": 0.03,         # 待定 3：H3 内部非劣效界值（Dice）
    "PRIMARY": "retouch",      # 待定 4：主要外部测试。retouch = RETOUCH 三家设备；local = 备用方案（本院 Optovue 含 IRF 眼）
    "LOCAL_CONFIRMATORY": False,  # 待定 5：本院 Optovue 是否进入假设检验（草稿：否，只作探索；PRIMARY=local 时自动作为主要）
    # —— SAP 已定参数 ——
    "PRIMARY_CFG": "RAS",
    "BASE_CFG": "baseline",
    "ALPHA": 0.05,
    "N_BOOT": 10000,
    "BOOT_SEED": 2026,
    "MIN_IRF_MM2": 0.005,      # S3 阈值（约 110 像素 @ 3.9×11.7 μm）
    "ALARM": 0.30,             # 设备级报警：含 IRF 图预测为空的比例 > 30%（沿用文章B 登记值）
    "HL_MAX_PAIRS": 2_000_000,  # Hodges–Lehmann：单位数很多时随机抽样 Walsh 平均数的上限
}

TP = f"Dataset{C.DS_TOWARDPI:03d}_TowardPi"
BASE = "nnUNetTrainer_250epochs"          # 文章B 基线（种子 1）
BASE_FREQ = "nnUNetTrainer_FreqBase"      # 基线的其余种子
# 开题 PDU-Net 的重新实现：回答"开题网络 vs nnU-Net"，与频域消融不是同一族问题，单独报告、不进 Holm 校正
OUTSIDE_HOLM = ("PDUNet", "PDUNetSerial")
RETOUCH_SETS = ["retouch_cirrus", "retouch_spectralis", "retouch_topcon"]
NODISP = "retouch_spectralis_nodisp"
GT = {"duke": ("duke", "gt1"), "octdl": None, "local": ("local", "gt"),
      "retouch_cirrus": ("retouch_cirrus", "gt"), "retouch_spectralis": ("retouch_spectralis", "gt"),
      "retouch_topcon": ("retouch_topcon", "gt"), NODISP: (NODISP, "gt")}
META = {"duke": "duke_cases.csv", "octdl": "octdl_cases.csv", "local": "local_cases.csv",
        "retouch_cirrus": "retouch_cases.csv", "retouch_spectralis": "retouch_cases.csv",
        "retouch_topcon": "retouch_cases.csv", NODISP: "retouch_cases.csv"}
DEVICE = {"retouch_cirrus": "Cirrus", "retouch_spectralis": "Spectralis", "retouch_topcon": "Topcon",
          NODISP: "Spectralis (no display transform)", "duke": "Spectralis (Duke)", "local": "Optovue (local)",
          "octdl": "OCTDL normal", "internal": "TowardPi"}
OUT = None
NOTES = []


def note(msg):
    NOTES.append(msg)
    print("[注意] " + msg)


# ============================================================ 参数
def load_params(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖参数，可多次")
    a = ap.parse_args(argv)
    P = dict(DEFAULTS)
    f = U.p("tables", "freq_sap_params.json")
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            P.update(json.load(fh))
        print(f"读取参数文件 {f}")
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in DEFAULTS:
            raise SystemExit(f"未知参数 {k}；可用：{', '.join(DEFAULTS)}")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        P[k] = v
    if P["UNIT"] not in ("volume", "bscan"):
        raise SystemExit("UNIT 只能是 volume 或 bscan")
    if P["PRIMARY"] not in ("retouch", "local"):
        raise SystemExit("PRIMARY 只能是 retouch 或 local")
    return P


# ============================================================ 收集逐图指标
def config_name(trainer):
    return "baseline" if trainer in (BASE, BASE_FREQ) else trainer.replace("nnUNetTrainer_", "")


def patient_of(row):
    p = row.get("patient")
    if isinstance(p, str) and p:
        return p
    m = re.match(r"(DUKE_Subject_\d+|TP_DME\d+|RETOUCH_[A-Za-z]+_\w+?(?=_\d{3}$)|RETOUCH_\w+?_\d+)", str(row["case"]))
    return m.group(1) if m else str(row["case"])


def collect():
    E.ROWS.clear()
    tp_gt = os.path.join(U.ds_dir(C.DS_TOWARDPI, "TowardPi"), "labelsTr")
    tp_meta = E.meta_from("towardpi_cases.csv")
    vdirs = glob.glob(os.path.join(C.WORK, "nnUNet_results_freq", "seed*", TP, "*", "fold_0", "validation"))
    vdirs.append(os.path.join(C.NNUNET_RESULTS, TP, f"{BASE}__{C.PLANS}__{C.CONFIG}", "fold_0", "validation"))
    for v in vdirs:
        if not os.path.isdir(v):
            continue
        trainer = os.path.basename(os.path.dirname(os.path.dirname(v))).split("__")[0]
        sd = re.search(r"seed(\d+)", v)
        E.score("internal", v, tp_gt, tp_meta, condition="freq", trainer=trainer, seed=int(sd.group(1)) if sd else 1)
    # 注意：不能把通配符交给 U.p（它会 makedirs：Linux 上会建出名为 *_s* 的空文件夹，Windows 上直接报错）
    pred_root = os.path.join(C.WORK, "pred", "freq")
    for run_dir in sorted(d for d in glob.glob(os.path.join(glob.escape(pred_root), "*_s*")) if os.path.isdir(d)):
        name = os.path.basename(run_dir)
        if name.startswith("smoke"):
            continue
        trainer, seed = name.rsplit("_s", 1)
        for set_name, gt in GT.items():
            d = os.path.join(run_dir, set_name)
            if os.path.isdir(d):
                E.score(set_name, d, U.p(*gt) if gt else None, E.meta_from(META[set_name]),
                        condition="freq", trainer=trainer, seed=int(seed))
    df = pd.DataFrame(E.ROWS)
    if df.empty:
        raise SystemExit("没有找到任何结果：请先在 GPU 上运行 gpu/run_freq.sh train / predict")
    df["config"] = df["trainer"].map(config_name)
    df["patient_id"] = df.apply(patient_of, axis=1)
    df["device"] = df["set"].map(DEVICE).fillna(df["set"])
    df["empty"] = (df["n_pr"] == 0).astype(float)
    # 基线种子 1 = 文章B 模型；若另有 FreqBase 种子 1，按 SAP 以文章B 模型为准
    dup = (df.trainer == BASE_FREQ) & (df.seed == 1)
    if dup.any() and (df.trainer == BASE).any():
        note("同时存在文章B 基线与 FreqBase 种子 1：按 SAP 只用文章B 模型作基线种子 1，FreqBase 种子 1 不计入")
        df = df[~dup]
    # RETOUCH 病变属性（分层用）
    rc = U.p("tables", "retouch_cases.csv")
    if os.path.exists(rc):
        attr = pd.read_csv(rc, dtype={"case": str})
        keep = [c for c in ("case", "has_srf", "has_ped", "irf_mm2") if c in attr.columns]
        df = df.merge(attr[keep], on="case", how="left")
    return df.reset_index(drop=True)


# ============================================================ 统计工具
def hodges_lehmann(d, max_pairs, rng):
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return np.nan
    if n * (n + 1) / 2 <= max_pairs:
        i, j = np.triu_indices(n)
        return float(np.median((d[i] + d[j]) / 2))
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    return float(np.median((d[i] + d[j]) / 2))


def cluster_boot_mean(d, clusters, P):
    """以 clusters 为单位重抽样的百分位 bootstrap 95% CI（单位均值的均值）"""
    d = np.asarray(d, float)
    cl = pd.Series(clusters).astype(str).values
    g = pd.DataFrame({"d": d, "c": cl}).groupby("c")["d"].agg(["sum", "count"])
    sums, cnts = g["sum"].values, g["count"].values
    k = len(sums)
    if k < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(P["BOOT_SEED"])
    B = int(P["N_BOOT"])
    out = np.empty(B)
    step = max(1, 2_000_000 // k)          # 分块，避免 B×k 矩阵过大
    for s in range(0, B, step):
        e = min(B, s + step)
        idx = rng.integers(0, k, (e - s, k))
        out[s:e] = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def paired(a, b, clusters, P, label=""):
    """配对比较：a、b 为按单位对齐的 Series（同 index），clusters 为每个单位的聚类编号"""
    df = pd.DataFrame({"a": a, "b": b, "c": clusters}).dropna()
    n = len(df)
    r = {"comparison": label, "n_units": n, "n_clusters": df["c"].nunique() if n else 0,
         "mean_a": df["a"].mean() if n else np.nan, "mean_b": df["b"].mean() if n else np.nan}
    if n == 0:
        return {**r, "mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "hl": np.nan, "n_better": 0, "p": np.nan}
    d = (df["a"] - df["b"]).values
    lo, hi = cluster_boot_mean(d, df["c"].values, P)
    if n < 6:
        p = np.nan
    elif np.all(d == 0):
        p = 1.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(stats.wilcoxon(df["a"], df["b"]).pvalue)
    rng = np.random.default_rng(P["BOOT_SEED"])
    return {**r, "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "hl": hodges_lehmann(d, int(P["HL_MAX_PAIRS"]), rng),
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "p": p}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


# ============================================================ 单位表
def image_value(d, metric, P):
    """逐图取值。dice/empty 只用含 IRF 图；dice_all 用全部图（S2）；dice_big 只用 IRF 面积 ≥ 阈值的图（S3）"""
    if metric == "dice":
        x = d[d.has_irf == 1]
        return x.assign(v=x["dice"])
    if metric == "empty":
        x = d[d.has_irf == 1]
        return x.assign(v=x["empty"])
    if metric == "dice_all":
        v = np.where(d.has_irf == 1, d["dice"], np.where(d["n_pr"] == 0, 1.0, 0.0))
        return d.assign(v=v)
    if metric == "dice_big":
        x = d[(d.has_irf == 1) & (d["gt_area"] >= P["MIN_IRF_MM2"])]
        return x.assign(v=x["dice"])
    raise ValueError(metric)


def unit_table(d, metric, unit, P, seeds=None):
    """返回 (宽表：行 = 单位、列 = 配置，值 = 种子平均；单位 → 聚类 映射)"""
    x = image_value(d, metric, P)
    if seeds is not None:
        x = x[x.apply(lambda r: r["seed"] in seeds.get(r["config"], [r["seed"]]), axis=1)]
    if x.empty:
        return pd.DataFrame(), pd.Series(dtype=str)
    key = "patient_id" if unit == "volume" else "case"
    # bscan 单位跨设备可能重名（同名 case 不同 set）：加 set 前缀
    x = x.assign(u=x[key] if unit == "volume" else x["set"] + "/" + x["case"])
    per_seed = x.groupby(["config", "seed", "u"])["v"].mean()
    wide = per_seed.groupby(["config", "u"]).mean().unstack("config")
    clusters = x.drop_duplicates("u").set_index("u")["patient_id"].reindex(wide.index)
    return wide, clusters


def compare(d, metric, unit, cfg_a, cfg_b, P, label, seeds=None):
    w, cl = unit_table(d, metric, unit, P, seeds)
    if w.empty or cfg_a not in w or cfg_b not in w:
        return {"comparison": label, "n_units": 0, "note": f"缺少 {cfg_a} 或 {cfg_b} 的结果"}
    return paired(w[cfg_a], w[cfg_b], cl, P, label)


# ============================================================ 分析
def primary_sets(df, P):
    if P["PRIMARY"] == "local":
        return ["local"]
    return [s for s in RETOUCH_SETS if s in set(df["set"])]


def hypotheses(df, P):
    sets = primary_sets(df, P)
    A, B = P["PRIMARY_CFG"], P["BASE_CFG"]
    prim = df[df["set"].isin(sets) & df["config"].isin([A, B])]
    rows = []
    if not sets:
        note("主要外部测试集（%s）没有任何预测结果：H1、H2 无法检验" % P["PRIMARY"])
    h1 = compare(prim, "dice", P["UNIT"], A, B, P, f"H1 {A} vs {B}：{'+'.join(sets) or '—'} IRF Dice（{P['UNIT']}）")
    h1_pass = bool(h1.get("p", np.nan) < P["ALPHA"] and h1.get("mean_diff", 0) > 0)
    h1.update(hypothesis="H1", role="主要", tested=True, passed=h1_pass,
              mcid_reached=bool(h1.get("mean_diff", np.nan) >= P["MCID"]) if h1.get("n_units") else None)
    rows.append(h1)
    h2 = compare(prim, "empty", P["UNIT"], A, B, P, f"H2 {A} vs {B}：静默失败率（{P['UNIT']}）")
    h2_pass = bool(h1_pass and h2.get("p", np.nan) < P["ALPHA"] and h2.get("mean_diff", 0) < 0)
    h2.update(hypothesis="H2", role="次要（H1 成立才判定）", tested=h1_pass, passed=h2_pass if h1_pass else None)
    rows.append(h2)
    # H3：内部非劣效，单位 = 眼
    internal = df[(df["set"] == "internal") & df["config"].isin([A, B])]
    w, cl = unit_table(internal, "dice", "volume", P)
    h3 = {"hypothesis": "H3", "role": "次要（H2 成立才判定）", "comparison": f"H3 {A} − {B}：TowardPi 第 0 折眼 IRF Dice 非劣效（界值 −{P['NI_MARGIN']}）"}
    if not w.empty and A in w and B in w:
        dd = (w[A] - w[B]).dropna()
        n = len(dd)
        h3.update(n_units=n, mean_a=w[A].mean(), mean_b=w[B].mean(), mean_diff=float(dd.mean()))
        if n >= 2:
            se = dd.std(ddof=1) / np.sqrt(n)
            t = stats.t.ppf(0.975, n - 1)
            h3.update(ci_low=float(dd.mean() - t * se), ci_high=float(dd.mean() + t * se), ci_method="t（按眼）")
        wb, clb = unit_table(internal, "dice", "bscan", P)
        pb = paired(wb[A], wb[B], clb, P, "")
        h3.update(boot_ci_low=pb["ci_low"], boot_ci_high=pb["ci_high"], boot_note="B-scan 级、以眼聚类 bootstrap（敏感性）")
        ni = bool(h3.get("ci_low", -np.inf) > -P["NI_MARGIN"])
        h3.update(noninferior=ni)
    else:
        h3.update(n_units=0, note="缺少内部验证结果")
        ni = False
    h3.update(tested=h2_pass, passed=ni if h2_pass else None)
    rows.append(h3)
    # 待定项 5：本院 Optovue 若进入假设检验，作为 H4 接在 H3 之后（固定顺序把关，不另耗 α）
    if P["LOCAL_CONFIRMATORY"] and P["PRIMARY"] == "retouch":
        loc = df[(df["set"] == "local") & df["config"].isin([A, B])]
        h4 = compare(loc, "dice", "volume", A, B, P, f"H4 {A} vs {B}：本院 Optovue 眼 IRF Dice")
        h3_pass = bool(h2_pass and ni)
        h4.update(hypothesis="H4", role="次要（H3 成立才判定；LOCAL_CONFIRMATORY=true）", tested=h3_pass,
                  passed=bool(h4.get("p", np.nan) < P["ALPHA"] and h4.get("mean_diff", 0) > 0) if h3_pass else None)
        rows.append(h4)
    return pd.DataFrame(rows), sets


def exploratory(df, sets, P):
    A, B = P["PRIMARY_CFG"], P["BASE_CFG"]
    prim = df[df["set"].isin(sets)]
    present = set(prim["config"].unique())
    others = sorted(c for c in present if c not in (A, B) and c not in OUTSIDE_HOLM)
    rows = [compare(prim, "dice", P["UNIT"], c, B, P, f"{c} vs {B}") for c in others]
    out = pd.DataFrame(rows)
    if len(out) and "p" in out:
        ok = out["p"].notna()
        out["p_holm"] = np.nan
        if ok.any():
            out.loc[ok, "p_holm"] = U.holm(out.loc[ok, "p"].values)
    extra = []
    if "RAS_WTConv" in others and A in present:
        extra.append(compare(prim, "dice", P["UNIT"], "RAS_WTConv", A, P, f"RAS_WTConv vs {A}（WTConv 增量，不进 Holm 族）"))
    if "PDUNet" in present:
        extra.append(compare(prim, "dice", P["UNIT"], "PDUNet", B, P, f"PDUNet vs {B}（开题 PDU-Net 重新实现，不进 Holm 族）"))
        if A in present:
            extra.append(compare(prim, "dice", P["UNIT"], "PDUNet", A, P, f"PDUNet vs {A}（不进 Holm 族）"))
        inter = df[df["set"] == "internal"]
        if {"PDUNet", B} <= set(inter["config"]):
            extra.append(compare(inter, "dice", "volume", "PDUNet", B, P,
                                 f"PDUNet vs {B}：TowardPi 第 0 折内部（按眼，不进 Holm 族）"))
        if "PDUNetSerial" in present:
            extra.append(compare(prim, "dice", P["UNIT"], "PDUNetSerial", "PDUNet", P,
                                 "PDUNetSerial vs PDUNet（DAN 串行 vs 并行，不进 Holm 族）"))
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    return out


def devices(df, P):
    rows = []
    for st in [s for s in RETOUCH_SETS + [NODISP, "local", "duke"] if s in set(df["set"])]:
        d = df[df["set"] == st]
        for cfg, g in d.groupby("config"):
            irf, neg = g[g.has_irf == 1], g[g.has_irf == 0]
            vol = irf.groupby(["seed", "patient_id"])["dice"].mean().groupby("patient_id").mean()
            silent_img = irf.groupby("seed")["empty"].mean().mean() if len(irf) else np.nan
            rows.append({"set": st, "device": DEVICE.get(st, st), "config": cfg, "n_seeds": g["seed"].nunique(),
                         "n_units": len(vol), "dice_unit_mean": vol.mean(), "dice_unit_sd": vol.std(ddof=1),
                         "silent_rate_img": silent_img,
                         "alarm": bool(silent_img > P["ALARM"]) if pd.notna(silent_img) else None,
                         "fp_rate": neg.assign(fp=(neg["n_pr"] > 0).astype(float)).groupby("seed")["fp"].mean().mean() if len(neg) else np.nan})
    return pd.DataFrame(rows)


def strata(df, sets, P):
    A, B = P["PRIMARY_CFG"], P["BASE_CFG"]
    prim = df[df["set"].isin(sets) & df["config"].isin([A, B])]
    if prim.empty or "has_srf" not in prim:
        return pd.DataFrame()
    vol = prim.drop_duplicates(["patient_id", "case"]).groupby("patient_id").agg(
        any_srf=("has_srf", "max"), any_ped=("has_ped", "max"), irf_total=("irf_mm2", "sum"))
    w, cl = unit_table(prim, "dice", "volume", P)
    if w.empty or A not in w or B not in w:
        return pd.DataFrame()
    vol = vol.reindex(w.index)
    pos = vol["irf_total"] > 0
    tert = pd.Series(index=vol.index, dtype=object)
    if pos.sum() >= 3:
        tert[pos] = pd.qcut(vol.loc[pos, "irf_total"], 3, labels=["IRF 面积低三分位", "中三分位", "高三分位"]).astype(str)
    rows = []
    for name, mask in [("含 SRF", vol["any_srf"] == 1), ("不含 SRF", vol["any_srf"] == 0),
                       ("含 PED", vol["any_ped"] == 1), ("不含 PED", vol["any_ped"] == 0)] + \
                      [(t, tert == t) for t in ["IRF 面积低三分位", "中三分位", "高三分位"]]:
        m = mask.fillna(False).values
        r = paired(w[A][m], w[B][m], cl[m], P, name)
        r.pop("p", None)                    # 分层只做描述，不报 P
        rows.append(r)
    return pd.DataFrame(rows)


def external_other(df, P):
    rows = []
    A, B = P["PRIMARY_CFG"], P["BASE_CFG"]
    for st in ["duke", "local"]:
        if st == "local" and (P["PRIMARY"] == "local" or P["LOCAL_CONFIRMATORY"]):
            continue                         # 本院数据已在假设检验中，不再重复列为探索性
        d = df[df["set"] == st]
        if d.empty:
            continue
        w, cl = unit_table(d, "dice", "volume", P)
        for cfg in w.columns:
            x = w[cfg].dropna()
            lo, hi = cluster_boot_mean(x.values, x.index, P)
            rows.append({"set": st, "config": cfg, "metric": "患者/眼 IRF Dice", "n": len(x), "mean": x.mean(),
                         "sd": x.std(ddof=1), "ci_low": lo, "ci_high": hi})
        if A in w and B in w:
            r = paired(w[A], w[B], cl, P, f"{st}: {A} vs {B}（描述性）")
            r.pop("p", None)
            rows.append({"set": st, "config": f"{A}−{B}", "metric": "配对差值", "n": r["n_units"], "mean": r["mean_diff"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    o = df[df["set"] == "octdl"]
    for cfg, g in o.groupby("config"):
        img = g.assign(fp=(g["n_pr"] > 0).astype(float)).groupby("case")["fp"].mean()   # 种子平均后的逐图假阳性
        n = len(img)
        k = img.sum()
        lo, hi = wilson(k, n)
        rows.append({"set": "octdl", "config": cfg, "metric": "假阳性率（种子平均，Wilson CI 以图像数为 n）",
                     "n": n, "mean": k / n if n else np.nan, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def mixedlm(prim, P):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"id": "S4", "description": "线性混合模型", "note": "未安装 statsmodels，跳过（云端 run_freq.sh 会安装）"}
    A, B = P["PRIMARY_CFG"], P["BASE_CFG"]
    x = prim[(prim.has_irf == 1) & prim["config"].isin([A, B])].copy()
    if x.empty or x["config"].nunique() < 2:
        return {"id": "S4", "description": "线性混合模型", "note": "数据不足"}
    x["is_a"] = (x["config"] == A).astype(int)
    x["seed_c"] = x["config"] + "_" + x["seed"].astype(str)
    multi_dev = x["device"].nunique() > 1
    formula = "dice ~ is_a" + (" + C(device)" if multi_dev else "")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(formula, x, groups=x["patient_id"], vc_formula={"seed": "0 + C(seed_c)"}).fit(reml=True)
        ci = m.conf_int().loc["is_a"]
        return {"id": "S4", "description": f"线性混合模型：{formula} + (1|容积) + (1|配置×种子)",
                "n_units": x["patient_id"].nunique(), "mean_diff": float(m.params["is_a"]),
                "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(m.pvalues["is_a"]),
                "note": "已收敛" if m.converged else "未收敛，结果仅供参考"}
    except Exception as e:  # noqa: BLE001
        return {"id": "S4", "description": "线性混合模型", "note": f"拟合失败：{type(e).__name__}: {e}"}


def sensitivity(df, sets, P):
    A, B = P["PRIMARY_CFG"], P["BASE_CFG"]
    prim = df[df["set"].isin(sets) & df["config"].isin([A, B])]
    rows = []

    def add(sid, desc, r):
        r = dict(r)
        r.update(id=sid, description=desc)
        if "p" in r and pd.notna(r.get("p")) and pd.notna(r.get("mean_diff")):
            r["consistent_with_H1"] = bool(r["p"] < P["ALPHA"] and r["mean_diff"] > 0)
        rows.append(r)

    add("S1", "逐幅等权：每幅含 IRF 的 B-scan 为单位，以容积聚类 bootstrap", compare(prim, "dice", "bscan", A, B, P, "S1"))
    add("S2", "纳入不含 IRF 的图：都为空 = 1，有假阳性 = 0", compare(prim, "dice_all", P["UNIT"], A, B, P, "S2"))
    add("S3", f"只用 IRF 面积 ≥ {P['MIN_IRF_MM2']} mm² 的 B-scan", compare(prim, "dice_big", P["UNIT"], A, B, P, "S3"))
    rows.append(mixedlm(prim, P))
    # S5：逐种子
    pairs = []
    sa = sorted(prim.loc[prim.config == A, "seed"].unique())
    sb = sorted(prim.loc[prim.config == B, "seed"].unique())
    for i in sa:
        for j in sb:
            r = compare(prim, "dice", P["UNIT"], A, B, P, f"{A} s{i} vs {B} s{j}", seeds={A: [i], B: [j]})
            r.update(seed_a=i, seed_b=j)
            pairs.append(r)
    s5 = pd.DataFrame(pairs)
    if len(s5) and "mean_diff" in s5:
        sig = int(((s5["p"] < P["ALPHA"]) & (s5["mean_diff"] > 0)).sum())
        rows.append({"id": "S5", "description": f"逐种子 {len(sa)}×{len(sb)} 配对", "n_pairs": len(s5),
                     "mean_diff_min": s5["mean_diff"].min(), "mean_diff_max": s5["mean_diff"].max(),
                     "n_sig_positive": sig, "note": f"{sig}/{len(s5)} 组 P<{P['ALPHA']} 且 RAS 更好"})
    else:
        rows.append({"id": "S5", "description": "逐种子配对", "note": "数据不足"})
    # S6：基线去掉文章B 模型
    s6 = prim[~((prim.config == B) & (prim.trainer == BASE))]
    add("S6", "基线只用 FreqBase 种子（去掉复用的文章B 模型）", compare(s6, "dice", P["UNIT"], A, B, P, "S6"))
    # S7：Spectralis 不做显示变换
    if NODISP in set(df["set"]) and P["PRIMARY"] == "retouch":
        s7sets = [NODISP if s == "retouch_spectralis" else s for s in sets]
        p7 = df[df["set"].isin(s7sets) & df["config"].isin([A, B])]
        add("S7", "Spectralis 换成不做 1/4 次方显示变换的版本", compare(p7, "dice", P["UNIT"], A, B, P, "S7"))
    else:
        rows.append({"id": "S7", "description": "Spectralis 不做显示变换",
                     "note": "未生成 retouch_spectralis_nodisp（原始数据非线性强度，或尚未转换），不适用"})
    return pd.DataFrame(rows), s5


# ============================================================ 报告
def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))


def report(P, H, sets, X, D, S, path):
    L = [f"# 文章D 分析报告（s12_freq_eval.py v{VERSION}）", "",
         f"参数：UNIT={P['UNIT']}，MCID={P['MCID']}，NI_MARGIN={P['NI_MARGIN']}，PRIMARY={P['PRIMARY']}，"
         f"N_BOOT={P['N_BOOT']}，BOOT_SEED={P['BOOT_SEED']}，ALPHA={P['ALPHA']}", f"主要测试集：{', '.join(sets) or '无'}", "",
         "## 假设（固定顺序把关 H1 → H2 → H3）", "",
         "| 假设 | 单位数 | RAS | 基线 | 差值 | 95% CI | P | 是否检验 | 结论 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in H.iterrows():
        concl = "未检验（前序未成立）" if not r.get("tested") else ("成立" if r.get("passed") else "不成立")
        L.append(f"| {r['hypothesis']} | {fmt(r.get('n_units'), 0)} | {fmt(r.get('mean_a'))} | {fmt(r.get('mean_b'))} | "
                 f"{fmt(r.get('mean_diff'))} | {fmt(r.get('ci_low'))} – {fmt(r.get('ci_high'))} | {fmt(r.get('p'), 4)} | "
                 f"{'是' if r.get('tested') else '否'} | {concl} |")
    h1 = H.iloc[0]
    if h1.get("n_units"):
        L += ["", f"H1：Hodges–Lehmann 中位差 {fmt(h1.get('hl'))}；RAS 更好的单位 {fmt(h1.get('n_better'), 0)}/{fmt(h1.get('n_units'), 0)}；"
              f"{'达到' if h1.get('mcid_reached') else '未达到'}最小有意义差值 {P['MCID']}（不影响判定）。"]
    if len(X):
        L += ["", "## 探索性比较（频域消融配置 vs 基线做 Holm 校正；P(Holm) 为 — 的行不进 Holm 族）", "", "| 比较 | 单位数 | 差值 | 95% CI | P | P(Holm) |", "|---|---|---|---|---|---|"]
        for _, r in X.iterrows():
            L.append(f"| {r['comparison']} | {fmt(r.get('n_units'), 0)} | {fmt(r.get('mean_diff'))} | "
                     f"{fmt(r.get('ci_low'))} – {fmt(r.get('ci_high'))} | {fmt(r.get('p'), 4)} | {fmt(r.get('p_holm'), 4)} |")
    if len(D):
        L += ["", "## 分设备（描述性）", "", "| 设备 | 配置 | 单位数 | Dice 均值 | 静默失败率 | 报警 | 假阳性率 |", "|---|---|---|---|---|---|---|"]
        for _, r in D.iterrows():
            L.append(f"| {r['device']} | {r['config']} | {r['n_units']} | {fmt(r['dice_unit_mean'])} | {fmt(r['silent_rate_img'])} | "
                     f"{'是' if r['alarm'] else ('否' if r['alarm'] is not None else '—')} | {fmt(r['fp_rate'])} |")
    if len(S):
        L += ["", "## 敏感性分析", "", "| 编号 | 内容 | 差值 | 95% CI | P | 与 H1 一致 | 备注 |", "|---|---|---|---|---|---|---|"]
        for _, r in S.iterrows():
            cons = r.get("consistent_with_H1")
            L.append(f"| {r['id']} | {r['description']} | {fmt(r.get('mean_diff'))} | {fmt(r.get('ci_low'))} – {fmt(r.get('ci_high'))} | "
                     f"{fmt(r.get('p'), 4)} | {'—' if cons is None or (isinstance(cons, float) and np.isnan(cons)) else ('是' if cons else '否')} | {r.get('note', '') if isinstance(r.get('note'), str) else ''} |")
    if NOTES:
        L += ["", "## 注意", ""] + [f"- {n}" for n in NOTES]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    global OUT
    NOTES.clear()
    P = load_params(argv)
    OUT = U.p("results", "freq")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "params_used.json"), "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, **P}, f, ensure_ascii=False, indent=2)
    df = collect()
    df.to_csv(os.path.join(OUT, "metrics_long.csv"), index=False)
    H, sets = hypotheses(df, P)
    X = exploratory(df, sets, P)
    D = devices(df, P)
    T = strata(df, sets, P)
    O = external_other(df, P)
    S, S5 = sensitivity(df, sets, P)
    for name, t in [("hypotheses", H), ("exploratory", X), ("devices", D), ("strata", T),
                    ("external_other", O), ("sensitivity", S), ("s5_seed_pairs", S5)]:
        t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    report(P, H, sets, X, D, S, os.path.join(OUT, "report.md"))
    with open(os.path.join(OUT, "report.md"), encoding="utf-8") as f:
        print(f.read())
    print(f"→ {OUT}")
    return {"H": H, "X": X, "D": D, "T": T, "O": O, "S": S, "S5": S5, "df": df, "P": P}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
