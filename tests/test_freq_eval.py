# -*- coding: utf-8 -*-
"""
s12_freq_eval.py v0.3 自测：用模拟数据跑完整分析，不需要 GPU、torch、nnU-Net 或任何真实测试集。
模拟内容：TowardPi 第 0 折验证集（9 眼）、RETOUCH 三家设备（每家 8 个容积，其中 1 个不含 IRF）、
          Spectralis 不做显示变换版本（S7）、Duke、本院、OCTDL；配置 = 基线（文章B 种子 1 + FreqBase 种子 2、3）、
          RAS（种子 1–3）、RASS / WTConv / FFC / RAS_WTConv / PDUNet（种子 1）。
场景：effect（植入 RAS 更好）/ null（RAS 与基线同分布）/ 参数覆盖 / 备用方案 PRIMARY=local。
用法（dme_code 目录）：python tests/test_freq_eval.py      应显示“全部通过”
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H, W = 32, 32
FAILED = []


def check(name, cond, info=""):
    print(("  通过  " if cond else "  失败  ") + name + (f"  ({info})" if info else ""))
    if not cond:
        FAILED.append(name)


def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((arr > 0).astype(np.uint8)).save(path)


def gt_mask(rng, has):
    m = np.zeros((H, W), np.uint8)
    if has:
        y, x = rng.integers(4, 18, 2)
        m[y:y + rng.integers(6, 12), x:x + rng.integers(6, 12)] = 1
    return m


def pred_mask(rng, gt, keep, p_empty, p_fp):
    if gt.any():
        if rng.random() < p_empty:
            return np.zeros_like(gt)
        return (gt.astype(bool) & (rng.random(gt.shape) < keep)).astype(np.uint8)
    out = np.zeros_like(gt)
    if rng.random() < p_fp:
        out[2:5, 2:5] = 1
    return out


# 各配置的“质量”：(keep, p_empty, p_fp)；effect 场景 RAS 明显更好，null 场景与基线相同
def quality(cfg, scenario, where):
    if where == "internal":
        return {"RAS": (0.86, 0.0, 0.05)}.get(cfg, (0.85, 0.0, 0.05))
    base = (0.45, 0.5, 0.05)
    if scenario == "null":
        return base
    return {"RAS": (0.80, 0.10, 0.08), "RASS": (0.70, 0.2, 0.06), "RAS_WTConv": (0.82, 0.1, 0.08),
            "WTConv": (0.50, 0.45, 0.05), "FFC": (0.47, 0.5, 0.05), "PDUNet": (0.48, 0.47, 0.06)}.get(cfg, base)


RUNS = [("nnUNetTrainer_250epochs", 1), ("nnUNetTrainer_FreqBase", 2), ("nnUNetTrainer_FreqBase", 3),
        ("nnUNetTrainer_RAS", 1), ("nnUNetTrainer_RAS", 2), ("nnUNetTrainer_RAS", 3),
        ("nnUNetTrainer_RASS", 1), ("nnUNetTrainer_WTConv", 1), ("nnUNetTrainer_FFC", 1), ("nnUNetTrainer_RAS_WTConv", 1),
        ("nnUNetTrainer_PDUNet", 1)]


def build(work, scenario, seed=0):
    rng = np.random.default_rng(seed)
    cfg_of = lambda tr: "baseline" if tr in ("nnUNetTrainer_250epochs", "nnUNetTrainer_FreqBase") else tr.replace("nnUNetTrainer_", "")
    # —— TowardPi 第 0 折验证集：9 眼 × 10 幅 ——
    tp = os.path.join(work, "nnUNet_raw", "Dataset101_TowardPi", "labelsTr")
    internal = []
    for e in range(9):
        for j in range(10):
            c = f"TP_DME01{e:02d}_{j:03d}"
            g = gt_mask(rng, j % 5 != 0)
            save(os.path.join(tp, c + ".png"), g)
            internal.append((c, g))
    # —— 外部集 ——
    ext = {}
    rows = []
    for vendor in ("Cirrus", "Spectralis", "Topcon"):
        st = f"retouch_{vendor.lower()}"
        ext[st] = []
        for v in range(8):
            pid = f"RETOUCH_{vendor}_TRAIN{v + 1:03d}"
            for k in range(6):
                c = f"{pid}_{k:03d}"
                g = gt_mask(rng, v != 0 and k % 3 != 0)          # 第 0 个容积不含 IRF
                save(os.path.join(work, st, "gt", c + ".png"), g)
                ext[st].append((c, g))
                rows.append({"case": c, "patient": pid, "vendor": vendor, "volume": f"TRAIN{v + 1:03d}", "bscan": k,
                             "has_irf": int(g.any()), "has_srf": int(v % 2 == 0), "has_ped": int(v % 3 == 0),
                             "irf_mm2": float(g.sum() * 3.9 * 11.7 / 1e6)})
    ext["retouch_spectralis_nodisp"] = ext["retouch_spectralis"]
    for c, g in ext["retouch_spectralis"]:
        save(os.path.join(work, "retouch_spectralis_nodisp", "gt", c + ".png"), g)
    os.makedirs(os.path.join(work, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(work, "tables", "retouch_cases.csv"), index=False)
    ext["duke"], ext["local"], ext["octdl"] = [], [], []
    for s in range(1, 11):
        for k in (5, 10, 15):
            c = f"DUKE_Subject_{s:02d}_{k}"
            g = gt_mask(rng, k != 5)
            save(os.path.join(work, "duke", "gt1", c + ".png"), g)
            ext["duke"].append((c, g))
    lrows = []
    for s in range(1, 13):
        c = f"L{s:04d}"
        g = gt_mask(rng, s <= 8)
        save(os.path.join(work, "local", "gt", c + ".png"), g)
        ext["local"].append((c, g))
        lrows.append({"case": c, "patient": c})
    pd.DataFrame(lrows).to_csv(os.path.join(work, "tables", "local_cases.csv"), index=False)
    for i in range(20):
        ext["octdl"].append((f"OCTDL_N{i:03d}", np.zeros((H, W), np.uint8)))
    # —— 预测 ——
    for tr, sd in RUNS:
        cfg = cfg_of(tr)
        vd = (os.path.join(work, "nnUNet_results", "Dataset101_TowardPi", f"{tr}__nnUNetPlans__2d", "fold_0", "validation")
              if tr == "nnUNetTrainer_250epochs" else
              os.path.join(work, "nnUNet_results_freq", f"seed{sd}", "Dataset101_TowardPi", f"{tr}__nnUNetPlans__2d", "fold_0", "validation"))
        q = quality(cfg, scenario, "internal")
        for c, g in internal:
            save(os.path.join(vd, c + ".png"), pred_mask(rng, g, *q))
        q = quality(cfg, scenario, "external")
        for st, items in ext.items():
            if st == "retouch_spectralis_nodisp" and cfg not in ("RAS", "baseline"):
                continue
            for c, g in items:
                save(os.path.join(work, "pred", "freq", f"{tr}_s{sd}", st, c + ".png"), pred_mask(rng, g, *q))


def run(work, *sets):
    env = dict(os.environ, DME_WORK=work, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    args = [sys.executable, os.path.join(ROOT, "s12_freq_eval.py")]
    for s in sets:
        args += ["--set", s]
    r = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    out = os.path.join(work, "results", "freq")
    def load(n):
        f = os.path.join(out, n + ".csv")
        try:
            return pd.read_csv(f)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()
    return r.returncode, {n: load(n) for n in ("hypotheses", "exploratory", "devices", "strata", "external_other",
                                                "sensitivity", "s5_seed_pairs")}, out


def main():
    tmp = tempfile.mkdtemp(prefix="freq_eval_test_")
    try:
        print("1) effect 场景（植入 RAS 更好），默认参数（N_BOOT=10000）")
        w1 = os.path.join(tmp, "effect")
        build(w1, "effect", seed=1)
        rc, R, out = run(w1)
        check("运行成功", rc == 0)
        Hh = R["hypotheses"].set_index("hypothesis")
        check("H1 单位数 = 21 个含 IRF 容积", int(Hh.loc["H1", "n_units"]) == 21, f"{Hh.loc['H1', 'n_units']}")
        check("H1 成立", bool(Hh.loc["H1", "passed"]), f"差值 {Hh.loc['H1', 'mean_diff']:.3f}, P={Hh.loc['H1', 'p']:.2g}")
        check("H1 bootstrap CI 包含点估计", Hh.loc["H1", "ci_low"] <= Hh.loc["H1", "mean_diff"] <= Hh.loc["H1", "ci_high"])
        check("H1 报告 Hodges–Lehmann 和 MCID", pd.notna(Hh.loc["H1", "hl"]) and "mcid_reached" in Hh.columns)
        check("H2 已检验且成立（RAS 静默失败率更低）", bool(Hh.loc["H2", "tested"]) and bool(Hh.loc["H2", "passed"]),
              f"差值 {Hh.loc['H2', 'mean_diff']:.3f}")
        check("H3 已检验，按 9 眼计算", bool(Hh.loc["H3", "tested"]) and int(Hh.loc["H3", "n_units"]) == 9)
        check("H3 有 t-CI 和 bootstrap CI", pd.notna(Hh.loc["H3", "ci_low"]) and pd.notna(Hh.loc["H3", "boot_ci_low"]))
        X = R["exploratory"]
        holm = X.dropna(subset=["p_holm"])
        check("探索性：4 个配置进入 Holm 族，P(Holm) ≥ P", len(holm) == 4 and bool((holm["p_holm"] >= holm["p"] - 1e-12).all()))
        check("探索性：含 RAS_WTConv vs RAS", X["comparison"].str.contains("RAS_WTConv vs RAS").any())
        pdu = X[X["comparison"].str.startswith("PDUNet vs")]
        check("PDU-Net：vs 基线、vs RAS、内部三项都有，且不进 Holm 族",
              len(pdu) == 3 and pdu["p_holm"].isna().all() and not holm["comparison"].str.contains("PDUNet").any(),
              "; ".join(pdu["comparison"]))
        pin = pdu[pdu["comparison"].str.contains("内部")]
        check("PDU-Net 内部比较按 9 眼", len(pin) == 1 and int(pin["n_units"].iloc[0]) == 9)
        D = R["devices"]
        check("分设备：三家设备 + nodisp 都有，基线触发报警（空预测约 50%）",
              {"Cirrus", "Spectralis", "Topcon"} <= set(D["device"]) and bool(D[(D.config == "baseline") & D.set.str.startswith("retouch")]["alarm"].all()))
        S = R["sensitivity"].set_index("id")
        check("敏感性 S1–S7 全部有结果行", set(S.index) >= {"S1", "S2", "S3", "S4", "S5", "S6", "S7"})
        check("S7 已计算（nodisp 存在）", pd.notna(S.loc["S7"].get("mean_diff")))
        check("S6 已计算", pd.notna(S.loc["S6"].get("mean_diff")))
        check("S5 = 3×3 = 9 组配对", len(R["s5_seed_pairs"]) == 9)
        try:
            import statsmodels  # noqa: F401
            check("S4 混合模型已拟合", pd.notna(S.loc["S4"].get("mean_diff")), str(S.loc["S4"].get("note")))
        except ImportError:
            check("S4 无 statsmodels 时跳过并说明", "statsmodels" in str(S.loc["S4"].get("note")))
        check("分层描述：7 个分层", len(R["strata"]) == 7)
        O = R["external_other"]
        check("其他外部：Duke、本院、OCTDL（Wilson）都有", {"duke", "local", "octdl"} <= set(O["set"]))
        P = json.load(open(os.path.join(out, "params_used.json"), encoding="utf-8"))
        check("参数记录：默认值 = SAP 草稿值", P["UNIT"] == "volume" and P["MCID"] == 0.05 and P["NI_MARGIN"] == 0.03
              and P["PRIMARY"] == "retouch" and P["N_BOOT"] == 10000)
        check("报告文件生成", os.path.exists(os.path.join(out, "report.md")))
        bad = [r for r, ds, fs in os.walk(w1) for n in ds + fs if "*" in n]
        check("没有生成名字含 * 的文件或文件夹（Windows 兼容）", not bad, str(bad[:3]))

        print("2) 重复运行结果一致（bootstrap 种子固定）")
        h1a = Hh.loc["H1", ["ci_low", "ci_high"]].values
        rc, R2, _ = run(w1)
        h1b = R2["hypotheses"].set_index("hypothesis").loc["H1", ["ci_low", "ci_high"]].values
        check("两次 CI 完全相同", np.allclose(h1a.astype(float), h1b.astype(float)))

        print("3) 参数覆盖：UNIT=bscan、NI_MARGIN=0.2、N_BOOT=2000")
        rc, R3, out3 = run(w1, "UNIT=bscan", "NI_MARGIN=0.2", "N_BOOT=2000")
        H3 = R3["hypotheses"].set_index("hypothesis")
        P3 = json.load(open(os.path.join(out3, "params_used.json"), encoding="utf-8"))
        check("参数生效并记录", rc == 0 and P3["UNIT"] == "bscan" and P3["NI_MARGIN"] == 0.2)
        check("bscan 单位：H1 单位数 = 含 IRF 的 B-scan 数", int(H3.loc["H1", "n_units"]) > 21, f"{H3.loc['H1', 'n_units']}")
        check("界值放宽到 0.2 后 H3 判定为非劣效", bool(H3.loc["H3", "noninferior"]))
        rc_bad, _, _ = run(w1, "FOO=1")
        check("未知参数会报错退出", rc_bad != 0)

        print("4) null 场景（RAS = 基线）：把关应在 H1 处关闭")
        w2 = os.path.join(tmp, "null")
        build(w2, "null", seed=2)
        rc, R4, _ = run(w2, "N_BOOT=2000")
        H4 = R4["hypotheses"].set_index("hypothesis")
        check("H1 不成立", rc == 0 and not bool(H4.loc["H1", "passed"]), f"P={H4.loc['H1', 'p']:.2g}")
        check("H2、H3 未检验（仍报告估计值）", not bool(H4.loc["H2", "tested"]) and not bool(H4.loc["H3", "tested"])
              and pd.notna(H4.loc["H2", "mean_diff"]))

        print("5) 备用方案 PRIMARY=local")
        rc, R5, _ = run(w1, "PRIMARY=local", "N_BOOT=2000")
        H5 = R5["hypotheses"].set_index("hypothesis")
        check("H1 改用本院含 IRF 的 8 眼", rc == 0 and int(H5.loc["H1", "n_units"]) == 8 and "local" in H5.loc["H1", "comparison"])
        check("S7 在备用方案下不适用", "不适用" in str(R5["sensitivity"].set_index("id").loc["S7"].get("note")))

        print("5b) LOCAL_CONFIRMATORY=true：本院作为 H4 接在 H3 之后")
        rc, R5b, _ = run(w1, "LOCAL_CONFIRMATORY=true", "NI_MARGIN=0.2", "N_BOOT=2000")
        H5b = R5b["hypotheses"].set_index("hypothesis")
        check("出现 H4，按 8 眼，H3 成立后才检验", rc == 0 and "H4" in H5b.index and int(H5b.loc["H4", "n_units"]) == 8
              and bool(H5b.loc["H4", "tested"]) == bool(H5b.loc["H3", "passed"]))
        check("本院不再重复列入探索性", "local" not in set(R5b["external_other"].get("set", pd.Series(dtype=str))))
        rc, R5c, _ = run(w2 if os.path.exists(os.path.join(tmp, "null")) else w1, "LOCAL_CONFIRMATORY=true", "N_BOOT=500")
        H5c = R5c["hypotheses"].set_index("hypothesis")
        check("前序不成立时 H4 不检验", rc == 0 and not bool(H5c.loc["H4", "tested"]))

        print("6) 没有任何 RETOUCH 结果时不崩溃")
        w3 = os.path.join(tmp, "noretouch")
        shutil.copytree(w1, w3)
        for d in os.listdir(os.path.join(w3, "pred", "freq")):
            for st in ("retouch_cirrus", "retouch_spectralis", "retouch_topcon", "retouch_spectralis_nodisp"):
                shutil.rmtree(os.path.join(w3, "pred", "freq", d, st), ignore_errors=True)
        rc, R6, _ = run(w3, "N_BOOT=500")
        check("运行成功，H1 单位数为 0", rc == 0 and int(R6["hypotheses"].set_index("hypothesis").loc["H1", "n_units"]) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 40)
    print("全部通过" if not FAILED else f"{len(FAILED)} 项失败：{FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
