"""
第 12b 步（文章D）：H3 按眼的内部非劣效分析（OSF 更新三，2026-11）

背景：登记（osf.io/peums）写 H3 为"TowardPi 第 0 折验证集 9 眼、按眼配对差值的 t 分布 95% CI"。
11-15 核对发现验证集实为 9 个容积、来自 7 眼（DME07 一眼 3 个容积）；冻结的 s12_freq_eval.py
（SHA-256 7bbac415…）以容积编号为单位，算的是 9 个容积。本脚本按 OSF 更新三实现按眼的主分析，
不改 s12，只读取 s12 的逐图输出。

主分析（H3 判定只看这一项）：
  1. 只用 TowardPi 内部验证集（set == "internal"）、配置 RAS 与 baseline、含 IRF 的 B-scan；
  2. 每个 (配置, 种子, 容积)：含 IRF B-scan 的 Dice 均值；再对种子取平均 → 每个 (配置, 容积) 一个值；
  3. 每个容积：d_v = RAS − baseline；
  4. 同一只眼的容积差值取平均 → 每只眼一个差值 d_e（眼等权）；
  5. 眼级平均差值的双侧 95% t 分布 CI（自由度 n_eye − 1）；下限 > −NI_MARGIN（0.03）即非劣效。
敏感性（照报，不用于判定）：
  a. 按眼重采样的百分位 bootstrap 95% CI（10,000 次，种子 2026）；
  b. 冻结实现的容积级结果（n = 9 个容积，t-CI），与 s12 hypotheses.csv 中 H3 行核对一致性。
把关：H1 → H2 → H3 固定顺序。H2 未通过（hypotheses.csv 中 H2 passed 不为 True）时，H3 照算但标"不判定"。

输入：work/results/freq/metrics_long.csv、hypotheses.csv（s12 输出）；work/tables/towardpi_cases.csv（eye 列）
输出：work/results/freq/h3_eye.csv、h3_eye_units.csv、h3_eye_report.md
用法：python s12b_h3_eye.py            （WORK 取 config.py，与 s12 相同，可用环境变量 DME_WORK）
      python s12b_h3_eye.py --work DIR  （指定 work 目录，自测用）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "s12b v1.0 (2026-11-17)"
A, B = "RAS", "baseline"
NI_MARGIN = 0.03
N_BOOT, BOOT_SEED, ALPHA = 10000, 2026, 0.05


def default_work():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config as C
        return C.WORK
    except Exception:
        return os.environ.get("DME_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))


def load(work):
    res = os.path.join(work, "results", "freq")
    ml = pd.read_csv(os.path.join(res, "metrics_long.csv"), dtype={"case": str})
    tp = pd.read_csv(os.path.join(work, "tables", "towardpi_cases.csv"), dtype=str)[["case", "vol", "eye"]]
    hyp_p = os.path.join(res, "hypotheses.csv")
    hyp = pd.read_csv(hyp_p) if os.path.exists(hyp_p) else pd.DataFrame()
    return res, ml, tp, hyp


def unit_values(ml, tp):
    """返回每个 (容积, 眼) 的 RAS、baseline 种子平均 Dice，以及种子数核对表"""
    x = ml[(ml["set"] == "internal") & ml["config"].isin([A, B]) & (ml["has_irf"] == 1)].copy()
    x = x.drop(columns=[c for c in ("vol", "eye") if c in x.columns]).merge(tp, on="case", how="left")
    miss = x["eye"].isna().sum()
    if miss:
        raise SystemExit(f"{miss} 行在 towardpi_cases.csv 中找不到 eye：请核对 case 名")
    seeds = x.groupby("config")["seed"].apply(lambda s: sorted(set(int(v) for v in s))).to_dict()
    per_seed = x.groupby(["config", "seed", "vol"])["dice"].mean()
    per_vol = per_seed.groupby(["config", "vol"]).mean().unstack("config")
    eye_of = x.drop_duplicates("vol").set_index("vol")["eye"]
    per_vol["eye"] = eye_of.reindex(per_vol.index)
    return per_vol, seeds


def t_ci(d):
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean()) if n else np.nan
    if n < 2:
        return n, m, np.nan, np.nan, np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - ALPHA / 2, n - 1)
    return n, m, m - t * se, m + t * se, float(d.std(ddof=1))


def boot_ci(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bm = d[idx].mean(axis=1)
    return float(np.percentile(bm, 100 * ALPHA / 2)), float(np.percentile(bm, 100 * (1 - ALPHA / 2)))


def h2_passed(hyp):
    if hyp.empty or "hypothesis" not in hyp:
        return None
    r = hyp[hyp["hypothesis"] == "H2"]
    if r.empty or "passed" not in r:
        return None
    v = r["passed"].iloc[0]
    return True if str(v).strip().lower() in ("true", "1", "1.0") else False


def analyse(per_vol, seeds, hyp):
    v = per_vol.dropna(subset=[A, B]).copy()
    v["d"] = v[A] - v[B]
    e = v.groupby("eye").agg(d=("d", "mean"), n_vol=("d", "size"), ras=(A, "mean"), base=(B, "mean"))
    n_e, m_e, lo_e, hi_e, sd_e = t_ci(e["d"])
    blo, bhi = boot_ci(e["d"].values) if n_e >= 2 else (np.nan, np.nan)
    n_v, m_v, lo_v, hi_v, sd_v = t_ci(v["d"])
    h2 = h2_passed(hyp)
    ni = bool(lo_e > -NI_MARGIN) if np.isfinite(lo_e) else None
    row = {"analysis": "H3 eye-level (primary, OSF update 3)", "version": VERSION,
           "n_eyes": n_e, "n_volumes": n_v, "seeds_RAS": json.dumps(seeds.get(A, [])),
           "seeds_baseline": json.dumps(seeds.get(B, [])),
           "mean_RAS_eye": float(e["ras"].mean()), "mean_baseline_eye": float(e["base"].mean()),
           "mean_diff_eye": m_e, "sd_diff_eye": sd_e, "t_ci_low": lo_e, "t_ci_high": hi_e,
           "ni_margin": -NI_MARGIN, "noninferior": ni,
           "boot_ci_low_eye": blo, "boot_ci_high_eye": bhi,
           "volume_level_mean_diff": m_v, "volume_level_t_ci_low": lo_v, "volume_level_t_ci_high": hi_v,
           "h2_passed": h2, "h3_tested": (h2 is True), "h3_passed": (ni if h2 is True else None)}
    # 与 s12 冻结实现的容积级 H3 核对
    if not hyp.empty and "hypothesis" in hyp and (hyp["hypothesis"] == "H3").any():
        r = hyp[hyp["hypothesis"] == "H3"].iloc[0]
        row["s12_H3_n_units"] = r.get("n_units", np.nan)
        row["s12_H3_mean_diff"] = r.get("mean_diff", np.nan)
        row["s12_H3_ci_low"] = r.get("ci_low", np.nan)
        try:
            row["s12_match"] = bool(int(r.get("n_units")) == n_v and abs(float(r.get("mean_diff")) - m_v) < 1e-9
                                    and abs(float(r.get("ci_low")) - lo_v) < 1e-9)
        except Exception:
            row["s12_match"] = None
    return row, v.reset_index(), e.reset_index()


def report(row, path):
    def f(x):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.4f}" if isinstance(x, float) else str(x))
    lines = [f"# H3 按眼内部非劣效（{VERSION}）", "",
             f"- 单位：{row['n_eyes']} 眼（由 {row['n_volumes']} 个容积汇总，眼等权）；种子 RAS {row['seeds_RAS']}，baseline {row['seeds_baseline']}",
             f"- 眼级平均 Dice：RAS {f(row['mean_RAS_eye'])}，baseline {f(row['mean_baseline_eye'])}",
             f"- 主分析：平均差值（RAS − baseline）{f(row['mean_diff_eye'])}，双侧 95% t-CI [{f(row['t_ci_low'])}, {f(row['t_ci_high'])}]，界值 {row['ni_margin']}，非劣效 = {f(row['noninferior'])}",
             f"- 把关：H2 通过 = {f(row['h2_passed'])} → H3 {'判定' if row['h3_tested'] else '不判定（只报告）'}；H3 通过 = {f(row['h3_passed'])}",
             f"- 敏感性 a（按眼 bootstrap，{N_BOOT} 次，种子 {BOOT_SEED}）：[{f(row['boot_ci_low_eye'])}, {f(row['boot_ci_high_eye'])}]",
             f"- 敏感性 b（冻结实现，容积级，未处理眼内相关）：平均差值 {f(row['volume_level_mean_diff'])}，t-CI [{f(row['volume_level_t_ci_low'])}, {f(row['volume_level_t_ci_high'])}]",
             f"- 与 s12 hypotheses.csv 中 H3 行一致：{f(row.get('s12_match'))}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    work = a.work or default_work()
    res, ml, tp, hyp = load(work)
    per_vol, seeds = unit_values(ml, tp)
    row, vt, et = analyse(per_vol, seeds, hyp)
    pd.DataFrame([row]).to_csv(os.path.join(res, "h3_eye.csv"), index=False)
    vt.merge(et[["eye", "d"]].rename(columns={"d": "d_eye"}), on="eye").to_csv(os.path.join(res, "h3_eye_units.csv"), index=False)
    print(report(row, os.path.join(res, "h3_eye_report.md")))
    return row


if __name__ == "__main__":
    main()
