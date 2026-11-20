"""s12b_h3_eye.py 自测（模拟数据）"""
import os, sys, tempfile, json, math
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # 放在 dme_code/tests/ 时
sys.path.insert(0, HERE)                   # 与 s12b 同目录时
import s12b_h3_eye as S

OK = []
def check(name, cond):
    OK.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

VOLS = {"DME0101": "DME01", "DME0401": "DME04", "DME0701": "DME07", "DME0702": "DME07", "DME0703": "DME07",
        "DME1101": "DME11", "DME2601": "DME26", "DME3101": "DME31", "DME3601": "DME36"}

def make(work, effect, h2="True", extra_bad=False, s12_row=True):
    os.makedirs(os.path.join(work, "results", "freq")); os.makedirs(os.path.join(work, "tables"))
    rng = np.random.default_rng(1)
    tp, rows = [], []
    truth = {}
    for vol, eye in VOLS.items():
        base_true = rng.uniform(0.6, 0.8)
        eff = effect[vol]
        for k in range(20):
            case = f"TP_{vol}_{k*4:03d}"
            irf = int(k % 4 != 0)
            tp.append({"case": case, "vol": vol, "eye": eye, "group": eye, "slice": k * 4, "has_irf": irf, "irf_area_mm2": 0})
            for cfg, trainer, seeds in (("baseline", "nnUNetTrainer_250epochs", [1]), ("baseline", "nnUNetTrainer_FreqBase", [2, 3]),
                                        ("RAS", "nnUNetTrainer_RAS", [1, 2, 3]), ("RASS", "nnUNetTrainer_RASS", [1])):
                for sd in seeds:
                    mu = base_true + (eff if cfg == "RAS" else 0) + 0.01 * (sd - 2)
                    dice = mu + rng.normal(0, 0.02) if irf else (0.0 if cfg != "RAS" else 0.9)  # IRF 阴性图值不应进入分析
                    rows.append({"set": "internal", "trainer": trainer, "seed": sd, "condition": "freq", "case": case,
                                 "vol": vol, "eye": eye, "has_irf": irf, "dice": dice, "config": cfg, "patient_id": f"TP_{vol}"})
    # 外部集噪声行（不应进入分析）
    for i in range(30):
        rows.append({"set": "retouch_cirrus", "trainer": "nnUNetTrainer_RAS", "seed": 1, "condition": "freq",
                     "case": f"RETOUCH_Cirrus_001_{i:03d}", "vol": None, "eye": None, "has_irf": 1, "dice": 0.0,
                     "config": "RAS", "patient_id": "RETOUCH_Cirrus_001"})
    ml = pd.DataFrame(rows); tpdf = pd.DataFrame(tp)
    if extra_bad:
        ml = pd.concat([ml, pd.DataFrame([{**rows[0], "case": "TP_DME9901_004", "has_irf": 1, "dice": 0.5}])])
    ml.to_csv(os.path.join(work, "results", "freq", "metrics_long.csv"), index=False)
    tpdf.to_csv(os.path.join(work, "tables", "towardpi_cases.csv"), index=False)
    # 独立的手工期望值
    x = ml[(ml.set == "internal") & ml.config.isin(["RAS", "baseline"]) & (ml.has_irf == 1)]
    pv = {}
    for (cfg, vol), g in x.groupby(["config", "vol"]):
        pv[(cfg, vol)] = np.mean([g[g.seed == s].dice.mean() for s in sorted(g.seed.unique())])
    dv = {v: pv[("RAS", v)] - pv[("baseline", v)] for v in VOLS}
    de = {}
    for v, e in VOLS.items(): de.setdefault(e, []).append(dv[v])
    de = np.array([np.mean(de[e]) for e in sorted(de)])
    dvv = np.array([dv[v] for v in sorted(dv)])
    if s12_row:
        n = len(dvv); m = dvv.mean(); t = stats.t.ppf(0.975, n - 1); se = dvv.std(ddof=1) / math.sqrt(n)
        hyp = pd.DataFrame([{"hypothesis": "H1", "passed": True}, {"hypothesis": "H2", "passed": h2},
                            {"hypothesis": "H3", "n_units": n, "mean_diff": m, "ci_low": m - t * se}])
        hyp.to_csv(os.path.join(work, "results", "freq", "hypotheses.csv"), index=False)
    return de, dvv

# 1. 已知效应：DME07 三个容积效应很大，其余眼为 0 → 眼等权与容积等权必须不同
eff = {v: 0.0 for v in VOLS}; eff.update({"DME0701": 0.06, "DME0702": 0.06, "DME0703": 0.06})
with tempfile.TemporaryDirectory() as w:
    de, dvv = make(w, eff)
    row = S.main(["--work", w])
    check("眼数 = 7", row["n_eyes"] == 7)
    check("容积数 = 9", row["n_volumes"] == 9)
    check("种子 RAS [1,2,3]、baseline [1,2,3]", row["seeds_RAS"] == "[1, 2, 3]" and row["seeds_baseline"] == "[1, 2, 3]")
    check("眼级均值与手工一致", abs(row["mean_diff_eye"] - de.mean()) < 1e-12)
    n = 7; t = stats.t.ppf(0.975, n - 1); se = de.std(ddof=1) / math.sqrt(n)
    check("眼级 t-CI 下限与手工一致", abs(row["t_ci_low"] - (de.mean() - t * se)) < 1e-12)
    check("眼级 t-CI 上限与手工一致", abs(row["t_ci_high"] - (de.mean() + t * se)) < 1e-12)
    check("容积级均值与手工一致", abs(row["volume_level_mean_diff"] - dvv.mean()) < 1e-12)
    check("眼等权 ≠ 容积等权（DME07 权重不同）", abs(row["mean_diff_eye"] - row["volume_level_mean_diff"]) > 0.005)
    check("IRF 阴性图与外部集未进入（RAS 阴性图设为 0.9 也不影响）", row["mean_diff_eye"] < 0.05)
    check("与模拟的 s12 容积级 H3 核对一致", row["s12_match"] is True)
    check("H2 通过 → H3 判定", row["h3_tested"] is True and row["h3_passed"] == row["noninferior"])
    check("bootstrap CI 有限且包含点估计", row["boot_ci_low_eye"] <= row["mean_diff_eye"] <= row["boot_ci_high_eye"])
    row2 = S.main(["--work", w])
    check("bootstrap 可复现（同种子两次相同）", row2["boot_ci_low_eye"] == row["boot_ci_low_eye"])
    for fn in ("h3_eye.csv", "h3_eye_units.csv", "h3_eye_report.md"):
        check(f"输出 {fn}", os.path.exists(os.path.join(w, "results", "freq", fn)))
    u = pd.read_csv(os.path.join(w, "results", "freq", "h3_eye_units.csv"))
    check("逐单位表 9 行", len(u) == 9)

# 2. 劣效：RAS 比基线低 0.05 → 不非劣效
eff = {v: -0.05 for v in VOLS}
with tempfile.TemporaryDirectory() as w:
    make(w, eff)
    row = S.main(["--work", w])
    check("明显劣效 → noninferior = False", row["noninferior"] is False)

# 3. 等效：差 0 → 非劣效
eff = {v: 0.0 for v in VOLS}
with tempfile.TemporaryDirectory() as w:
    make(w, eff)
    row = S.main(["--work", w])
    check("无差异 → noninferior = True（t-CI 下限 > −0.03）", row["noninferior"] is True)

# 4. H2 未通过 → H3 不判定
with tempfile.TemporaryDirectory() as w:
    make(w, eff, h2="False")
    row = S.main(["--work", w])
    check("H2 未通过 → h3_tested False、h3_passed None", row["h3_tested"] is False and row["h3_passed"] is None)

# 5. 没有 hypotheses.csv → 照算、不判定
with tempfile.TemporaryDirectory() as w:
    make(w, eff, s12_row=False)
    row = S.main(["--work", w])
    check("无 hypotheses.csv → 不判定、不报错", row["h3_tested"] is False and "s12_match" not in row)

# 6. case 不在 towardpi_cases.csv → 报错停止
with tempfile.TemporaryDirectory() as w:
    make(w, eff, extra_bad=True)
    try:
        S.main(["--work", w]); check("缺 eye 时报错", False)
    except SystemExit:
        check("缺 eye 时报错", True)

print(f"\n{sum(OK)}/{len(OK)} 项通过")
sys.exit(0 if all(OK) else 1)
