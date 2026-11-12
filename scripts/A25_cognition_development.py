"""IQ, direction of deviations and developmental change.

Same calibrated z as A14/A15. IQ: WISC full-scale IQ (HBN release 11) and site-reported full-scale IQ (ABIDE).
(a) burden ~ IQ (per SD) + age + sex + site (+ diagnosis), negative binomial; below- and above-range counts separately.
(b) partial correlations of global volume z with IQ (age, sex, site).
(c) diagnosis vs held-out controls by direction; below-range excess with and without IQ adjustment; ASD vs ADHD without ASD.
(d) burden ~ age in held-out controls; diagnosis x age interactions in HBN.
Output: results/A25_cognition_development.json
Usage: python -I scripts/A25_cognition_development.py
"""
import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NJ = ROOT / "derivatives/norms_joint"
PH = ROOT / "data/raw/hbn_pheno_R11_20261112/extracted/HBN_phenotypic_R11/HBN_phenotypic_R11"
Z95 = 1.96
r3 = lambda x: round(float(x), 3)
meta = pd.read_csv(ROOT / "data/interim/A_joint_norm_input_20261112.csv", dtype={"sub_n": str}, low_memory=False)
regions = [c for c in meta.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]


dz = pd.read_csv(NJ / "A13_disease_z.csv.gz", dtype={"sub_n": str}); dz = dz[dz.db == "HBN"]
w = dz.pivot_table(index=["site", "sub_n", "group", "split"], columns="region", values="z").reset_index()
out = []
for s, g in w.groupby("site"):
    ref = g[(g.group == "TC") & (g.split == "train")]
    mu = ref[regions].mean(); sd = ref[regions].std(ddof=1) if len(ref) >= 10 else 1.0
    gg = g[~((g.group == "TC") & (g.split == "train"))].copy(); gg[regions] = (gg[regions] - mu) / sd; out.append(gg)
h = pd.concat(out, ignore_index=True).merge(meta[["site", "sub_n", "age", "sex", "dx_asd", "dx_adhd", "dx_ld", "dx_anx", "dx_dep"]], on=["site", "sub_n"], how="left")
h["burden"] = (h[regions].abs() > Z95).sum(axis=1); h["dx"] = (h.group != "TC").astype(int)
wisc = pd.read_csv(PH / "WISC.csv", encoding="utf-8-sig", usecols=["EID", "WISC_FSIQ"], low_memory=False)
wisc["EID"] = wisc.EID.str.strip().str.upper(); wisc = wisc.drop_duplicates("EID")
h = h.merge(wisc, left_on="sub_n", right_on="EID", how="left"); h["iq"] = pd.to_numeric(h.WISC_FSIQ, errors="coerce")
h["db"] = "HBN"

a = pd.read_csv(NJ / "A15_abide_adjusted_z.csv.gz", dtype={"sub_n": str})
a = a.merge(meta[meta.db == "ABIDE"][["site", "sub_n", "fiq"]], on=["site", "sub_n"], how="left")
a["iq"] = pd.to_numeric(a.fiq, errors="coerce"); a["dx"] = a.case; a["db"] = "ABIDE"


def nb_iq(d, extra=""):
    d = d[d.iq.notna()].copy(); d["iqz"] = (d.iq - d.iq.mean()) / d.iq.std()
    m = smf.glm(f"burden ~ iqz + age + C(sex) + C(site){extra}", data=d, family=sm.families.NegativeBinomial()).fit()
    c = np.exp(m.conf_int().loc["iqz"])
    return {"n": int(len(d)), "IRR_per_SD": r3(np.exp(m.params.iqz)), "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.iqz),
            "spearman_rho": r3(d[["iq", "burden"]].corr("spearman").iloc[0, 1])}


def pcor(d, region):
    d = d[d.iq.notna()].copy()
    X = pd.get_dummies(d[["age", "sex", "site"]], columns=["sex", "site"], drop_first=True).astype(float); X = sm.add_constant(X)
    ry = sm.OLS(d[region].to_numpy(float), X).fit().resid; rx = sm.OLS(d.iq.to_numpy(float), X).fit().resid
    r = float(np.corrcoef(rx, ry)[0, 1]); n = len(d)
    se = 1 / np.sqrt(n - X.shape[1] - 1); lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    from scipy.stats import t as tdist
    tt = r * np.sqrt((n - X.shape[1]) / (1 - r * r)); p = 2 * tdist.sf(abs(tt), n - X.shape[1])
    return {"n": n, "partial_r": r3(r), "ci95": [r3(lo), r3(hi)], "p": float(p)}


res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A25_cognition_development.py",
       "iq_available": {"HBN": int(h.iq.notna().sum()), "HBN_controls": int((h.iq.notna() & (h.dx == 0)).sum()),
                        "ABIDE": int(a.iq.notna().sum()), "ABIDE_controls": int((a.iq.notna() & (a.dx == 0)).sum())},
       "iq_mean_sd": {"HBN_controls": [r3(h.loc[h.dx == 0, "iq"].mean()), r3(h.loc[h.dx == 0, "iq"].std())],
                      "HBN_dx": [r3(h.loc[h.dx == 1, "iq"].mean()), r3(h.loc[h.dx == 1, "iq"].std())],
                      "ABIDE_controls": [r3(a.loc[a.dx == 0, "iq"].mean()), r3(a.loc[a.dx == 0, "iq"].std())],
                      "ABIDE_ASD": [r3(a.loc[a.dx == 1, "iq"].mean()), r3(a.loc[a.dx == 1, "iq"].std())]}}
cog = {"HBN_all": nb_iq(h, " + dx"), "HBN_controls": nb_iq(h[h.dx == 0]), "HBN_dx_only": nb_iq(h[h.dx == 1]),
       "ABIDE_all": nb_iq(a, " + dx"), "ABIDE_controls": nb_iq(a[a.dx == 0])}
q = multipletests([v["p"] for v in cog.values()], method="fdr_bh")[1]
for (k, v), qq in zip(cog.items(), q):
    v["p_fdr"] = float(qq)
res["burden_vs_iq"] = cog
pc = {}
for db, d in (("HBN", h), ("ABIDE", a)):
    for reg in ("G_BrainNotVent", "G_CorticalGM", "G_CerebralWM"):
        pc[f"{db}_{reg}"] = pcor(d, reg)
q = multipletests([v["p"] for v in pc.values()], method="fdr_bh")[1]
for (k, v), qq in zip(pc.items(), q):
    v["p_fdr"] = float(qq)
res["global_volume_z_vs_iq_partial_r"] = pc



dirs = {}
for db, d in (("HBN", h), ("ABIDE", a)):
    d = d.copy(); d["burden_neg"] = (d[regions] < -Z95).sum(axis=1); d["burden_pos"] = (d[regions] > Z95).sum(axis=1)
    for side in ("neg", "pos"):
        dd = d.assign(burden=d[f"burden_{side}"])
        dirs[f"{db}_{side}"] = nb_iq(dd, " + dx")
q = multipletests([v["p"] for v in dirs.values()], method="fdr_bh")[1]
for (k, v), qq in zip(dirs.items(), q):
    v["p_fdr"] = float(qq)
res["burden_vs_iq_by_direction"] = dirs


grp = {}
for db, d, defs in (("HBN", h, {"any_dx": h.dx == 1, "ASD": h.dx_asd == 1, "ADHD": h.dx_adhd == 1, "SLD": h.dx_ld == 1, "ANX": h.dx_anx == 1, "DEP": h.dx_dep == 1}), ("ABIDE", a, {"ASD": a.dx == 1})):
    d = d.copy(); d["burden_neg"] = (d[regions] < -Z95).sum(axis=1); d["burden_pos"] = (d[regions] > Z95).sum(axis=1)
    for lab, mask in defs.items():
        dd = d[mask | (d.dx == 0)].copy(); dd["case"] = mask[dd.index].astype(int)
        for side in ("neg", "pos"):
            m = smf.glm(f"burden_{side} ~ case + age + C(sex) + C(site)", data=dd, family=sm.families.NegativeBinomial()).fit()
            c = np.exp(m.conf_int().loc["case"])
            grp[f"{db}_{lab}_{side}"] = {"IRR": r3(np.exp(m.params.case)), "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.case),
                                         "case_median": r3(dd.loc[dd.case == 1, f"burden_{side}"].median()), "ctrl_median": r3(dd.loc[dd.case == 0, f"burden_{side}"].median())}
q = multipletests([v["p"] for v in grp.values()], method="fdr_bh")[1]
for (k, v), qq in zip(grp.items(), q):
    v["p_fdr"] = float(qq)
res["diagnosis_by_direction"] = grp


adj = {}
for db, d, defs in (("HBN", h, {"any_dx": h.dx == 1, "ASD": h.dx_asd == 1, "ADHD": h.dx_adhd == 1, "SLD": h.dx_ld == 1, "ANX": h.dx_anx == 1, "DEP": h.dx_dep == 1}), ("ABIDE", a, {"ASD": a.dx == 1})):
    d = d[d.iq.notna()].copy(); d["burden_neg"] = (d[regions] < -Z95).sum(axis=1); d["iqz"] = (d.iq - d.iq.mean()) / d.iq.std()
    for lab, mask in defs.items():
        mask = mask[d.index]
        dd = d[mask | (d.dx == 0)].copy(); dd["case"] = mask[dd.index].astype(int)
        out_ = {}
        for tag, f in (("without_iq", "burden_neg ~ case + age + C(sex) + C(site)"), ("with_iq", "burden_neg ~ case + iqz + age + C(sex) + C(site)")):
            m = smf.glm(f, data=dd, family=sm.families.NegativeBinomial()).fit(); c = np.exp(m.conf_int().loc["case"])
            out_[tag] = {"IRR": r3(np.exp(m.params.case)), "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.case)}
        adj[f"{db}_{lab}_neg"] = {"n_case": int(dd.case.sum()), "n_ctrl": int((dd.case == 0).sum()), **out_}
res["diagnosis_neg_adjusted_for_iq"] = adj


cc = h[(h.dx_asd == 1) | ((h.dx_adhd == 1) & (h.dx_asd == 0))].copy()
cc["case"] = cc.dx_asd.astype(int); cc["burden_neg"] = (cc[regions] < -Z95).sum(axis=1); cc["burden_pos"] = (cc[regions] > Z95).sum(axis=1)
cci = cc[cc.iq.notna()].copy(); cci["iqz"] = (cci.iq - cci.iq.mean()) / cci.iq.std()
ccres = {}
for side in ("neg", "pos"):
    for tag, d, f in (("all", cc, f"burden_{side} ~ case + age + C(sex) + C(site)"),
                      ("iq_subset_without_iq", cci, f"burden_{side} ~ case + age + C(sex) + C(site)"),
                      ("iq_subset_with_iq", cci, f"burden_{side} ~ case + iqz + age + C(sex) + C(site)")):
        m = smf.glm(f, data=d, family=sm.families.NegativeBinomial()).fit(); c = np.exp(m.conf_int().loc["case"])
        ccres[f"{side}_{tag}"] = {"n_ASD": int(d.case.sum()), "n_ADHD_noASD": int((d.case == 0).sum()), "IRR": r3(np.exp(m.params.case)),
                                  "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.case)}
res["hbn_asd_vs_adhd_noASD_by_direction"] = ccres


ctrl = pd.concat([h[h.dx == 0][["age", "sex", "site", "burden", "db"]], a[a.dx == 0][["age", "sex", "site", "burden", "db"]]], ignore_index=True)
ctrl["age5"] = ctrl.age / 5
m = smf.glm("burden ~ age5 + C(sex) + C(site)", data=ctrl, family=sm.families.NegativeBinomial()).fit()
c = np.exp(m.conf_int().loc["age5"])
res["controls_burden_vs_age"] = {"n": int(len(ctrl)), "IRR_per_5y": r3(np.exp(m.params.age5)), "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.age5),
                                 "any_extreme_pct_by_band": {b: r3((g.burden > 0).mean() * 100) for b, g in ctrl.groupby(pd.cut(ctrl.age, [5, 9, 13, 21.01], right=False).astype(str))}}
inter = {}
h["age5"] = h.age / 5
for lab, mask in (("any_dx", h.dx == 1), ("ASD", h.dx_asd == 1), ("ADHD", h.dx_adhd == 1)):
    d = h[mask | (h.dx == 0)].copy(); d["case"] = mask[d.index].astype(int)
    m = smf.glm("burden ~ case * age5 + C(sex) + C(site)", data=d, family=sm.families.NegativeBinomial()).fit()
    c = np.exp(m.conf_int().loc["case:age5"])
    inter[lab] = {"n_case": int(d.case.sum()), "interaction_ratio_per_5y": r3(np.exp(m.params["case:age5"])), "ci95": [r3(c[0]), r3(c[1])], "p": float(m.pvalues["case:age5"])}
res["hbn_diagnosis_by_age_interaction"] = inter
(ROOT / "results/A25_cognition_development.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
