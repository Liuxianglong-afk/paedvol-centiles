"""Sensitivity analysis with volumes relative to intracranial volume (input from A26a, models from A13 into derivatives/norms_joint_icv).

Repeats leave-one-site-out and between-dataset calibration, the control reference (any out-of-range region, 95th centile)
and diagnostic-group incidence rate ratios (all, below-range and above-range regions), and ASD vs ADHD without ASD.
Output: results/A26_icv_sensitivity.json
Usage: python -I scripts/A26_icv_sensitivity.py
"""
import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
NI = ROOT / "derivatives/norms_joint_icv"
Z90, Z95 = 1.6449, 1.96
r3 = lambda x: round(float(x), 3)
meta = pd.read_csv(ROOT / "data/interim/A_joint_norm_input_20261112.csv", dtype={"sub_n": str}, low_memory=False)
regions = [c for c in meta.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]


def cal(z):
    z = np.asarray(z)
    return {"mean": r3(z.mean()), "sd": r3(z.std(ddof=1)), "cov90": r3(np.mean(np.abs(z) < Z90)), "cov95": r3(np.mean(np.abs(z) < Z95))}


res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A26_icv_sensitivity.py",
       "families": pd.read_csv(NI / "A13_families.csv")["family"].value_counts().to_dict()}
lz = pd.read_csv(NI / "A13_loso_z.csv.gz", dtype={"sub_n": str})
res["loso"] = cal(lz.z) | {"site_offset_median": r3(lz.groupby(["site", "region"]).z.mean().abs().groupby("site").mean().median()),
                           "regions_sd_0.9_1.1": int(lz.groupby("region").z.std().between(0.9, 1.1).sum())}
xz = pd.read_csv(NI / "A13_xdb_z.csv.gz", dtype={"sub_n": str})
res["xdb"] = {"ABIDE2HBN": cal(xz[xz.dir == "ABIDE2HBN"].z), "HBN2ABIDE_7_15": cal(xz[(xz.dir == "HBN2ABIDE") & xz.age.between(7, 15)].z)}

dz = pd.read_csv(NI / "A13_disease_z.csv.gz", dtype={"sub_n": str})
w = dz.pivot_table(index=["db", "site", "sub_n", "group", "split"], columns="region", values="z").reset_index()
adj = []
for (db, s), g in w.groupby(["db", "site"]):
    ref = g[(g.group == "TC") & (g.split == "train")]
    if len(ref) < 5:
        continue
    mu = ref[regions].mean(); sd = ref[regions].std(ddof=1) if len(ref) >= 10 else 1.0
    gg = g[~((g.group == "TC") & (g.split == "train"))].copy(); gg[regions] = (gg[regions] - mu) / sd; adj.append(gg)
a = pd.concat(adj, ignore_index=True).merge(meta[["site", "sub_n", "age", "sex", "dx_asd", "dx_adhd", "dx_anx", "dx_ld", "dx_dep"]], on=["site", "sub_n"], how="left")
a["burden"] = (a[regions].abs() > Z95).sum(axis=1); a["neg"] = (a[regions] < -Z95).sum(axis=1); a["pos"] = (a[regions] > Z95).sum(axis=1)
a["tc"] = (a.group == "TC").astype(int)
ctrl = a[a.tc == 1]
res["control_reference"] = {"n": int(len(ctrl)), "any_extreme_pct": r3((ctrl.burden > 0).mean() * 100), "median": r3(ctrl.burden.median()), "P95": r3(np.percentile(ctrl.burden, 95))}


def irr(d, mask, y="burden"):
    dd = d[mask | (d.tc == 1)].copy(); dd["case"] = mask[dd.index].astype(int)
    m = smf.glm(f"{y} ~ case + age + C(sex) + C(site)", data=dd, family=sm.families.NegativeBinomial()).fit(); c = np.exp(m.conf_int().loc["case"])
    return {"n": int(dd.case.sum()), "IRR": r3(np.exp(m.params.case)), "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.case)}


H, A = a[a.db == "HBN"], a[a.db == "ABIDE"]
p95 = res["control_reference"]["P95"]
grp = {}
for lab, d, mask in (("ABIDE_ASD", A, A.group == "ASD"), ("HBN_ASD", H, H.dx_asd == 1), ("HBN_ADHD", H, H.dx_adhd == 1),
                     ("HBN_ANX", H, H.dx_anx == 1), ("HBN_SLD", H, H.dx_ld == 1), ("HBN_DEP", H, H.dx_dep == 1)):
    grp[lab] = {"all": irr(d, mask), "neg": irr(d, mask, "neg"), "pos": irr(d, mask, "pos"),
                "above_P95_pct": r3((d[mask].burden > p95).mean() * 100)}
res["groups"] = grp
cc = H[(H.dx_asd == 1) | ((H.dx_adhd == 1) & (H.dx_asd == 0))].copy(); cc["case"] = cc.dx_asd.astype(int)
m = smf.glm("burden ~ case + age + C(sex) + C(site)", data=cc, family=sm.families.NegativeBinomial()).fit(); c = np.exp(m.conf_int().loc["case"])
res["hbn_asd_vs_adhd_noASD"] = {"IRR": r3(np.exp(m.params.case)), "IRR_95ci": [r3(c[0]), r3(c[1])], "p": float(m.pvalues.case)}
(ROOT / "results/A26_icv_sensitivity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
