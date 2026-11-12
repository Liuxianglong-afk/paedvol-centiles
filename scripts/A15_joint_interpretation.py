"""Interpretation of out-of-range regions with the joint centiles.

1. ABIDE ASD re-analysed with the joint model (sites calibrated on ABIDE model-building controls; sites with <5 excluded).
2. Control reference: held-out controls of both datasets; 95th centile of the number of out-of-range regions; share of each group above it.
3. Regional pattern: correlations of Cohen's d maps across datasets (descriptive; regions are not independent).
4. Direct case-case comparison in HBN: ASD vs ADHD without ASD (no shared control group).
Outputs: results/A15_interpretation.json; derivatives/norms_joint/A15_*.csv
Usage: python -I scripts/A15_joint_interpretation.py
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
Z95 = 1.96
r3 = lambda x: round(float(x), 3)

meta = pd.read_csv(ROOT / "data/interim/A_joint_norm_input_20261112.csv", dtype={"sub_n": str}, low_memory=False)
regions = [c for c in meta.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]
res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A15_joint_interpretation.py"}


dz = pd.read_csv(NJ / "A13_disease_z.csv.gz", dtype={"sub_n": str})
dz = dz[dz.db == "ABIDE"]
w = dz.pivot_table(index=["site", "sub_n", "group", "split"], columns="region", values="z").reset_index()
adj, excl = [], []
for s, g in w.groupby("site"):
    ref = g[(g.group == "TC") & (g.split == "train")]
    if len(ref) < 5:
        excl.append({"site": s, "n_train_tc": int(len(ref)), "n": int(len(g))}); continue
    mu = ref[regions].mean(); sd = ref[regions].std(ddof=1) if len(ref) >= 10 else 1.0
    gg = g[~((g.group == "TC") & (g.split == "train"))].copy()
    gg[regions] = (gg[regions] - mu) / sd
    adj.append(gg)
a = pd.concat(adj, ignore_index=True).merge(meta[["site", "sub_n", "age", "sex", "fiq"]], on=["site", "sub_n"], how="left")
a["case"] = (a.group == "ASD").astype(int)
a["burden"] = (a[regions].abs() > Z95).sum(axis=1)
nb = smf.glm("burden ~ case + age + C(sex) + C(site)", data=a, family=sm.families.NegativeBinomial()).fit()
ci = np.exp(nb.conf_int().loc["case"])
rows = []
for r in regions:
    m = smf.ols(f"Q('{r}') ~ case + age + C(sex) + C(site)", data=a).fit()
    x1, x0 = a.loc[a.case == 1, r], a.loc[a.case == 0, r]
    rows.append({"group": "ABIDE_ASD", "region": r, "beta": m.params["case"], "p": m.pvalues["case"],
                 "cohen_d": (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2),
                 "case_pct_pos": (x1 > Z95).mean() * 100, "case_pct_neg": (x1 < -Z95).mean() * 100})
ra = pd.DataFrame(rows); ra["p_fdr"] = multipletests(ra.p, method="fdr_bh")[1]
ra.round(5).to_csv(NJ / "A15_abide_asd_regional.csv", index=False)
res["abide_asd"] = {"n_ASD": int(a.case.sum()), "n_TC_test": int((a.case == 0).sum()), "excluded_sites": excl,
                    "IRR": r3(np.exp(nb.params.case)), "IRR_95ci": [r3(ci[0]), r3(ci[1])], "p": float(nb.pvalues.case),
                    "n_fdr_sig": int((ra.p_fdr < 0.05).sum()),
                    "fdr_sig": ra[ra.p_fdr < 0.05].sort_values("p")[["region", "cohen_d", "p_fdr"]].round(4).to_dict("records"),
                    "max_abs_d": r3(ra.cohen_d.abs().max()),
                    "max_pct_extreme_any_region": r3((ra.case_pct_pos + ra.case_pct_neg).max())}
a[["site", "sub_n", "case", "age", "sex", "burden"] + regions].to_csv(NJ / "A15_abide_adjusted_z.csv.gz", index=False)


h = pd.read_csv(NJ / "A14_hbn_subject_level.csv")
ctrl = pd.concat([a.loc[a.case == 0, ["age", "burden"]].assign(db="ABIDE"),
                  h.loc[h.group == "TC", ["age", "burden"]].assign(db="HBN")], ignore_index=True)
p95 = float(np.percentile(ctrl.burden, 95))
band = pd.cut(ctrl.age, [5, 9, 13, 21.01], right=False, labels=["5-9", "9-13", "13-21"])
res["control_burden_reference"] = {
    "n": int(len(ctrl)), "by_db_n": ctrl.db.value_counts().to_dict(),
    "median_iqr": [r3(ctrl.burden.median()), r3(ctrl.burden.quantile(.25)), r3(ctrl.burden.quantile(.75))],
    "P90": r3(np.percentile(ctrl.burden, 90)), "P95": r3(p95), "P99": r3(np.percentile(ctrl.burden, 99)),
    "any_extreme_pct": r3((ctrl.burden > 0).mean() * 100),
    "by_db": {d: {"median": r3(g.burden.median()), "P95": r3(np.percentile(g.burden, 95)), "any_extreme_pct": r3((g.burden > 0).mean() * 100)} for d, g in ctrl.groupby("db")},
    "by_age_band": {str(b): {"n": int(len(g)), "median": r3(g.burden.median()), "P95": r3(np.percentile(g.burden, 95))} for b, g in ctrl.groupby(band, observed=True)},
    "expected_any_if_independent_pct": r3((1 - 0.95 ** len(regions)) * 100),
}
grp = {"Controls (ABIDE)": a.loc[a.case == 0, "burden"], "ASD (ABIDE)": a.loc[a.case == 1, "burden"],
       "Controls (HBN)": h.loc[h.group == "TC", "burden"]}
for lab, col in (("ASD (HBN)", "dx_asd"), ("ADHD (HBN)", "dx_adhd"), ("Anxiety (HBN)", "dx_anx"),
                 ("Specific learning disorder (HBN)", "dx_ld"), ("Depression (HBN)", "dx_dep")):
    grp[lab] = h.loc[h[col] == 1, "burden"]
res["groups_vs_control_P95"] = {k: {"n": int(len(v)), "median": r3(v.median()), "q1": r3(v.quantile(.25)), "q3": r3(v.quantile(.75)),
                                    "any_extreme_pct": r3((v > 0).mean() * 100), "above_P95_pct": r3((v > p95).mean() * 100)}
                                for k, v in grp.items()}
pd.concat([v.rename("burden").to_frame().assign(group=k) for k, v in grp.items()]).to_csv(NJ / "A15_burden_by_group.csv", index=False)


hr = pd.read_csv(NJ / "A14_hbn_dx_regional.csv")
D = pd.DataFrame({"ABIDE_ASD": ra.set_index("region").cohen_d})
for g in ("ASD", "ADHD", "ANX", "SLD", "DEP", "ADHD_noASD"):
    D["HBN_" + g] = hr[hr.group == g].set_index("region").cohen_d
D.round(5).to_csv(NJ / "A15_effect_maps.csv")
pairs = [("ABIDE_ASD", "HBN_ASD"), ("HBN_ASD", "HBN_ADHD"), ("HBN_ASD", "HBN_ADHD_noASD"), ("HBN_ASD", "HBN_ANX"),
         ("HBN_ASD", "HBN_SLD"), ("HBN_ASD", "HBN_DEP"), ("HBN_ADHD", "HBN_ANX"), ("ABIDE_ASD", "HBN_ADHD")]
res["effect_map_correlation"] = {f"{x}~{y}": {"pearson": r3(D[x].corr(D[y])), "spearman": r3(D[x].corr(D[y], method="spearman"))} for x, y in pairs}
sig_both = set(ra[ra.p_fdr < 0.05].region) & set(hr[(hr.group == "ASD") & (hr.p_fdr < 0.05)].region)
res["asd_fdr_regions_shared_ABIDE_HBN"] = sorted(sig_both)
res["hbn_asd_fdr_sig"] = hr[(hr.group == "ASD") & (hr.p_fdr < 0.05)][["region", "cohen_d", "p_fdr"]].round(4).to_dict("records")



hz = pd.read_csv(NJ / "A13_disease_z.csv.gz", dtype={"sub_n": str}); hz = hz[hz.db == "HBN"]
hw = hz.pivot_table(index=["site", "sub_n", "group", "split"], columns="region", values="z").reset_index()
ha = []
for s, g in hw.groupby("site"):
    ref = g[(g.group == "TC") & (g.split == "train")]
    mu = ref[regions].mean(); sd = ref[regions].std(ddof=1) if len(ref) >= 10 else 1.0
    gg = g[g.group != "TC"].copy(); gg[regions] = (gg[regions] - mu) / sd; ha.append(gg)
ha = pd.concat(ha, ignore_index=True).merge(meta[["site", "sub_n", "age", "sex", "dx_asd", "dx_adhd"]], on=["site", "sub_n"], how="left")
cc = ha[(ha.dx_asd == 1) | ((ha.dx_adhd == 1) & (ha.dx_asd == 0))].copy()
cc["case"] = cc.dx_asd.astype(int); cc["burden"] = (cc[regions].abs() > Z95).sum(axis=1)
nb2 = smf.glm("burden ~ case + age + C(sex) + C(site)", data=cc, family=sm.families.NegativeBinomial()).fit()
ci2 = np.exp(nb2.conf_int().loc["case"])
rr = []
for r in regions:
    m = smf.ols(f"Q('{r}') ~ case + age + C(sex) + C(site)", data=cc).fit()
    x1, x0 = cc.loc[cc.case == 1, r], cc.loc[cc.case == 0, r]
    rr.append({"region": r, "p": m.pvalues["case"], "cohen_d": (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2)})
rr = pd.DataFrame(rr); rr["p_fdr"] = multipletests(rr.p, method="fdr_bh")[1]
res["hbn_asd_vs_adhd_noASD"] = {"n_ASD": int(cc.case.sum()), "n_ADHD_noASD": int((cc.case == 0).sum()),
    "burden_IRR": r3(np.exp(nb2.params.case)), "burden_IRR_95ci": [r3(ci2[0]), r3(ci2[1])], "burden_p": float(nb2.pvalues.case),
    "n_fdr_sig": int((rr.p_fdr < 0.05).sum()), "max_abs_d": r3(rr.cohen_d.abs().max()),
    "fdr_sig": rr[rr.p_fdr < 0.05].sort_values("p")[["region", "cohen_d", "p_fdr"]].round(4).to_dict("records")}
res["note_within_hbn_map_correlations"] = "inflated by shared HBN control group; descriptive only"

(ROOT / "results/A15_interpretation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
