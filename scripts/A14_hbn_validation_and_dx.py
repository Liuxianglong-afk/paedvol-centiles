"""Joint-centile validation, between-dataset validation, simulated local calibration and HBN diagnostic-group analyses.

Inputs: derivatives/norms_joint/A13_*.csv.gz (A13_gamlss_joint.R); data/interim/A_joint_norm_input_20261112.csv; HBN release 11 scales.
Definitions as in A03: coverage |z| < 1.645 (target 90%) and |z| < 1.96 (95%); local calibration re-centres on k local controls
(divided by their SD when k >= 10); out of range |z| > 1.96; burden = number of out-of-range regions.
Between datasets: ABIDE -> HBN for all ages and 5-7 / 7-13 / 13-21 years; HBN -> ABIDE summarised at 7-15 years.
HBN groups: each site calibrated on its model-building controls; comparison with held-out HBN controls; flags not mutually exclusive.
Negative binomial models adjusted for age, sex and site; regional OLS with Benjamini-Hochberg correction.
Discrimination (supplementary): z + age + sex, L2 logistic regression, leave-one-site-out across HBN sites; age + sex and z-only baselines.
Symptom scales: burden ~ scale (SD) + age + sex + site, negative binomial, Benjamini-Hochberg.
Outputs: results/A14_hbn_results.json; derivatives/norms_joint/A14_*.csv
Usage: python -I scripts/A14_hbn_validation_and_dx.py
"""
import datetime as dt
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NJ = ROOT / "derivatives/norms_joint"
PH = ROOT / "data/raw/hbn_pheno_R11_20261112/extracted/HBN_phenotypic_R11/HBN_phenotypic_R11"
spec = importlib.util.spec_from_file_location("feas", HERE / "07_feasibility_abide.py")
feas = importlib.util.module_from_spec(spec); spec.loader.exec_module(feas)
spec3 = importlib.util.spec_from_file_location("a03", HERE / "A03_norm_validation_and_asd.py")
a03 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(a03)
calib_stats, Z95 = a03.calib_stats, a03.Z95
SEEDS = [42, 7, 123, 2024]
FLAGS = {"ASD": "dx_asd", "ADHD": "dx_adhd", "ANX": "dx_anx", "SLD": "dx_ld", "DEP": "dx_dep"}
SCALES = {"SRS": ("SRS", "SRS_Total_T"), "SCQ": ("SCQ", "SCQ_Total"), "SWAN": ("SWAN", "SWAN_Total"),
          "SCARED_P": ("SCARED_P", "SCARED_P_Total"), "MFQ_P": ("MFQ_P", "MFQ_P_Total"),
          "CBCL_Total_T": ("CBCL", "CBCL_Total_T"), "CBCL_Int_T": ("CBCL", "CBCL_Int_T"), "CBCL_Ext_T": ("CBCL", "CBCL_Ext_T")}
r3 = lambda x: round(float(x), 3)


def stats_block(z):
    return {k: r3(v) for k, v in calib_stats(z).items()} | {"n_values": int(len(z))}


def age_band(a):
    return pd.cut(a, [5, 7, 13, 21.01], right=False, labels=["5-7", "7-13", "13-21"])


def main():
    meta = pd.read_csv(ROOT / "data/interim/A_joint_norm_input_20261112.csv", dtype={"sub_n": str}, low_memory=False)
    regions = [c for c in meta.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A14_hbn_validation_and_dx.py"}
    res["families"] = pd.read_csv(NJ / "A13_families.csv")["family"].value_counts().to_dict()


    lz = pd.read_csv(NJ / "A13_loso_z.csv.gz", dtype={"sub_n": str})
    lz = lz.merge(meta[["site", "sub_n", "age"]], on=["site", "sub_n"], how="left")
    res["loso_joint"] = {"pooled": stats_block(lz.z),
                         "by_db": {d: stats_block(g.z) for d, g in lz.groupby("db")},
                         "HBN_by_site": {s: stats_block(g.z) for s, g in lz[lz.db == "HBN"].groupby("site")},
                         "by_age_band": {str(b): stats_block(g.z) for b, g in lz.groupby(age_band(lz.age), observed=True)},
                         "fallback_fits_pct": r3((lz.fam_used != lz.region.map(pd.read_csv(NJ / "A13_families.csv").set_index("region").family)).mean() * 100)}
    pr = lz.groupby("region")["z"].apply(lambda s: pd.Series(calib_stats(s))).unstack()
    res["loso_joint"]["regions_sd_within_0.9_1.1"] = int(pr["sd"].between(0.9, 1.1).sum())
    pr.round(4).to_csv(NJ / "A14_loso_per_region.csv")


    xz = pd.read_csv(NJ / "A13_xdb_z.csv.gz", dtype={"sub_n": str})
    xz = xz.merge(meta[["site", "sub_n", "split"]], on=["site", "sub_n"], how="left")
    xres = {}
    for d, g in xz.groupby("dir"):
        if d == "HBN2ABIDE":
            g = g[g.age.between(7, 15)]
        blk = {"uncalibrated_all": stats_block(g.z),
               "uncalibrated_by_age_band": {str(b): stats_block(gg.z) for b, gg in g.groupby(age_band(g.age), observed=True)}}

        w = g.pivot_table(index=["site", "sub_n", "split"], columns="region", values="z").reset_index()
        adj = []
        for s, gs in w.groupby("site"):
            ref = gs[gs.split == "train"]; te = gs[gs.split == "test"]
            if len(ref) < 5 or len(te) == 0:
                continue
            mu = ref[regions].mean(); sd = ref[regions].std(ddof=1) if len(ref) >= 10 else 1.0
            adj.append(((te[regions] - mu) / sd).to_numpy().ravel())
        blk["locally_calibrated_test"] = stats_block(np.concatenate(adj)) if adj else None
        blk["n_subjects"] = int(g.drop_duplicates(["site", "sub_n"]).shape[0])
        xres[d] = blk
    res["cross_database"] = xres


    lz2 = lz.rename(columns={"db": "cohort"})
    adapt = {}
    for s in SEEDS:
        r, summ, n_sites = a03.adaptation(lz2[["site", "cohort", "sub_n", "region", "z"]], s)
        adapt[s] = summ
        if s == 42:
            res["adaptation_seed42"] = {f"k{k}_{m}": v for (k, m), v in summ.to_dict("index").items()}
            res["adaptation_n_sites"] = {str(k): int(v) for k, v in n_sites.items()}
            hb = r[r.site.str.startswith("HBN")].groupby(["k", "method"])[["abs_bias", "sd", "cov90"]].mean().round(4)
            res["adaptation_HBN_sites_seed42"] = {f"k{k}_{m}": v for (k, m), v in hb.to_dict("index").items()}
    cov = pd.DataFrame({s: adapt[s]["cov90"] for s in SEEDS})
    res["adaptation_cov90_across_seeds"] = {f"k{k}_{m}": [r3(v.mean()), r3(v.std(ddof=1))] for (k, m), v in cov.iterrows()}


    dz = pd.read_csv(NJ / "A13_disease_z.csv.gz", dtype={"sub_n": str})
    dz = dz[dz.db == "HBN"]
    w = dz.pivot_table(index=["site", "sub_n", "group", "split"], columns="region", values="z").reset_index()
    adj = []
    for s, g in w.groupby("site"):
        ref = g[(g.group == "TC") & (g.split == "train")]
        mu = ref[regions].mean(); sd = ref[regions].std(ddof=1) if len(ref) >= 10 else 1.0
        gg = g[~((g.group == "TC") & (g.split == "train"))].copy()
        gg[regions] = (gg[regions] - mu) / sd
        adj.append(gg)
    a = pd.concat(adj, ignore_index=True)
    a = a.merge(meta[["site", "sub_n", "age", "sex", "dx_primary_cat"] + list(FLAGS.values())], on=["site", "sub_n"], how="left")
    a["burden"] = (a[regions].abs() > Z95).sum(axis=1)
    a["is_tc"] = (a.group == "TC").astype(int)
    tcb = a.loc[a.is_tc == 1, "burden"]
    dres = {"n_TC_test": int(a.is_tc.sum()), "n_DX": int((a.is_tc == 0).sum()),
            "TC_burden_median_iqr": [r3(tcb.median()), r3(tcb.quantile(.25)), r3(tcb.quantile(.75))],
            "TC_any_extreme_pct": r3((tcb > 0).mean() * 100), "groups": {}}
    defs = {k: a[v] == 1 for k, v in FLAGS.items()}
    defs["ADHD_noASD"] = (a.dx_adhd == 1) & (a.dx_asd == 0)
    defs["ASD_noADHD"] = (a.dx_asd == 1) & (a.dx_adhd == 0)
    defs["ANY_DX"] = a.is_tc == 0
    reg_all = []
    for name, mask in defs.items():
        sub = a[mask | (a.is_tc == 1)].copy()
        sub["case"] = mask[sub.index].astype(int)
        nb = smf.glm("burden ~ case + age + C(sex) + C(site)", data=sub, family=sm.families.NegativeBinomial()).fit()
        ci = np.exp(nb.conf_int().loc["case"])
        cb = sub.loc[sub.case == 1, "burden"]
        g = {"n": int(sub.case.sum()), "burden_median_iqr": [r3(cb.median()), r3(cb.quantile(.25)), r3(cb.quantile(.75))],
             "any_extreme_pct": r3((cb > 0).mean() * 100), "IRR": r3(np.exp(nb.params["case"])),
             "IRR_95ci": [r3(ci[0]), r3(ci[1])], "p": float(nb.pvalues["case"])}
        if name in FLAGS or name in ("ADHD_noASD",):
            rows = []
            for r in regions:
                m = smf.ols(f"Q('{r}') ~ case + age + C(sex) + C(site)", data=sub).fit()
                x1, x0 = sub.loc[sub.case == 1, r], sub.loc[sub.case == 0, r]
                rows.append({"group": name, "region": r, "beta": m.params["case"], "p": m.pvalues["case"],
                             "cohen_d": (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2),
                             "case_mean_z": x1.mean(), "case_pct_extreme": (x1.abs() > Z95).mean() * 100})
            rr = pd.DataFrame(rows); rr["p_fdr"] = multipletests(rr.p, method="fdr_bh")[1]
            g["regional_n_fdr_sig"] = int((rr.p_fdr < 0.05).sum())
            g["regional_top5"] = rr.sort_values("p").head(5)[["region", "cohen_d", "p_fdr"]].round(4).to_dict("records")
            g["regional_abs_d_max"] = r3(rr.cohen_d.abs().max())
            reg_all.append(rr)
        dres["groups"][name] = g
    ps = [dres["groups"][k]["p"] for k in FLAGS]
    for k, q in zip(FLAGS, multipletests(ps, method="fdr_bh")[1]):
        dres["groups"][k]["p_fdr_across_5_dx"] = float(q)
    pd.concat(reg_all).round(5).to_csv(NJ / "A14_hbn_dx_regional.csv", index=False)


    disc = {}
    for name in list(FLAGS) + ["ADHD_noASD"]:
        sub = a[defs[name] | (a.is_tc == 1)]
        y = defs[name][sub.index].astype(int).to_numpy()
        X = np.column_stack([sub[regions].to_numpy(), sub.age, (sub.sex == "M").astype(float)])
        oof, _ = feas.loso(X, y, sub.site.to_numpy())
        disc[name] = feas.summarize(f"HBN {name} vs TC (z)", y, oof, sub.site.to_numpy())
        auc, lo, hi = feas.delong_ci(y, sub.burden.to_numpy() + 1e-6 * sub[regions].abs().sum(axis=1).to_numpy())
        disc[name]["burden_only_auc"] = [r3(auc), r3(lo), r3(hi)]

        Xb = np.column_stack([sub.age, (sub.sex == "M").astype(float)])
        oofb, _ = feas.loso(Xb, y, sub.site.to_numpy())
        ab_, lb, hb_ = feas.delong_ci(y, oofb)
        disc[name]["baseline_age_sex_auc"] = [r3(ab_), r3(lb), r3(hb_)]

        oofz, _ = feas.loso(sub[regions].to_numpy(), y, sub.site.to_numpy())
        az, lz_, hz = feas.delong_ci(y, oofz)
        disc[name]["z_only_auc"] = [r3(az), r3(lz_), r3(hz)]
        disc[name]["age_case_vs_tc_median"] = [r3(sub.age[y == 1].median()), r3(sub.age[y == 0].median())]
    dres["discrimination"] = disc


    sc_rows = []
    for key, (tab, col) in SCALES.items():
        t = pd.read_csv(PH / f"{tab}.csv", encoding="utf-8-sig", usecols=["EID", col], low_memory=False)
        t["EID"] = t.EID.str.strip().str.upper()
        t = t.drop_duplicates("EID")
        sub = a.merge(t, left_on="sub_n", right_on="EID", how="inner")
        sub = sub[pd.to_numeric(sub[col], errors="coerce").notna()].copy()
        sub["s"] = pd.to_numeric(sub[col]); sub["s"] = (sub.s - sub.s.mean()) / sub.s.std()
        nb = smf.glm("burden ~ s + age + C(sex) + C(site)", data=sub, family=sm.families.NegativeBinomial()).fit()
        ci = np.exp(nb.conf_int().loc["s"])
        sc_rows.append({"scale": key, "n": int(len(sub)), "IRR_per_SD": r3(np.exp(nb.params.s)),
                        "IRR_95ci": [r3(ci[0]), r3(ci[1])], "p": float(nb.pvalues.s),
                        "spearman_rho": r3(sub[["s", "burden"]].corr("spearman").iloc[0, 1])})
    q = multipletests([r["p"] for r in sc_rows], method="fdr_bh")[1]
    for r, qq in zip(sc_rows, q):
        r["p_fdr"] = float(qq)
    dres["scale_associations"] = sc_rows
    res["hbn_dx"] = dres
    a[["site", "group", "age", "sex", "burden", "dx_primary_cat"] + list(FLAGS.values())].to_csv(NJ / "A14_hbn_subject_level.csv", index=False)
    (ROOT / "results/A14_hbn_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float)[:9000])


if __name__ == "__main__":
    main()
