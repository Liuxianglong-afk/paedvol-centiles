"""Leave-one-site-out calibration, simulated local calibration and ASD deviation analysis (ABIDE).

Inputs: derivatives/norms/A02_loso_z.csv.gz, A02_disease_z.csv.gz; data/interim/A_abide_norm_input_*.csv
Definitions:
  - Coverage: proportion with |z| < 1.645 (target 90%) and |z| < 1.96 (target 95%).
  - Local calibration: re-centre z on the mean of k local controls; for k >= 10 also divide by their SD
    (planned); re-centring only (added after the initial results).
  - Simulation: held-out sites with >= k + 5 controls; 100 random draws per site and k (seed 42; 7/123/2024).
  - ASD analysis: z re-centred per site on that site's model-building controls; sites with < 5 excluded.
  - Extreme deviation: |z| > 1.96; deviation burden = number of extreme regions per participant.
Outputs: results/A03_results.json; derivatives/norms/A03_*.csv
Usage: python -I scripts/A03_norm_validation_and_asd.py
"""
import datetime as dt
import importlib.util
import json
import sys
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
NORM = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "derivatives/norms"
spec = importlib.util.spec_from_file_location("feas", HERE / "07_feasibility_abide.py")
feas = importlib.util.module_from_spec(spec); spec.loader.exec_module(feas)
SEEDS = [42, 7, 123, 2024]
KS = [5, 10, 20, 40]
Z90, Z95 = 1.6449, 1.96


def latest(pattern):
    return sorted(ROOT.glob(pattern))[-1]


def calib_stats(z):
    z = np.asarray(z)
    return {"mean": float(np.mean(z)), "sd": float(np.std(z, ddof=1)),
            "cov90": float(np.mean(np.abs(z) < Z90)), "cov95": float(np.mean(np.abs(z) < Z95))}



def loso_calibration(lz):
    per_region = lz.groupby("region")["z"].apply(lambda s: pd.Series(calib_stats(s))).unstack()
    per_site = lz.groupby("site")["z"].apply(lambda s: pd.Series(calib_stats(s))).unstack()
    site_region_mean = lz.groupby(["site", "region"])["z"].mean()
    return per_region, per_site, {
        "pooled": calib_stats(lz["z"]),
        "regions_sd_within_0.9_1.1": int(per_region["sd"].between(0.9, 1.1).sum()),
        "n_regions": int(len(per_region)),
        "region_sd_median_iqr": [float(per_region["sd"].median()), float(per_region["sd"].quantile(.25)), float(per_region["sd"].quantile(.75))],
        "region_cov90_median_iqr": [float(per_region["cov90"].median()), float(per_region["cov90"].quantile(.25)), float(per_region["cov90"].quantile(.75))],
        "site_offset_abs_mean_median": float(site_region_mean.abs().groupby("site").mean().median()),
        "n_sites": int(lz["site"].nunique()), "n_subjects": int(lz.drop_duplicates(["site", "sub_n", "cohort"]).shape[0]),
    }



def adaptation(lz, seed, n_draw=100):
    rng = np.random.default_rng(seed)
    wide = lz.pivot_table(index=["site", "cohort", "sub_n"], columns="region", values="z")
    rows = []
    for site, g in wide.groupby(level="site"):
        Z = g.to_numpy()
        n = len(Z)
        for k in KS:
            if n < k + 5:
                continue
            for _ in range(n_draw):
                idx = rng.permutation(n)
                A, E = Z[idx[:k]], Z[idx[k:]]

                for label, ZE in (("none", E), ("adapted", (E - A.mean(0)) / (A.std(0, ddof=1) if k >= 10 else 1.0)),
                                  ("shift_only", E - A.mean(0))):
                    rows.append({"site": site, "k": k, "method": label,
                                 "abs_bias": float(np.mean(np.abs(ZE.mean(0)))),
                                 "sd": float(np.mean(ZE.std(0, ddof=1))),
                                 "cov90": float(np.mean(np.abs(ZE) < Z90)),
                                 "cov95": float(np.mean(np.abs(ZE) < Z95))})
    r = pd.DataFrame(rows)
    summ = r.groupby(["k", "method"])[["abs_bias", "sd", "cov90", "cov95"]].mean().round(4)
    n_sites = r.groupby("k")["site"].nunique().to_dict()
    return r, summ, n_sites



def disease(dz, meta):
    w = dz.pivot_table(index=["site", "cohort", "sub_n", "group", "split"], columns="region", values="z").reset_index()
    regions = [c for c in w.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]
    adj, excluded = [], []
    for site, g in w.groupby("site"):
        ref = g[(g.group == "TC") & (g.split == "train")]
        if len(ref) < 5:
            excluded.append({"site": site, "n_train_tc": int(len(ref)), "n_subjects": int(len(g))})
            continue
        mu = ref[regions].mean()
        sd = ref[regions].std(ddof=1) if len(ref) >= 10 else pd.Series(1.0, index=regions)
        gg = g[~((g.group == "TC") & (g.split == "train"))].copy()
        gg[regions] = (gg[regions] - mu) / sd
        adj.append(gg)
    a = pd.concat(adj, ignore_index=True)
    a = a.merge(meta[["site", "cohort", "sub_n", "age", "sex", "fiq"]], on=["site", "cohort", "sub_n"], how="left")
    a["asd"] = (a.group == "ASD").astype(int)
    a["burden"] = (a[regions].abs() > Z95).sum(axis=1)
    a["inst"] = a["site"].map(lambda s: feas.INSTITUTION.get(s, s.replace("ABIDEII-", "").split("_")[0]))
    out = {"n_TC_test": int((a.asd == 0).sum()), "n_ASD": int(a.asd.sum()),
           "excluded_sites_no_local_controls": excluded}

    nb = smf.glm("burden ~ asd + age + C(sex)", data=a, family=sm.families.NegativeBinomial()).fit()
    ci = np.exp(nb.conf_int().loc["asd"])
    out["burden"] = {"TC_median_iqr": [float(a.loc[a.asd == 0, "burden"].median()), float(a.loc[a.asd == 0, "burden"].quantile(.25)), float(a.loc[a.asd == 0, "burden"].quantile(.75))],
                     "ASD_median_iqr": [float(a.loc[a.asd == 1, "burden"].median()), float(a.loc[a.asd == 1, "burden"].quantile(.25)), float(a.loc[a.asd == 1, "burden"].quantile(.75))],
                     "TC_any_extreme_pct": float((a.loc[a.asd == 0, "burden"] > 0).mean() * 100),
                     "ASD_any_extreme_pct": float((a.loc[a.asd == 1, "burden"] > 0).mean() * 100),
                     "IRR": float(np.exp(nb.params["asd"])), "IRR_95ci": [float(ci[0]), float(ci[1])], "p": float(nb.pvalues["asd"])}

    rows = []
    for r in regions:
        m = smf.ols(f"Q('{r}') ~ asd + age + C(sex)", data=a).fit()
        d = (a.loc[a.asd == 1, r].mean() - a.loc[a.asd == 0, r].mean()) / np.sqrt((a.loc[a.asd == 1, r].var() + a.loc[a.asd == 0, r].var()) / 2)
        rows.append({"region": r, "beta": m.params["asd"], "p": m.pvalues["asd"], "cohen_d": d,
                     "ASD_mean_z": a.loc[a.asd == 1, r].mean(), "TC_mean_z": a.loc[a.asd == 0, r].mean(),
                     "ASD_pct_pos_extreme": (a.loc[a.asd == 1, r] > Z95).mean() * 100,
                     "ASD_pct_neg_extreme": (a.loc[a.asd == 1, r] < -Z95).mean() * 100,
                     "TC_pct_pos_extreme": (a.loc[a.asd == 0, r] > Z95).mean() * 100,
                     "TC_pct_neg_extreme": (a.loc[a.asd == 0, r] < -Z95).mean() * 100})
    reg = pd.DataFrame(rows)
    reg["p_fdr"] = multipletests(reg["p"], method="fdr_bh")[1]
    reg = reg.sort_values("p")
    out["regional"] = {"n_fdr_sig": int((reg.p_fdr < 0.05).sum()),
                       "top10": reg.head(10)[["region", "cohen_d", "beta", "p_fdr", "ASD_pct_pos_extreme", "ASD_pct_neg_extreme"]].round(4).to_dict("records"),
                       "max_ASD_pct_extreme_any_region": float((reg.ASD_pct_pos_extreme + reg.ASD_pct_neg_extreme).max())}

    X = a[regions].to_numpy()
    X = np.column_stack([X, a["age"], (a["sex"] == "M").astype(float)])
    y, groups = a["asd"].to_numpy(), a["inst"].to_numpy()
    oof, _ = feas.loso(X, y, groups)
    out["discrimination_z_features"] = feas.summarize("A | ASD vs TC (z features)", y, oof, groups)
    auc, lo, hi = feas.delong_ci(y, a["burden"].to_numpy() + 1e-6 * a[regions].abs().sum(axis=1).to_numpy())
    out["discrimination_burden_only"] = {"auc": round(auc, 3), "auc_95ci": [round(lo, 3), round(hi, 3)]}
    a["oof_z_model"] = oof
    return a, reg, out


def main():
    lz = pd.read_csv(NORM / "A02_loso_z.csv.gz", dtype={"sub_n": str})
    dz = pd.read_csv(NORM / "A02_disease_z.csv.gz", dtype={"sub_n": str})
    meta = pd.read_csv(latest("data/interim/A_abide_norm_input_*.csv"), dtype={"sub_n": str})
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A03_norm_validation_and_asd.py"}
    fam = pd.read_csv(NORM / "A02_families.csv")
    res["families"] = fam["family"].value_counts().to_dict()

    per_region, per_site, res["loso"] = loso_calibration(lz)
    per_region.round(4).to_csv(NORM / "A03_loso_per_region.csv")
    per_site.round(4).to_csv(NORM / "A03_loso_per_site.csv")

    adapt_all = {}
    for s in SEEDS:
        r, summ, n_sites = adaptation(lz, s)
        adapt_all[s] = summ
        if s == 42:
            r.to_csv(NORM / "A03_adaptation_draws_seed42.csv.gz", index=False)
            res["adaptation_seed42"] = {f"k{k}_{m}": v for (k, m), v in summ.to_dict("index").items()}
            res["adaptation_n_sites"] = {str(k): int(v) for k, v in n_sites.items()}
    cov = pd.DataFrame({s: adapt_all[s]["cov90"] for s in SEEDS})
    res["adaptation_cov90_across_seeds"] = {f"k{k}_{m}": [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)]
                                            for (k, m), v in cov.iterrows()}

    a, reg, res["asd"] = disease(dz, meta)
    reg.round(5).to_csv(NORM / "A03_asd_regional.csv", index=False)
    a[["site", "inst", "asd", "age", "sex", "burden", "oof_z_model"]].to_csv(NORM / "A03_asd_subject_level.csv", index=False)

    a[["site", "cohort", "sub_n", "asd", "age", "sex", "burden"] + [c for c in a.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]].to_csv(
        NORM / "A03_asd_adjusted_z.csv.gz", index=False)
    (ROOT / "results/A03_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("families", "loso", "adaptation_n_sites", "adaptation_cov90_across_seeds")}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res["asd"].items() if k != "excluded_sites_no_local_controls"}, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
