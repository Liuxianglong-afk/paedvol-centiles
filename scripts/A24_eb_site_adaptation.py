"""Local calibration at held-out sites: re-centring vs re-centring and rescaling vs empirical Bayes shrinkage.

Leave-one-site-out z scores of the joint model; k = 5/10/20/40 local controls; 100 draws per site; seeds 42/7/123/2024;
sites with at least k + 5 controls.
Shrinkage: the site offset of each region is B x local mean, B = k*tau2/(k*tau2 + sigma2), with tau2 (between-site variance of
site means, minus the mean sampling variance, floored at 0) and sigma2 (within-site variance) estimated from the other sites.
Also writes per-region tau2, sigma2 and B for k = 5/10/20/40 from all sites (Supplementary Table S8).
Outputs: results/A24_eb_adaptation.json; derivatives/norms_joint/A24_draws_seed42.csv.gz, A24_eb_parameters_by_region.csv
Usage: python -I scripts/A24_eb_site_adaptation.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NJ = ROOT / "derivatives/norms_joint"
KS, SEEDS, Z90, Z95 = [5, 10, 20, 40], [42, 7, 123, 2024], 1.6449, 1.96

lz = pd.read_csv(NJ / "A13_loso_z.csv.gz", dtype={"sub_n": str})
wide = lz.pivot_table(index=["site", "sub_n"], columns="region", values="z")
sites = {s: g.to_numpy() for s, g in wide.groupby(level="site")}
stats = {s: (Z.mean(0), Z.var(0, ddof=1), len(Z)) for s, Z in sites.items() if len(Z) >= 2}


def tau_sigma(exclude):
    M = np.array([v[0] for s, v in stats.items() if s != exclude])
    V = np.array([v[1] for s, v in stats.items() if s != exclude])
    N = np.array([v[2] for s, v in stats.items() if s != exclude])[:, None]
    tau2 = np.clip(M.var(0, ddof=1) - (V / N).mean(0), 0, None)
    sigma2 = ((N - 1) * V).sum(0) / (N - 1).sum()
    return tau2, sigma2


def metrics(ZE):
    return {"abs_bias": float(np.mean(np.abs(ZE.mean(0)))), "sd": float(np.mean(ZE.std(0, ddof=1))),
            "cov90": float(np.mean(np.abs(ZE) < Z90)), "cov95": float(np.mean(np.abs(ZE) < Z95))}


TS = {s: tau_sigma(s) for s in sites}
res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A24_eb_site_adaptation.py",
       "tau2_median_over_regions_all_sites": float(np.median(tau_sigma(None)[0])), "by_seed": {}}
allrows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed); rows = []
    for s, Z in sites.items():
        n = len(Z); tau2, sigma2 = TS[s]
        for k in KS:
            if n < k + 5:
                continue
            for _ in range(100):
                idx = rng.permutation(n); A, E = Z[idx[:k]], Z[idx[k:]]
                mA = A.mean(0); B = k * tau2 / (k * tau2 + sigma2)
                for lab, ZE in (("none", E), ("recentre", E - mA),
                                ("recentre_rescale", (E - mA) / (A.std(0, ddof=1) if k >= 10 else 1.0)),
                                ("eb_shrinkage", E - B * mA)):
                    rows.append({"seed": seed, "site": s, "k": k, "method": lab, **metrics(ZE)})
    df = pd.DataFrame(rows); allrows.append(df)
    res["by_seed"][str(seed)] = {f"k{k}_{m}": {c: round(float(v), 4) for c, v in r.items()}
                                 for (k, m), r in df.groupby(["k", "method"])[["abs_bias", "sd", "cov90", "cov95"]].mean().iterrows()}
D = pd.concat(allrows)
D[D.seed == 42].to_csv(NJ / "A24_draws_seed42.csv.gz", index=False)
summ = D.groupby(["seed", "k", "method"])[["abs_bias", "cov90"]].mean().groupby(["k", "method"]).agg(["mean", "std"])
res["across_seeds"] = {f"k{k}_{m}": {"abs_bias_mean": round(float(r[("abs_bias", "mean")]), 4), "abs_bias_sd": round(float(r[("abs_bias", "std")]), 4),
                                     "cov90_mean": round(float(r[("cov90", "mean")]), 4), "cov90_sd": round(float(r[("cov90", "std")]), 4)}
                       for (k, m), r in summ.iterrows()}

d42 = D[D.seed == 42].groupby(["site", "k", "method"])["abs_bias"].mean().unstack()
res["eb_vs_recentre_site_level_seed42"] = {f"k{k}": {"n_sites": int(len(g)), "pct_sites_eb_smaller_offset": round(float((g.eb_shrinkage < g.recentre).mean() * 100), 1),
                                                    "median_diff_eb_minus_recentre": round(float((g.eb_shrinkage - g.recentre).median()), 4)}
                                           for k, g in d42.groupby(level="k")}

t_all, s_all = tau_sigma(None)
eb_tab = pd.DataFrame({"region": wide.columns, "tau2": t_all.round(5), "sigma2": s_all.round(5)})
for k in KS:
    eb_tab[f"B_k{k}"] = (k * t_all / (k * t_all + s_all)).round(4)
eb_tab.to_csv(NJ / "A24_eb_parameters_by_region.csv", index=False)
res["B_median_by_k"] = {f"k{k}": round(float(eb_tab[f"B_k{k}"].median()), 3) for k in KS}
(ROOT / "results/A24_eb_adaptation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("tau2_median_over_regions_all_sites", "across_seeds", "eb_vs_recentre_site_level_seed42")}, indent=1))
