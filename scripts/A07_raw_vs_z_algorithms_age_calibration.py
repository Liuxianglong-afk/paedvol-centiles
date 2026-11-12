"""Additional analyses (defined after the initial results).

1. ASD discrimination with raw volumes and site-centred raw volumes vs deviation z scores
   (same participants and leave-one-institution-out L2 logistic regression as A03; paired DeLong test).
2. Nested search over feature sets {z, raw, raw_site} x classifiers {LR-L2, LR-L1, SVM-RBF, RF, HGB}
   x hyperparameters: outer leave-one-institution-out, inner GroupKFold(5) by institution. The best
   combination on the outer results ("direct") is also reported to show optimism.
3. Local calibration and age:
   a. site-by-age interaction per region (F test, Benjamini-Hochberg);
   b. at held-out sites with >= 20 controls, re-centring on the 10 youngest vs 10 random controls
      (100 draws, seed 42), evaluated on the remaining controls.
Inputs: derivatives/norms/A02_loso_z.csv.gz, A03_asd_adjusted_z.csv.gz; data/interim/A_abide_norm_input_*.csv
Output: results/A07_supplementary.json
Usage: python -I scripts/A07_raw_vs_z_algorithms_age_calibration.py
"""
import datetime as dt
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NORM = ROOT / "derivatives/norms"
spec = importlib.util.spec_from_file_location("feas", HERE / "07_feasibility_abide.py")
feas = importlib.util.module_from_spec(spec); spec.loader.exec_module(feas)
SEED = 42
Z90 = 1.6449
REG_PREFIX = ("G_", "S_", "V_", "C_", "CB")


def latest(pattern):
    return sorted(ROOT.glob(pattern))[-1]



meta = pd.read_csv(latest("data/interim/A_abide_norm_input_*.csv"), dtype={"sub_n": str})
a = pd.read_csv(NORM / "A03_asd_adjusted_z.csv.gz", dtype={"sub_n": str})
regions = [c for c in a.columns if c.startswith(REG_PREFIX)]
raw = meta.set_index(["site", "cohort", "sub_n"])[regions]
a = a.set_index(["site", "cohort", "sub_n"])
Xraw = raw.loc[a.index].to_numpy(float)
a = a.reset_index()
a["inst"] = a["site"].map(lambda s: feas.INSTITUTION.get(s, s.replace("ABIDEII-", "").split("_")[0]))
y = a["asd"].to_numpy()
groups = a["inst"].to_numpy()
cov = np.column_stack([a["age"], (a["sex"] == "M").astype(float)])
Xz = np.column_stack([a[regions].to_numpy(float), cov])
Xr = np.column_stack([Xraw, cov])


tr = meta.drop(columns=[c for c in ("group", "split") if c in meta.columns]).merge(pd.read_csv(NORM / "A02_disease_z.csv.gz", dtype={"sub_n": str},
                            usecols=lambda c: c in ("site", "cohort", "sub_n", "group", "split")).drop_duplicates(),
                on=["site", "cohort", "sub_n"], how="inner")
tr = tr[(tr.group == "TC") & (tr.split == "train")]
site_mu = tr.groupby("site")[regions].mean()
Xrs_vol = Xraw - site_mu.loc[a["site"]].to_numpy(float)
Xrs = np.column_stack([Xrs_vol, cov])
FEATS = {"z": Xz, "raw": Xr, "raw_site": Xrs}
res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
       "script": "scripts/A07_raw_vs_z_algorithms_age_calibration.py",
       "status": "post hoc / exploratory (2026-11-04)",
       "n": {"ASD": int(y.sum()), "TC": int((y == 0).sum()), "institutions": int(len(set(groups)))}}


same = {}
for name, X in FEATS.items():
    oof, _ = feas.loso(X, y, groups)
    s = feas.summarize(f"A07 | {name}", y, oof, groups)
    same[name] = {k: s[k] for k in ("auc", "auc_95ci", "brier", "calibration_intercept", "calibration_slope", "per_institution_auc_median")}
    same[name]["_oof"] = oof
res["same_setting_as_A03"] = {k: {kk: vv for kk, vv in v.items() if kk != "_oof"} for k, v in same.items()}

res["self_check_z_auc_equals_A03"] = same["z"]["auc"]


def delong_diff(y, p1, p2):
    """Paired DeLong test for the difference between two AUCs."""
    pos, neg = p1[y == 1], p1[y == 0]
    def v(p):
        ps, ns = p[y == 1], p[y == 0]
        cmp = (ps[:, None] > ns[None, :]).astype(float) + 0.5 * (ps[:, None] == ns[None, :])
        return cmp.mean(1), cmp.mean(0), cmp.mean()
    v10a, v01a, a1 = v(p1); v10b, v01b, a2 = v(p2)
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / len(v10a) + s01 / len(v01a)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    zz = (a1 - a2) / np.sqrt(var)
    from scipy.stats import norm
    return {"diff": round(float(a1 - a2), 4), "ci95": [round(float(a1 - a2 - 1.96 * np.sqrt(var)), 4), round(float(a1 - a2 + 1.96 * np.sqrt(var)), 4)],
            "p": float(2 * norm.sf(abs(zz)))}


res["delong_z_minus_raw"] = delong_diff(y, same["z"]["_oof"], same["raw"]["_oof"])
res["delong_z_minus_raw_site"] = delong_diff(y, same["z"]["_oof"], same["raw_site"]["_oof"])


def models():
    out = []
    for C in (0.01, 0.1, 1.0):
        out.append((f"lr_l2_C{C}", make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))))
        out.append((f"lr_l1_C{C}", make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=5000))))
        out.append((f"svm_rbf_C{C}", make_pipeline(StandardScaler(), SVC(C=C, probability=False, random_state=SEED))))
    for d in (3, None):
        out.append((f"rf_depth{d}", RandomForestClassifier(n_estimators=300, max_depth=d, min_samples_leaf=5, n_jobs=-1, random_state=SEED)))
    for lr in (0.03, 0.1):
        out.append((f"hgb_lr{lr}", HistGradientBoostingClassifier(learning_rate=lr, max_depth=3, max_iter=200, random_state=SEED)))
    return out


MODELS = models()
CONFIGS = [(f, m, est) for f in FEATS for m, est in MODELS]


def score(est, X):
    return est.decision_function(X) if hasattr(est, "decision_function") else est.predict_proba(X)[:, 1]


def fit_pred(est, Xtr, ytr, Xte):
    e = clone(est).fit(Xtr, ytr)
    return score(e, Xte)


def rank01(p):
    from scipy.stats import rankdata
    return rankdata(p) / len(p)


insts = np.unique(groups)

direct = {}
for f, m, est in CONFIGS:
    X = FEATS[f]; oof = np.zeros(len(y))
    for g in insts:
        te = groups == g
        oof[te] = rank01(fit_pred(est, X[~te], y[~te], X[te]))
    direct[f"{f}|{m}"] = float(roc_auc_score(y, oof))
best_direct = max(direct, key=direct.get)


nested_oof = np.zeros(len(y)); chosen = []
for g in insts:
    te = groups == g
    trn = ~te
    gkf = GroupKFold(n_splits=5)
    inner = {}
    for f, m, est in CONFIGS:
        X = FEATS[f][trn]; yy = y[trn]; gg = groups[trn]
        p = np.zeros(trn.sum())
        for itr, ite in gkf.split(X, yy, gg):
            p[ite] = rank01(fit_pred(est, X[itr], yy[itr], X[ite]))
        inner[(f, m)] = roc_auc_score(yy, p)
    f_b, m_b = max(inner, key=inner.get)
    est_b = dict((m, e) for _, m, e in [(None, mm, ee) for mm, ee in MODELS])[m_b]
    nested_oof[te] = rank01(fit_pred(est_b, FEATS[f_b][trn], y[trn], FEATS[f_b][te]))
    chosen.append(f"{f_b}|{m_b}")
auc_n, lo_n, hi_n = feas.delong_ci(y, nested_oof)
res["algorithm_search"] = {
    "n_configs": len(CONFIGS),
    "configs_auc_direct_top10": dict(sorted(direct.items(), key=lambda kv: -kv[1])[:10]),
    "best_direct": {"config": best_direct, "auc": round(direct[best_direct], 4)},
    "nested": {"auc": round(auc_n, 4), "auc_95ci": [round(lo_n, 4), round(hi_n, 4)],
               "chosen_counts": pd.Series(chosen).value_counts().to_dict()},
    "note": "OOF scores are ranked within each held-out institution before pooling; direct = best combination chosen on all outer results (optimistic)",
}


lz = pd.read_csv(NORM / "A02_loso_z.csv.gz", dtype={"sub_n": str})
lz = lz.merge(meta[["site", "cohort", "sub_n", "age", "sex"]], on=["site", "cohort", "sub_n"], how="left")
ps, r2 = [], []
for r, g in lz.groupby("region"):
    g = g[g.groupby("site")["z"].transform("size") >= 10]
    m0 = smf.ols("z ~ C(site)", data=g).fit()
    m1 = smf.ols("z ~ C(site) + C(site):age", data=g).fit()
    ftest = m1.compare_f_test(m0)
    ps.append(ftest[1]); r2.append(m1.rsquared - m0.rsquared)
q = multipletests(ps, method="fdr_bh")[1]
res["age_by_site_interaction"] = {"n_regions": len(ps), "n_fdr_sig": int((q < 0.05).sum()),
                                  "delta_r2_median_iqr": [float(np.median(r2)), float(np.quantile(r2, .25)), float(np.quantile(r2, .75))],
                                  "sites_with_ge10_controls": int((lz.drop_duplicates(["site", "sub_n", "cohort"]).groupby("site").size() >= 10).sum())}

subj = lz.pivot_table(index=["site", "cohort", "sub_n", "age"], columns="region", values="z").reset_index()
rng = np.random.default_rng(SEED)
rows = []
for site, g in subj.groupby("site"):
    if len(g) < 20:
        continue
    Z = g[sorted(lz.region.unique())].to_numpy(); ages = g["age"].to_numpy()
    young = np.argsort(ages)[:10]
    for scheme in ("youngest10", "random10"):
        reps = 1 if scheme == "youngest10" else 100
        for _ in range(reps):
            idx = young if scheme == "youngest10" else rng.choice(len(g), 10, replace=False)
            rest = np.setdiff1d(np.arange(len(g)), idx)
            zc = Z[rest] - Z[idx].mean(0)
            rows.append({"site": site, "scheme": scheme, "abs_bias": float(np.abs(zc.mean(0)).mean()),
                         "cov90": float((np.abs(zc) < Z90).mean()),
                         "age_gap": float(ages[rest].mean() - ages[idx].mean())})
cal = pd.DataFrame(rows).groupby(["site", "scheme"]).mean(numeric_only=True).reset_index()
w = cal.pivot(index="site", columns="scheme", values="abs_bias")
res["local_controls_age_spread"] = {
    "n_sites": int(len(w)),
    "abs_bias_mean": cal.groupby("scheme")["abs_bias"].mean().round(4).to_dict(),
    "cov90_mean": cal.groupby("scheme")["cov90"].mean().round(4).to_dict(),
    "age_gap_mean": cal.groupby("scheme")["age_gap"].mean().round(2).to_dict(),
    "youngest_minus_random_abs_bias_median": float((w["youngest10"] - w["random10"]).median()),
    "sites_youngest_worse": int((w["youngest10"] > w["random10"]).sum()),
}

out = ROOT / "results/A07_supplementary.json"
out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "algorithm_search"}, indent=1, ensure_ascii=False, default=float))
print(json.dumps(res["algorithm_search"], indent=1, ensure_ascii=False, default=float))
