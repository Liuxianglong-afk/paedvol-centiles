"""Helper module: leave-one-institution-out cross-validation, DeLong CI and calibration for ABIDE.

Model: standardisation + L2 logistic regression; C chosen by institution-grouped 5-fold CV within the
training data (roc_auc). Validation: leave one institution out (ABIDE I/II sites of one institution grouped).
Random seed: 42 (bootstrap).
"""
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonize import SiteStandardizer  # noqa: E402
SEED = 42
TODAY = dt.date.today().strftime("%Y%m%d")


INSTITUTION = {
    "CMU_a": "CMU", "CMU_b": "CMU", "MaxMun_a": "MaxMun", "MaxMun_b": "MaxMun", "MaxMun_c": "MaxMun", "MaxMun_d": "MaxMun",
    "Leuven_1": "Leuven", "Leuven_2": "Leuven", "ABIDEII-KUL_3": "Leuven",
    "UCLA_1": "UCLA", "UCLA_2": "UCLA", "ABIDEII-UCLA_1": "UCLA",
    "UM_1": "UM", "UM_2": "UM", "NYU": "NYU", "ABIDEII-NYU_1": "NYU", "ABIDEII-NYU_2": "NYU",
    "KKI": "KKI", "ABIDEII-KKI_1": "KKI", "SDSU": "SDSU", "ABIDEII-SDSU_1": "SDSU",
    "OHSU": "OHSU", "ABIDEII-OHSU_1": "OHSU", "USM": "USM", "ABIDEII-USM_1": "USM",
    "Trinity": "TCD", "ABIDEII-TCD_1": "TCD",
}


def latest(pattern):
    return sorted(ROOT.glob(pattern))[-1]


def delong_ci(y, p, alpha=0.05):
    """DeLong variance and confidence interval for a single AUC."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = cmp.mean()
    v10, v01 = cmp.mean(axis=1), cmp.mean(axis=0)
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se))


def calibration(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=1e6).fit(lp[:, None], y).coef_[0, 0]

    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        a += (y - q).sum() / (q * (1 - q)).sum()
    return float(a), float(slope)


def load():
    ph = pd.read_csv(latest("data/interim/abide_pheno_harmonized_*.csv"), dtype={"sub_n": str})
    fv = pd.read_csv(latest("derivatives/features/abide_fastsurfer_volumes_*.csv"), dtype={"sub_n": str})
    df = fv.merge(ph[["cohort", "site", "sub_n", "dx_group", "age", "sex", "fiq", "in_age_5_21"]],
                  on=["cohort", "site", "sub_n"], how="left", validate="one_to_one")
    df = df[df["in_age_5_21"] & df["dx_group"].isin([1, 2])].copy()
    df["y"] = (df["dx_group"] == 1).astype(int)          # 1 = ASD
    df["institution"] = df["site"].map(lambda s: INSTITUTION.get(s, s.replace("ABIDEII-", "").split("_")[0]))
    return df


def features(df, supra_only=False):
    vcols = [c for c in df.columns if c.startswith("v_")]
    cbcols = [c for c in df.columns if c.startswith("cb_")]
    cols = vcols + ([] if supra_only else cbcols)
    if supra_only:
        cols = [c for c in cols if not any(k in c for k in ("Cerebellum", "Brain_Stem", "4th_Ventricle"))]
    cols = [c for c in cols if (df[c].fillna(0) <= 0).mean() <= 0.05]
    lm = np.log(df["g_MaskVol"])
    X = pd.DataFrame({c: np.log(df[c].clip(lower=1)) - lm for c in cols}, index=df.index)
    X["log_MaskVol"] = lm
    X["age"] = df["age"]
    X["sex_male"] = (df["sex"] == 1).astype(float)
    return X


def site_z(X, groups, cols):
    """Within-institution z standardisation (features only, no labels)."""
    X = X.copy()
    for g in np.unique(groups):
        idx = groups == g
        mu, sd = X.loc[idx, cols].mean(), X.loc[idx, cols].std().replace(0, 1).fillna(1)
        X.loc[idx, cols] = (X.loc[idx, cols] - mu) / sd
    return X


def loso(X, y, groups, harm=None):
    """harm = (column index, covariate matrix, batch): covariate effects fitted on training folds only; the held-out institution is corrected with its own data (no labels)."""
    oof = np.full(len(y), np.nan)
    chosen = []
    grid = {"logisticregression__C": np.logspace(-4, 1, 12)}
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        Xf = X.copy()
        if harm is not None:
            cols, cov, batch = harm
            ss = SiteStandardizer().fit(Xf[tr][:, cols], cov[tr], batch[tr])
            Xf[np.ix_(tr, cols)] = ss.transform(Xf[tr][:, cols], cov[tr], batch[tr])
            Xf[np.ix_(te, cols)] = ss.transform(Xf[te][:, cols], cov[te], batch[te])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        n_inner = min(5, len(np.unique(groups[tr])))
        gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=n_inner), scoring="roc_auc", n_jobs=4)
        gs.fit(Xf[tr], y[tr], groups=groups[tr])
        oof[te] = gs.predict_proba(Xf[te])[:, 1]
        chosen.append(float(gs.best_params_["logisticregression__C"]))
    return oof, chosen


def summarize(name, y, p, groups):
    auc, lo, hi = delong_ci(y, p)
    a, s = calibration(y, p)
    per = {}
    for g in np.unique(groups):
        m = groups == g
        if y[m].sum() >= 5 and (1 - y[m]).sum() >= 5:
            per[g] = round(roc_auc_score(y[m], p[m]), 3)
    out = {"auc": round(auc, 3), "auc_95ci": [round(lo, 3), round(hi, 3)], "brier": round(brier_score_loss(y, p), 4),
           "calibration_intercept": round(a, 3), "calibration_slope": round(s, 3),
           "per_institution_auc": per, "per_institution_auc_median": round(float(np.median(list(per.values()))), 3) if per else None}
    print(f"{name:42s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  median per-inst {out['per_institution_auc_median']}")
    return out


def run(df, label, supra_only=False):
    X = features(df, supra_only)
    y = df["y"].to_numpy()
    groups = df["institution"].to_numpy()
    res = {"n": int(len(df)), "n_asd": int(y.sum()), "n_tc": int((1 - y).sum()),
           "n_institutions": int(len(np.unique(groups))), "n_features": int(X.shape[1])}
    vol_cols = [c for c in X.columns if c not in ("age", "sex_male")]
    Xb = X[["age", "sex_male", "log_MaskVol"]]
    for nm, XX in [("baseline_age_sex_maskvol", Xb), ("volumes_no_harmonization", X),
                   ("volumes_site_z", site_z(X, groups, vol_cols))]:
        oof, cs = loso(XX.to_numpy(), y, groups)
        res[nm] = summarize(f"{label} | {nm}", y, oof, groups)
        res[nm]["chosen_C_median"] = float(np.median(cs))
        df[f"oof_{label}_{nm}"] = oof

    cols = [X.columns.get_loc(c) for c in vol_cols]
    cov = np.column_stack([df["age"], df["age"] ** 2, (df["sex"] == 1).astype(float)])
    oof, cs = loso(X.to_numpy(), y, groups, harm=(cols, cov, df["site"].to_numpy()))
    nm = "volumes_site_standardized_cov_preserving"
    res[nm] = summarize(f"{label} | {nm}", y, oof, groups)
    res[nm]["chosen_C_median"] = float(np.median(cs))
    df[f"oof_{label}_{nm}"] = oof
    return res


def main():
    np.random.seed(SEED)
    df = load()
    cb = [c for c in df.columns if c.startswith("cb_")]
    df["qc_fov_incomplete"] = df[cb].isna().any(axis=1) | (df[[c for c in cb if "Cortex" in c]].fillna(0) <= 0).any(axis=1)
    flow = {"eligible_with_fastsurfer": int(len(df)),
            "excluded_incomplete_fov": int(df["qc_fov_incomplete"].sum()),
            "excluded_by_institution": df.loc[df["qc_fov_incomplete"], "institution"].value_counts().to_dict()}
    main_df = df[~df["qc_fov_incomplete"]].copy()
    flow["analysed_main"] = int(len(main_df))
    print(json.dumps(flow, ensure_ascii=False))

    results = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
               "script": "scripts/07_feasibility_abide.py", "seed": SEED, "flow": flow,
               "main_all_features": run(main_df, "main"),
               "sensitivity_supratentorial_keep_fov": run(df.copy(), "supra", supra_only=True)}
    gate = results["main_all_features"]["volumes_site_standardized_cov_preserving"]
    results["feasibility_gate_signal"] = {
        "criterion": "lower 95% CI bound of pooled AUC > 0.5",
        "passed": bool(gate["auc_95ci"][0] > 0.5)}
    (ROOT / "results/feasibility_abide.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out = ROOT / "derivatives/models"; out.mkdir(parents=True, exist_ok=True)
    main_df.filter(regex="^(cohort|site|sub_n|institution|y|oof_)").to_csv(out / f"feasibility_abide_oof_{TODAY}.csv", index=False)
    print("gate passed:", results["feasibility_gate_signal"]["passed"])


if __name__ == "__main__":
    main()
