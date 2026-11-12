"""Prepare normative-model input from ABIDE I/II FastSurfer outputs.

Regions: 7 global measures, 16 subcortical structures (left/right), third and fourth ventricles,
brainstem, 62 DKT cortical regions, and left/right cerebellar cortex and white matter (92 regions, cm3).
Quality control: incomplete cerebellar coverage (any cerebellar lobule missing or zero cerebellar cortex)
is excluded; any non-ventricular region <25% or >300% of the sex-specific median is excluded.
Controls are split within site into a model-building half (train) and a held-out half (test), seed 42;
all participants with ASD are test.
Outputs:
  data/interim/A_abide_norm_input_<YYYYMMDD>.csv  (participant level, not distributed)
  results/A01_norm_input_summary.json
Usage: python -I scripts/A01_prepare_norm_input.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TODAY = dt.date.today().strftime("%Y%m%d")
SUBCORT = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens_area", "VentralDC"]


def latest(pattern):
    return sorted(ROOT.glob(pattern))[-1]


def build_regions(f):
    ctx = [c for c in f.columns if c.startswith("v_ctx_")]
    r = pd.DataFrame(index=f.index)
    r["G_IntracranialMask"] = f["g_MaskVol"]
    r["G_BrainNotVent"] = f["g_BrainSegVolNotVent"]
    r["G_CorticalGM"] = f[ctx].sum(axis=1)
    r["G_CerebralWM"] = f["g_CerebralWhiteMatterVol"]
    r["G_SubcorticalGM"] = f["g_SubCortGrayVol"]
    r["G_LateralVentricles"] = f["v_Left_Lateral_Ventricle"] + f["v_Right_Lateral_Ventricle"]
    r["G_Cerebellum"] = f[["v_Left_Cerebellum_Cortex", "v_Right_Cerebellum_Cortex",
                           "v_Left_Cerebellum_White_Matter", "v_Right_Cerebellum_White_Matter"]].sum(axis=1)
    for s in SUBCORT:
        for h in ("Left", "Right"):
            r[f"S_{h}_{s}"] = f[f"v_{h}_{s}"]
    r["V_3rd_Ventricle"] = f["v_3rd_Ventricle"]
    r["V_4th_Ventricle"] = f["v_4th_Ventricle"]
    r["S_Brain_Stem"] = f["v_Brain_Stem"]
    for c in ctx:
        r["C_" + c[len("v_ctx_"):]] = f[c]
    for h in ("Left", "Right"):
        r[f"CB_{h}_Cortex"] = f[f"v_{h}_Cerebellum_Cortex"]
        r[f"CB_{h}_WM"] = f[f"v_{h}_Cerebellum_White_Matter"]
    return r / 1000.0  # cm³


def main():
    ph = pd.read_csv(latest("data/interim/abide_pheno_harmonized_*.csv"), dtype={"sub_n": str})
    fv = pd.read_csv(latest("derivatives/features/abide_fastsurfer_volumes_*.csv"), dtype={"sub_n": str})
    df = fv.merge(ph[["cohort", "site", "sub_n", "dx_group", "age", "sex", "fiq", "in_age_5_21"]],
                  on=["cohort", "site", "sub_n"], how="left", validate="one_to_one")
    n0 = len(df)
    df = df[df["in_age_5_21"] & df["dx_group"].isin([1, 2])].copy()
    cb = [c for c in df.columns if c.startswith("cb_")]
    fov_bad = df[cb].isna().any(axis=1) | (df[[c for c in cb if "Cortex" in c]].fillna(0) <= 0).any(axis=1)
    flow = {"fastsurfer_success": int(n0), "eligible_age_dx": int(len(df)), "excluded_incomplete_fov": int(fov_bad.sum())}
    df = df[~fov_bad].copy()
    regions = build_regions(df)


    nonvent = [c for c in regions.columns if "Ventric" not in c]
    med = regions[nonvent].groupby(df["sex"].to_numpy()).transform("median")
    ratio = regions[nonvent] / med.to_numpy()
    gross = ((ratio < 0.25) | (ratio > 3.0)).any(axis=1).to_numpy()
    flow["excluded_gross_volume_outlier"] = int(gross.sum())
    flow["gross_outlier_by_site"] = df.loc[gross, "site"].value_counts().to_dict()
    df, regions = df[~gross].copy(), regions[~gross]
    out = pd.concat([df[["cohort", "site", "sub_n", "age", "sex", "fiq"]].reset_index(drop=True),
                     regions.reset_index(drop=True)], axis=1)
    out["group"] = np.where(df["dx_group"].to_numpy() == 1, "ASD", "TC")
    out["sex"] = np.where(out["sex"] == 1, "M", "F")

    rng = np.random.default_rng(SEED)
    out["split"] = "test"
    for s, idx in out[out["group"] == "TC"].groupby("site").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        out.loc[idx[: int(np.ceil(len(idx) / 2))], "split"] = "train"
    flow.update({"analysed": int(len(out)), "TC": int((out.group == "TC").sum()), "ASD": int((out.group == "ASD").sum()),
                 "TC_train": int(((out.group == "TC") & (out.split == "train")).sum()),
                 "TC_test": int(((out.group == "TC") & (out.split == "test")).sum()),
                 "n_sites": int(out["site"].nunique()), "n_regions": int(regions.shape[1])})
    age_bins = pd.cut(out.loc[out.group == "TC", "age"], [5, 7, 9, 11, 13, 15, 17, 19, 21.01], right=False)
    flow["TC_per_2y_bin"] = {str(k): int(v) for k, v in age_bins.value_counts().sort_index().items()}
    flow["TC_sex"] = out.loc[out.group == "TC", "sex"].value_counts().to_dict()
    p = ROOT / f"data/interim/A_abide_norm_input_{TODAY}.csv"
    out.to_csv(p, index=False)
    flow.update({"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A01_prepare_norm_input.py",
                 "output": str(p.relative_to(ROOT)).replace("\\", "/"), "regions": list(regions.columns)})
    (ROOT / "results/A01_norm_input_summary.json").write_text(json.dumps(flow, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in flow.items() if k != "regions"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
