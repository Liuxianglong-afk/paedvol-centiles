"""Prepare the joint ABIDE + HBN normative-model input.

HBN quality control as in A01: incomplete cerebellar coverage excluded; any non-ventricular region <25% or >300% of the
sex-specific median (computed within HBN) excluded.
HBN groups (clinician consensus diagnoses, any listed diagnosis):
  TC = DX_01 "No Diagnosis Given" ("Incomplete Eval" excluded);
  ASD / ADHD / anxiety disorders / specific learning disorder / depressive disorders as non-exclusive flags.
HBN controls split within site 50/50 into model-building (train) and held-out (test) halves, seed 42; ABIDE keeps the A01 split.
Sex: HBN Basic_Demos 0 = male, 1 = female (checked against the data: larger intracranial volume and 83% of ASD coded 0).
Outputs: data/interim/A_joint_norm_input_20261112.csv (participant level; not distributed); results/A12_joint_input_summary.json
Usage: python -I scripts/A12_prepare_joint_input.py
"""
import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("a01", HERE / "A01_prepare_norm_input.py")
a01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a01)
SEED = 42
FLAGS = ["dx_asd", "dx_adhd", "dx_anx", "dx_ld", "dx_dep"]

h = pd.read_csv(ROOT / "data/interim/hbn_A_pheno_vol_R11_20261112.csv", low_memory=False)
flow = {"hbn_volumes": int(len(h))}
h = h[h["Age"].notna() & h["NoDX"].notna()].copy()
flow["with_age_and_dx"] = int(len(h))
inc = h["DX_01"].astype(str).str.contains("Incomplete Eval")
flow["excluded_incomplete_eval"] = int(inc.sum())
h = h[~inc & h["Age"].between(5, 21)].copy()
flow["eligible"] = int(len(h))
cb = [c for c in h.columns if c.startswith("cb_")]
fov_bad = h[cb].isna().any(axis=1) | (h[[c for c in cb if "Cortex" in c]].fillna(0) <= 0).any(axis=1)
flow["excluded_incomplete_fov"] = int(fov_bad.sum())
h = h[~fov_bad].copy()
reg = a01.build_regions(h)
nonvent = [c for c in reg.columns if "Ventric" not in c]
med = reg[nonvent].groupby(h["Sex"].to_numpy()).transform("median")
ratio = reg[nonvent] / med.to_numpy()
gross = ((ratio < 0.25) | (ratio > 3.0)).any(axis=1).to_numpy()
flow["excluded_gross_volume_outlier"] = int(gross.sum())
h, reg = h[~gross].copy(), reg[~gross]

hb = pd.DataFrame({"db": "HBN", "cohort": "HBN", "site": "HBN-" + h["site"].str.replace("Site-", "", regex=False),
                   "sub_n": h["EID"].to_numpy(), "age": h["Age"].to_numpy(),
                   "sex": np.where(h["Sex"].to_numpy() == 0, "M", "F"), "fiq": np.nan})
hb["group"] = np.where(h["dx_none"].to_numpy(), "TC", "DX")
hb["dx_primary_cat"] = h["dx_first_cat"].to_numpy()
for f in FLAGS:
    hb[f] = h[f].astype(bool).to_numpy().astype(int)
hb = pd.concat([hb.reset_index(drop=True), reg.reset_index(drop=True)], axis=1)
rng = np.random.default_rng(SEED)
hb["split"] = "test"
for s, idx in hb[hb.group == "TC"].groupby("site").groups.items():
    idx = np.array(sorted(idx)); rng.shuffle(idx)
    hb.loc[idx[: int(np.ceil(len(idx) / 2))], "split"] = "train"

ab = pd.read_csv(ROOT / "data/interim/A_abide_norm_input_20261018.csv", dtype={"sub_n": str})
ab.insert(0, "db", "ABIDE")
ab["dx_primary_cat"] = np.where(ab.group == "ASD", "ASD", "TC")
for f in FLAGS:
    ab[f] = 0
ab["dx_asd"] = (ab.group == "ASD").astype(int)
cols = list(hb.columns)
out = pd.concat([ab[cols], hb[cols]], ignore_index=True)

tc = hb[hb.group == "TC"]
flow.update({
    "HBN_analysed": int(len(hb)), "HBN_TC": int(len(tc)),
    "HBN_TC_train": int((tc.split == "train").sum()), "HBN_TC_test": int((tc.split == "test").sum()),
    "HBN_TC_by_site": tc.site.value_counts().to_dict(),
    "HBN_TC_sex": tc.sex.value_counts().to_dict(),
    "HBN_TC_per_2y_bin": {str(k): int(v) for k, v in pd.cut(tc.age, [5, 7, 9, 11, 13, 15, 17, 19, 21.01], right=False).value_counts().sort_index().items()},
    "HBN_flags": {f: int(hb[f].sum()) for f in FLAGS},
    "HBN_any_dx_none_of_flags": int(((hb.group == "DX") & (hb[FLAGS].sum(axis=1) == 0)).sum()),
    "ABIDE_rows": int(len(ab)), "joint_TC": int((out.group == "TC").sum()),
    "joint_TC_per_2y_bin": {str(k): int(v) for k, v in pd.cut(out.loc[out.group == "TC", "age"], [5, 7, 9, 11, 13, 15, 17, 19, 21.01], right=False).value_counts().sort_index().items()},
    "n_sites_joint": int(out.site.nunique()),
})
p = ROOT / "data/interim/A_joint_norm_input_20261112.csv"
out.to_csv(p, index=False)
flow.update({"generated": dt.datetime.now().isoformat(timespec="seconds"), "script": "scripts/A12_prepare_joint_input.py",
             "output": str(p.relative_to(ROOT)).replace("\\", "/")})
(ROOT / "results/A12_joint_input_summary.json").write_text(json.dumps(flow, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(flow, indent=1, ensure_ascii=False))
