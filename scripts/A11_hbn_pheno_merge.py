"""Merge HBN FastSurfer volumes with R11 phenotype (demographics + clinician consensus Dx).

Outputs
  data/interim/hbn_A_pheno_vol_R11_20261112.csv  (per-subject, LOCAL ONLY: contains EID)
  results/A11_hbn_pheno_merge.json               (aggregate counts only, no IDs)
Run: .venv_B/Scripts/python -I scripts/A11_hbn_pheno_merge.py
"""
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PH = ROOT / "data/raw/hbn_pheno_R11_20261112/extracted/HBN_phenotypic_R11/HBN_phenotypic_R11"
VOL = ROOT / "derivatives/features/hbn_fastsurfer_volumes_20261018.csv"
OUT = ROOT / "data/interim/hbn_A_pheno_vol_R11_20261112.csv"
REP = ROOT / "results/A11_hbn_pheno_merge.json"

def rd(name):  # EID upper-cased below
    return pd.read_csv(PH / f"{name}.csv", encoding="utf-8-sig", low_memory=False)

vol = pd.read_csv(VOL)
vol["EID"] = vol["sub_n"].astype(str).str.strip().str.upper()  # sub_n == HBN EID
demo = rd("Basic_Demos")[["EID", "Sex", "Age", "Study_Site", "Release_Number"]]
dx = rd("Diagnosis_ClinicianConsensus"); dx["EID"] = dx["EID"].str.strip().str.upper(); demo["EID"] = demo["EID"].str.strip().str.upper()
dxcols = [f"DX_{i:02d}" for i in range(1, 11)]
catcols = [f"DX_{i:02d}_Cat" for i in range(1, 11)]

rep = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "script": "scripts/A11_hbn_pheno_merge.py",
       "n_vol": int(len(vol)), "n_demo": int(len(demo)), "n_dx_rows": int(len(dx)),
       "n_dx_unique_eid": int(dx["EID"].nunique()),
       "demo_dup_eid": int(demo["EID"].duplicated().sum())}

dx = dx.drop_duplicates("EID", keep="first")
m = vol.merge(demo.drop_duplicates("EID"), on="EID", how="left").merge(
    dx[["EID", "NoDX"] + dxcols + catcols], on="EID", how="left")
rep["n_vol_with_demo"] = int(m["Age"].notna().sum())
rep["n_vol_with_dx"] = int(m["NoDX"].notna().sum())
rep["NoDX_values"] = {str(k): int(v) for k, v in m["NoDX"].value_counts(dropna=False).items()}

# diagnosis labels
def has(row, pat):
    return any(isinstance(row[c], str) and pat.lower() in row[c].lower() for c in dxcols)

def first_cat(row):
    return row["DX_01_Cat"] if isinstance(row["DX_01_Cat"], str) else np.nan

m["dx_none"] = m["DX_01"].astype(str).str.strip().eq("No Diagnosis Given")
m["dx_asd"] = m.apply(lambda r: has(r, "Autism Spectrum"), axis=1)
m["dx_adhd"] = m.apply(lambda r: has(r, "ADHD"), axis=1)
m["dx_anx"] = m.apply(lambda r: any(isinstance(r[c], str) and r[c].startswith("Anxiety") for c in catcols), axis=1)
m["dx_ld"] = m.apply(lambda r: has(r, "Specific Learning Disorder"), axis=1)
m["dx_dep"] = m.apply(lambda r: any(isinstance(r[c], str) and r[c].startswith("Depressive") for c in catcols), axis=1)
m["dx_first_cat"] = m.apply(first_cat, axis=1)

complete = m.drop(columns=[c for c in m.columns if c.startswith(("DX_",))]).filter(regex="^v_").notna().all(axis=1)
m["vol_complete"] = complete
q = m[m["vol_complete"] & m["Age"].notna() & m["NoDX"].notna()]
rep["n_analysable"] = int(len(q))
rep["groups_analysable"] = {g: int(q[g].sum()) for g in ["dx_none", "dx_asd", "dx_adhd", "dx_anx", "dx_ld", "dx_dep"]}
rep["dx_first_cat_top"] = {str(k): int(v) for k, v in q["dx_first_cat"].value_counts().head(15).items()}
rep["DX_01_raw_top"] = {str(k): int(v) for k, v in q["DX_01"].value_counts().head(25).items()}
ctrl = q[q["dx_none"]]
bins = list(range(5, 23, 2))
rep["ctrl_age_bins_2y"] = {f"{a}-{a+2}": int(((ctrl.Age >= a) & (ctrl.Age < a + 2)).sum()) for a in bins}
rep["ctrl_sex"] = {str(k): int(v) for k, v in ctrl["Sex"].value_counts().items()}
rep["ctrl_site"] = {str(k): int(v) for k, v in ctrl["site"].value_counts().items()}
rep["all_site"] = {str(k): int(v) for k, v in q["site"].value_counts().items()}
rep["age_range_all"] = [round(float(q.Age.min()), 2), round(float(q.Age.max()), 2)]
rep["note_sex_coding"] = "Basic_Demos Sex: HBN codebook 0=male,1=female (verify)"

OUT.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT, index=False)
REP.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(rep, ensure_ascii=False, indent=1))
