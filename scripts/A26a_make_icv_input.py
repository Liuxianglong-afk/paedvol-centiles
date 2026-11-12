"""Sensitivity analysis input: regional volumes relative to intracranial volume (FastSurfer mask volume).

Each region except G_IntracranialMask is divided by G_IntracranialMask and multiplied by 1000 (per mille);
the intracranial volume itself is kept in absolute units. The output is then modelled with A13_gamlss_joint.R.
Input:  data/interim/A_joint_norm_input_20261112.csv (participant level; not distributed)
Output: data/interim/A_joint_norm_input_icv_20261112.csv
Usage:  python -I scripts/A26a_make_icv_input.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "data/interim/A_joint_norm_input_20261112.csv", dtype={"sub_n": str}, low_memory=False)
regs = [c for c in m.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB") and c != "G_IntracranialMask"]
m[regs] = m[regs].div(m["G_IntracranialMask"], axis=0) * 1000
assert (m[regs] > 0).all().all()
m.to_csv(ROOT / "data/interim/A_joint_norm_input_icv_20261112.csv", index=False)
print(len(regs), "regions normalised")
