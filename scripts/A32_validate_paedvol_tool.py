"""Validate the paedvol tool against exact GAMLSS z scores (validation sample from A31_zgrid_centiles.R).

Compares tool z (dense z-grid interpolation) with exact model z for 400 random participants x 92 regions;
reports absolute error and agreement of the |z| > 1.96 classification (values beyond |z| = 4 are truncated by the tool).
Also checks the calibration function: re-centring with all local controls must give a site mean of zero.
Output: results/A31_tool_validation.json
Usage:  python -I scripts/A32_validate_paedvol_tool.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools_A"))
import paedvol  # noqa: E402

ref = paedvol.Reference(ROOT / "derivatives/norms_joint/A31_zgrid.csv.gz")
v = pd.read_csv(ROOT / "derivatives/norms_joint/A31_validation_exact_z.csv.gz", dtype={"sub_n": str})
key = ["site", "sub_n", "age", "sex"]
wide = v.pivot_table(index=key, columns="region", values="volume").reset_index()
ex = v.pivot_table(index=key, columns="region", values="z_exact").reset_index()
regs = [c for c in wide.columns if c[:2] in ("G_", "S_", "V_", "C_", "CB")]
z = ref.z(wide[regs], wide.age, wide.sex)
E, Z = ex[z.columns].to_numpy(float), z.to_numpy(float)
inside = np.abs(E) < 4
d = np.abs(Z - E)[inside]
res = {"script": "scripts/A32_validate_paedvol_tool.py", "n_participants": int(len(wide)), "n_values": int(E.size),
       "n_within_grid": int(inside.sum()), "abs_err_median": float(np.median(d)), "abs_err_p99": float(np.percentile(d, 99)),
       "abs_err_max": float(d.max()), "out_of_range_classification_agreement": float(((np.abs(E) > 1.96) == (np.abs(Z) > 1.96)).mean()),
       "n_truncated_at_4": int((~inside).sum())}
cal = paedvol.calibrate(z, wide.site, np.ones(len(z), bool), method="recentre", eb_path=ROOT / "derivatives/norms_joint/A24_eb_parameters_by_region.csv")
res["recentre_all_controls_max_abs_site_mean"] = float(cal.groupby(wide.site.to_numpy()).mean().abs().max().max())
(ROOT / "results/A31_tool_validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
