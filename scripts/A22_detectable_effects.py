"""Smallest detectable standardised differences for the main comparisons (two-sided alpha .05, 80% power).

d_min = (z_{1-alpha/2} + z_power) * sqrt(1/n1 + 1/n2); regional level with Bonferroni alpha = .05/92.
Output: results/A22_detectable_effects.json
Usage: python -I scripts/A22_detectable_effects.py
"""
import json
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
A14 = json.load(open(ROOT / "results/A14_hbn_results.json", encoding="utf-8"))
A15 = json.load(open(ROOT / "results/A15_interpretation.json", encoding="utf-8"))
G, AA, CC = A14["hbn_dx"]["groups"], A15["abide_asd"], A15["hbn_asd_vs_adhd_noASD"]
n_hc = A14["hbn_dx"]["n_TC_test"]


def dmin(n1, n2, alpha=0.05, power=0.80):
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * (1 / n1 + 1 / n2) ** 0.5


comps = {"ABIDE ASD vs controls": (AA["n_ASD"], AA["n_TC_test"]),
         "HBN ASD vs controls": (G["ASD"]["n"], n_hc), "HBN ADHD vs controls": (G["ADHD"]["n"], n_hc),
         "HBN anxiety vs controls": (G["ANX"]["n"], n_hc), "HBN SLD vs controls": (G["SLD"]["n"], n_hc),
         "HBN depression vs controls": (G["DEP"]["n"], n_hc), "HBN ASD vs ADHD without ASD": (CC["n_ASD"], CC["n_ADHD_noASD"])}
out = {"script": "scripts/A22_detectable_effects.py", "alpha": 0.05, "power": 0.80, "regional_alpha_bonferroni": 0.05 / 92,
       "comparisons": {k: {"n1": n1, "n2": n2, "d_min": round(dmin(n1, n2), 3), "d_min_regional_bonferroni": round(dmin(n1, n2, 0.05 / 92), 3)}
                       for k, (n1, n2) in comps.items()}}
(ROOT / "results/A22_detectable_effects.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
