"""Summary of the visual quality control: 50 randomly selected analysed scans inspected, no segmentation errors.

Upper 95% confidence limit of the error rate by the Clopper-Pearson method.
Output: results/A27_visual_qc.json
Usage: python -I scripts/A27_visual_qc_summary.py
"""
import json
from pathlib import Path
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
n, x = 50, 0
upper = float(beta.ppf(0.975, x + 1, n - x))
res = {"script": "scripts/A27_visual_qc_summary.py", "source": "user report 2026-11-12: 50 randomly selected analysed scans visually inspected; no segmentation errors",
       "n_inspected": n, "n_errors": x, "error_rate_95ci_clopper_pearson": [0.0, round(upper, 4)]}
(ROOT / "results/A27_visual_qc.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(res)
