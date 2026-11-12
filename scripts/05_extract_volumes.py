"""Collect FastSurfer segmentation-only statistics into a wide table (one row per participant).

Sources:
  stats/aseg+DKT.VINN.stats         partial-volume-corrected regional volumes (Volume_mm3) and global measures ('# Measure' lines)
  stats/cerebellum.CerebNet.stats   cerebellar lobule volumes
Columns: g_<global measure>, v_<structure>, cb_<cerebellar structure> ('-' in names replaced by '_').
Outputs:
  derivatives/features/<cohort>_fastsurfer_volumes_<YYYYMMDD>.csv
  results/<cohort>_volumes_summary.json
Usage: python -I scripts/05_extract_volumes.py --cohort abide
"""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TODAY = dt.date.today().strftime("%Y%m%d")


def parse_stats(path, prefix):
    """FreeSurfer-format stats: '# Measure' lines and the table after '# ColHeaders'."""
    out, header = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            if len(parts) >= 4:
                out[f"g_{parts[1]}"] = float(parts[3])
        elif line.startswith("# ColHeaders"):
            header = line.split()[2:]
        elif header and line.strip() and not line.startswith("#"):
            row = dict(zip(header, line.split()))
            name = re.sub(r"[^0-9A-Za-z]+", "_", row["StructName"]).strip("_")
            out[f"{prefix}_{name}"] = float(row["Volume_mm3"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="abide")
    args = ap.parse_args()
    sd = ROOT / "derivatives/fastsurfer" / args.cohort
    rows, incomplete = [], []
    for subj in sorted(p for p in sd.iterdir() if p.is_dir()):
        a, c = subj / "stats/aseg+DKT.VINN.stats", subj / "stats/cerebellum.CerebNet.stats"
        if not (a.exists() and c.exists()):
            incomplete.append(subj.name)
            continue
        rec = {"sid": subj.name}
        rec.update(parse_stats(a, "v"))
        cb = parse_stats(c, "cb")
        rec.update({k: v for k, v in cb.items() if k.startswith("cb_")})
        rows.append(rec)
    df = pd.DataFrame(rows)

    parts = df["sid"].str.extract(r"^(?P<cohort>abide[12]|hbn|corr)_(?P<site>.+)_sub-(?P<sub_n>[^_]+)$")
    df = pd.concat([parts, df], axis=1)
    feat = ROOT / "derivatives/features"
    feat.mkdir(parents=True, exist_ok=True)
    out = feat / f"{args.cohort}_fastsurfer_volumes_{TODAY}.csv"
    df.to_csv(out, index=False)

    vcols = [c for c in df.columns if c.startswith("v_")]
    cbcols = [c for c in df.columns if c.startswith("cb_")]
    gcols = [c for c in df.columns if c.startswith("g_")]
    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/05_extract_volumes.py",
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
        "n_subjects": int(len(df)),
        "n_incomplete_dirs": len(incomplete),
        "n_region_volumes": len(vcols),
        "n_cerebellum_volumes": len(cbcols),
        "global_measures": gcols,
        "subjects_with_any_missing_volume": int(df[vcols + cbcols].isna().any(axis=1).sum()),
        "g_MaskVol_mean_cm3": round(float(df["g_MaskVol"].mean()) / 1000, 1) if "g_MaskVol" in df else None,
        "g_BrainSegVol_mean_cm3": round(float(df["g_BrainSegVol"].mean()) / 1000, 1) if "g_BrainSegVol" in df else None,
    }
    (ROOT / "results" / f"{args.cohort}_volumes_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
