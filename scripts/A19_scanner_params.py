"""T1 acquisition parameters per site from BIDS JSON sidecars on the public FCP-INDI bucket.

Up to five analysed participants per site (seed 42); the mode per site is recorded with a consistency flag.
Sites without sidecars (all ABIDE sites) were documented from the ABIDE site documentation instead.
Outputs: results/A19_scanner_params.json; tables_A/A19_scanner_params.csv (no participant identifiers)
Usage: python -I scripts/A19_scanner_params.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
FIELDS = ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "SliceThickness"]
rng = np.random.default_rng(42)

meta = pd.read_csv(ROOT / "data/interim/A_joint_norm_input_20261112.csv", dtype={"sub_n": str}, usecols=["db", "cohort", "site", "sub_n"])
hm = pd.read_csv(ROOT / "data/interim/hbn_t1_manifest_20261014.csv", dtype=str)
am = pd.read_csv(ROOT / "data/interim/abide_t1_manifest_20261014.csv", dtype=str)


def keys_for(db, site, subs):
    if db == "HBN":
        m = hm[(hm.site == site.replace("HBN-", "Site-")) & hm["sub"].str.replace("sub-", "", regex=False).str.upper().isin([s.upper() for s in subs])]
        return list(m["key"])
    m = am[(am.site == site) & am["sub"].str.replace("sub-", "", regex=False).isin(subs)]
    return list(m["key"])


def fetch(key):
    for k in (key.replace(".nii.gz", ".json"), key.rsplit("/", 1)[0] + "/../../T1w.json"):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{k}", timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
    return None


rows, rep = [], {}
for (db, site), g in meta.groupby(["db", "site"]):
    subs = list(g.sub_n.drop_duplicates())
    pick = [subs[i] for i in rng.permutation(len(subs))[:5]]
    js = [j for j in (fetch(k) for k in keys_for(db, site, pick)) if j]
    rec = {"db": db, "site": site, "n_analysed": int(len(g)), "n_json": len(js)}
    for fld in FIELDS:
        vals = [j.get(fld) for j in js if j.get(fld) is not None]
        if vals:
            s = pd.Series([str(v) for v in vals])
            rec[fld] = s.mode().iloc[0]; rec[fld + "_consistent"] = bool(s.nunique() == 1)
        else:
            rec[fld] = None
    rows.append(rec)
df = pd.DataFrame(rows)
(ROOT / "tables_A").mkdir(exist_ok=True)
df.to_csv(ROOT / "tables_A/A19_scanner_params.csv", index=False)
rep = {"script": "scripts/A19_scanner_params.py", "n_sites": int(len(df)), "sites_with_json": int((df.n_json > 0).sum()),
       "sites_without_json": df.loc[df.n_json == 0, "site"].tolist(),
       "manufacturer_counts": df.Manufacturer.value_counts(dropna=False).to_dict(),
       "field_strength_counts": df.MagneticFieldStrength.value_counts(dropna=False).to_dict()}
(ROOT / "results/A19_scanner_params.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
print(df[["db", "site", "n_json", "Manufacturer", "ManufacturersModelName", "MagneticFieldStrength"]].to_string())
