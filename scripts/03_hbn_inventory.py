"""HBN inventory: list T1-weighted files at the four sites on FCP-INDI and download the public participants.tsv.

Clinician consensus diagnoses (DSM-5) require a data use agreement with the Child Mind Institute; this script uses public fields only
(age, sex, handedness, release).
Outputs:
  data/raw/hbn/<site>/participants.tsv
  data/interim/hbn_t1_manifest_<YYYYMMDD>.csv   one row per T1 file (including acquisition variants)
  data/interim/hbn_public_pheno_<YYYYMMDD>.csv  one row per participant (with preferred T1)
  results/hbn_inventory.json
Usage: python -I scripts/03_hbn_inventory.py
"""
import datetime as dt
import io
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://fcp-indi.s3.us-east-1.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
PREFIX = "data/Projects/HBN/MRI/"
SITES = ["Site-CBIC", "Site-CUNY", "Site-RU", "Site-SI"]
AGE_MIN, AGE_MAX = 5.0, 21.0

ACQ_PRIORITY = {"HCP": 0, "VNav": 1, "VNavNorm": 2}
TODAY = dt.date.today().strftime("%Y%m%d")


def http_get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {e}")
            time.sleep(2 ** i)


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(http_get(f"{BUCKET}/?{urllib.parse.urlencode(q)}"))
        for c in root.findall("s3:Contents", NS):
            keys.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


T1_RE = re.compile(r"^sub-(?P<sub>[^/]+)/anat/sub-[^/]+?(?:_acq-(?P<acq>[^_]+))?(?:_run-(?P<run>\d+))?_T1w\.nii\.gz$")


def list_subject_dirs(prefix):
    """List first-level subject folders only (delimiter='/') to avoid paging through derivatives."""
    dirs, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(http_get(f"{BUCKET}/?{urllib.parse.urlencode(q)}"))
        dirs += [p.findtext("s3:Prefix", namespaces=NS) for p in root.findall("s3:CommonPrefixes", NS)]
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return [d for d in dirs if d[len(prefix):].startswith("sub-")]
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def main():
    from concurrent.futures import ThreadPoolExecutor
    rows, phenos = [], []
    for site in SITES:
        prefix = f"{PREFIX}{site}/"
        subdirs = list_subject_dirs(prefix)
        print(f"{site}: {len(subdirs)} subject dirs, listing anat/ ...", flush=True)
        with ThreadPoolExecutor(16) as ex:
            for keys in ex.map(lambda d: list_keys(d + "anat/"), subdirs):
                for key, size in keys:
                    m = T1_RE.match(key[len(prefix):])
                    if m:
                        rows.append({"site": site, "sub": m["sub"], "acq": m["acq"] or "none",
                                     "run": m["run"] or "1", "key": key, "size_bytes": size})
        print(f"  T1 files so far: {len(rows)}", flush=True)
        out = ROOT / "data/raw/hbn" / site / "participants.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_bytes(http_get(f"{BUCKET}/{prefix}participants.tsv"))
        text = out.read_bytes().decode("utf-8", errors="replace")

        sep = "\t" if "\t" in text.splitlines()[0] else ","
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, na_values=["n/a", ""])
        df["site"] = site
        phenos.append(df)

    manifest = pd.DataFrame(rows)
    interim = ROOT / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(interim / f"hbn_t1_manifest_{TODAY}.csv", index=False)

    ph = pd.concat(phenos, ignore_index=True)
    ph["sub"] = ph["participant_id"].str.replace("sub-", "", regex=False).str.strip()
    ph["age"] = pd.to_numeric(ph["Age"], errors="coerce")
    ph["sex"] = pd.to_numeric(ph["Sex"], errors="coerce")

    manifest["acq_rank"] = manifest["acq"].map(ACQ_PRIORITY).fillna(9)
    first = (manifest.sort_values(["acq_rank", "run"])
             .drop_duplicates(["site", "sub"])
             [["site", "sub", "acq", "key", "size_bytes"]]
             .rename(columns={"acq": "t1_acq", "key": "t1_key", "size_bytes": "t1_bytes"}))
    ph = ph.merge(first, on=["site", "sub"], how="left")
    ph["has_t1"] = ph["t1_key"].notna()
    ph["in_age_5_21"] = ph["age"].between(AGE_MIN, AGE_MAX)
    ph.to_csv(interim / f"hbn_public_pheno_{TODAY}.csv", index=False)


    dup_subs = int(ph[ph["has_t1"]].duplicated("sub", keep=False).sum())
    orphan = manifest.merge(ph[["site", "sub"]], how="left", indicator=True, on=["site", "sub"])
    elig = ph[ph["has_t1"] & ph["in_age_5_21"]]
    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/03_hbn_inventory.py",
        "n_t1_files": int(len(manifest)),
        "t1_acq_counts": manifest["acq"].value_counts().to_dict(),
        "n_participants_rows": int(len(ph)),
        "n_t1_subjects_without_participants_row": int(
            (orphan["_merge"] == "left_only").sum()),
        "subjects_with_t1_in_multiple_sites_rows": dup_subs,
        "by_site": {s: {"rows": int(len(g)), "with_t1": int(g["has_t1"].sum()),
                        "eligible_5_21": int((g["has_t1"] & g["in_age_5_21"]).sum())}
                    for s, g in ph.groupby("site")},
        "eligible_5_21_with_t1": {
            "total": int(len(elig)),
            "unique_subjects": int(elig["sub"].nunique()),
            "sex_counts_raw": {str(k): int(v) for k, v in elig["sex"].value_counts().items()},
            "age_mean": round(float(elig["age"].mean()), 2),
            "age_sd": round(float(elig["age"].std()), 2),
            "t1_download_GB": round(float(elig["t1_bytes"].sum()) / 1e9, 2),
            "first_t1_acq_counts": elig["t1_acq"].value_counts().to_dict(),
        },
        "note": "Sex coding in the public HBN participants.tsv should be checked against the HBN data dictionary; diagnoses require the data use agreement.",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/hbn_inventory.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
