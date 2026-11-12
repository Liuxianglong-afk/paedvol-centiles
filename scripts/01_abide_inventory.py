"""ABIDE I/II inventory: list T1-weighted files on FCP-INDI, download each site's participants.tsv and harmonise phenotypes.

Downloads phenotype tables (kilobytes) and file listings only, no images.
Outputs:
  data/raw/abide/{abide1,abide2}/<site>/participants.tsv
  data/interim/abide_t1_manifest_<YYYYMMDD>.csv       one row per T1 file
  data/interim/abide_pheno_harmonized_<YYYYMMDD>.csv  one row per participant (with T1 availability)
  results/abide_inventory.json                         summary
Usage: python -I scripts/01_abide_inventory.py
"""
import csv
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
PREFIXES = {
    "abide1": "data/Projects/ABIDE/RawDataBIDS/",
    "abide2": "data/Projects/ABIDE2/RawData/",
}
AGE_MIN, AGE_MAX = 5.0, 21.0
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
    """List all objects (key, size) under a prefix, page by page."""
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


T1_RE = re.compile(r"^(?P<site>[^/]+)/sub-(?P<sub>[^/_]+)/(?:ses-(?P<ses>[^/]+)/)?anat/[^/]*T1w\.nii\.gz$")


def build_manifest():
    rows, pheno_keys = [], []
    for cohort, prefix in PREFIXES.items():
        print(f"listing {cohort} ...")
        keys = list_keys(prefix)
        print(f"  {len(keys)} objects")
        for key, size in keys:
            rel = key[len(prefix):]
            if rel.endswith("participants.tsv") and rel.count("/") == 1:
                pheno_keys.append((cohort, rel.split("/")[0], key))
                continue
            m = T1_RE.match(rel)
            if m:
                run = re.search(r"run-(\d+)", rel)
                rows.append({
                    "cohort": cohort, "site": m["site"], "sub": m["sub"],
                    "ses": m["ses"] or "1", "run": run.group(1) if run else "1",
                    "key": key, "size_bytes": size,
                })
    return pd.DataFrame(rows), pheno_keys


def fetch_phenotypes(pheno_keys):
    frames = []
    for cohort, site, key in pheno_keys:
        out = ROOT / "data/raw/abide" / cohort / site / "participants.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_bytes(http_get(f"{BUCKET}/{key}"))
        raw = out.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
            print(f"  {cohort}/{site}: non-UTF-8 participants.tsv, decoded as cp1252")
        df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str,
                         na_values=["n/a", "-9999", "-9999.0", ""])
        df.columns = [c.strip().lower() for c in df.columns]
        df["cohort"], df["site"] = cohort, site
        frames.append(df)
    return frames



COLMAP = {
    "dx_group": ["dx_group"],
    "age": ["age_at_scan"],
    "sex": ["sex"],
    "fiq": ["fiq"], "viq": ["viq"], "piq": ["piq"],
    "handedness_category": ["handedness_category"],
    "ados_css": ["ados_gotham_severity", "ados_2_severity_total"],
    "ados_total": ["ados_total", "ados_g_total"],
    "adi_r_social_a": ["adi_r_social_total_a"],
    "srs_raw_total": ["srs_raw_total", "srs_total_raw"],
    "srs_total_t": ["srs_total_t"],
    "scq_total": ["scq_total"],
    "aq_total": ["aq_total"],
    "dsm_iv_tr": ["dsm_iv_tr", "pdd_dsm_iv_tr"],
    "asd_dsm_5": ["asd_dsm_5"],
    "comorbidity": ["comorbidity"],
    "current_med_status": ["current_med_status"],
}
NUMERIC = ["dx_group", "age", "sex", "fiq", "viq", "piq", "ados_css", "ados_total",
           "adi_r_social_a", "srs_raw_total", "srs_total_t", "scq_total", "aq_total",
           "current_med_status"]


def harmonize(frames):
    out = []
    for df in frames:
        h = pd.DataFrame({"cohort": df["cohort"], "site": df["site"],
                          "sub": df["participant_id"].str.strip()})
        for std, cands in COLMAP.items():
            col = next((c for c in cands if c in df.columns), None)
            h[std] = df[col] if col else pd.NA
        out.append(h)
    ph = pd.concat(out, ignore_index=True)
    for c in NUMERIC:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")
    return ph


def norm_sub(s):
    """ABIDE I folders use zero-padded IDs (sub-00NNNNN) while phenotypes use NNNNN: strip leading zeros."""
    return str(s).lstrip("0") or "0"


def main():
    manifest, pheno_keys = build_manifest()
    interim = ROOT / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(interim / f"abide_t1_manifest_{TODAY}.csv", index=False)

    ph = harmonize(fetch_phenotypes(pheno_keys))
    manifest["sub_n"] = manifest["sub"].map(norm_sub)
    ph["sub_n"] = ph["sub"].map(norm_sub)


    first_t1 = (manifest.sort_values(["ses", "run"])
                .drop_duplicates(["cohort", "site", "sub_n"])
                [["cohort", "site", "sub_n", "key", "size_bytes"]]
                .rename(columns={"key": "t1_key", "size_bytes": "t1_bytes"}))
    n_t1_files = manifest.groupby(["cohort", "site", "sub_n"]).size().rename("n_t1_files").reset_index()
    ph = (ph.merge(first_t1, on=["cohort", "site", "sub_n"], how="left")
            .merge(n_t1_files, on=["cohort", "site", "sub_n"], how="left"))
    ph["has_t1"] = ph["t1_key"].notna()
    ph["in_age_5_21"] = ph["age"].between(AGE_MIN, AGE_MAX)
    ph.to_csv(interim / f"abide_pheno_harmonized_{TODAY}.csv", index=False)


    orphan = manifest.merge(ph[["cohort", "site", "sub_n"]], how="left", indicator=True,
                            on=["cohort", "site", "sub_n"])
    orphan = orphan[orphan["_merge"] == "left_only"]
    orphan_n = int(len(orphan))


    orphan_sites = {f"{c}/{s}": int(n) for (c, s), n in
                    orphan.drop_duplicates(["cohort", "site", "sub_n"]).groupby(["cohort", "site"]).size().items()}

    elig = ph[ph["has_t1"] & ph["in_age_5_21"] & ph["dx_group"].isin([1, 2])]
    by_site = (elig.groupby(["cohort", "site", "dx_group"]).size().unstack(fill_value=0)
               .rename(columns={1: "ASD", 2: "TC"}))
    miss_cols = ["fiq", "ados_css", "srs_raw_total", "srs_total_t", "scq_total", "aq_total",
                 "current_med_status"]
    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/01_abide_inventory.py",
        "n_t1_files": int(len(manifest)),
        "n_pheno_rows": int(len(ph)),
        "n_t1_without_pheno_match": orphan_n,
        "subjects_without_pheno_by_site": orphan_sites,
        "by_cohort_all": {k: {"rows": int(len(g)), "with_t1": int(g["has_t1"].sum())}
                          for k, g in ph.groupby("cohort")},
        "eligible_5_21_with_t1": {
            "total": int(len(elig)),
            "ASD": int((elig["dx_group"] == 1).sum()),
            "TC": int((elig["dx_group"] == 2).sum()),
            "male": int((elig["sex"] == 1).sum()),
            "female": int((elig["sex"] == 2).sum()),
            "age_mean": round(float(elig["age"].mean()), 2),
            "age_sd": round(float(elig["age"].std()), 2),
            "t1_download_GB": round(float(elig["t1_bytes"].sum()) / 1e9, 2),
        },
        "eligible_by_site": {f"{c}/{s}": {"ASD": int(r.get("ASD", 0)), "TC": int(r.get("TC", 0))}
                             for (c, s), r in by_site.iterrows()},
        "eligible_missing_pct": {c: round(float(elig[c].isna().mean()) * 100, 1) for c in miss_cols},
        "eligible_missing_pct_ASD_only": {
            c: round(float(elig.loc[elig["dx_group"] == 1, c].isna().mean()) * 100, 1)
            for c in miss_cols},
    }
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "abide_inventory.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "eligible_by_site"},
                     indent=2, ensure_ascii=False))
    print(f"sites with eligible subjects: {len(by_site)}")


if __name__ == "__main__":
    main()
