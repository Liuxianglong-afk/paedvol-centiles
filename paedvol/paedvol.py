"""paedvol: paediatric brain volume centiles (5-21 years) for FastSurfer, with local site calibration.

Implements the workflow recommended in the accompanying article:
  1. read FastSurfer 2.6 segmentation-only outputs (aseg+DKT.VINN.stats, cerebellum.CerebNet.stats) or a table of volumes;
  2. convert the 92 regional volumes to z scores with the sex-specific reference centiles (ABIDE + HBN controls);
  3. estimate each site's offset from its local typically developing controls and remove it by empirical Bayes
     shrinkage (B = k*tau2 / (k*tau2 + sigma2)) or by plain re-centring;
  4. count regions below the 2.5th and above the 97.5th centile separately.

Only numpy and pandas are required. Data files (in ../data):
  zgrid.csv.gz         region x age (5-21 y, step 0.25) x sex: volume (cm3) at z = -4.0 ... 4.0 (step 0.1)
  eb_parameters.csv    per-region between-site variance (tau2) and within-site variance (sigma2)

Command line:
  python paedvol.py --input participants.csv --output results.csv
  participants.csv needs columns: id, site, age (years), sex (F/M), is_control (1 = typically developing local control, else 0),
  and either the 92 region columns (cm3, names as in data/eb_parameters.csv) or a column fastsurfer_dir pointing to each FastSurfer subject directory.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SUBCORT = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens_area", "VentralDC"]
Z_LOW, Z_HIGH = -1.96, 1.96


# ---------------------------------------------------------------- reading FastSurfer outputs
def parse_stats(path, prefix):
    """FreeSurfer-format stats: '# Measure' lines and the table after '# ColHeaders'."""
    out, header = {}, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
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


def read_fastsurfer(subject_dir):
    """Raw FastSurfer measures of one subject (mm3)."""
    d = Path(subject_dir)
    rec = parse_stats(d / "stats/aseg+DKT.VINN.stats", "v")
    rec.update({k: v for k, v in parse_stats(d / "stats/cerebellum.CerebNet.stats", "cb").items() if k.startswith("cb_")})
    return rec


def build_regions(raw):
    """92 analysis regions (cm3) from raw FastSurfer measures (DataFrame, one row per subject)."""
    f = raw
    ctx = [c for c in f.columns if c.startswith("v_ctx_")]
    r = pd.DataFrame(index=f.index)
    r["G_IntracranialMask"] = f["g_MaskVol"]
    r["G_BrainNotVent"] = f["g_BrainSegVolNotVent"]
    r["G_CorticalGM"] = f[ctx].sum(axis=1)
    r["G_CerebralWM"] = f["g_CerebralWhiteMatterVol"]
    r["G_SubcorticalGM"] = f["g_SubCortGrayVol"]
    r["G_LateralVentricles"] = f["v_Left_Lateral_Ventricle"] + f["v_Right_Lateral_Ventricle"]
    r["G_Cerebellum"] = f[["v_Left_Cerebellum_Cortex", "v_Right_Cerebellum_Cortex",
                           "v_Left_Cerebellum_White_Matter", "v_Right_Cerebellum_White_Matter"]].sum(axis=1)
    for s in SUBCORT:
        for h in ("Left", "Right"):
            r[f"S_{h}_{s}"] = f[f"v_{h}_{s}"]
    r["V_3rd_Ventricle"] = f["v_3rd_Ventricle"]
    r["V_4th_Ventricle"] = f["v_4th_Ventricle"]
    r["S_Brain_Stem"] = f["v_Brain_Stem"]
    for c in ctx:
        r["C_" + c[len("v_ctx_"):]] = f[c]
    for h in ("Left", "Right"):
        r[f"CB_{h}_Cortex"] = f[f"v_{h}_Cerebellum_Cortex"]
        r[f"CB_{h}_WM"] = f[f"v_{h}_Cerebellum_White_Matter"]
    return r / 1000.0


# ---------------------------------------------------------------- reference centiles
class Reference:
    """Sex-specific reference centiles on a dense z grid; z by linear interpolation in age and volume."""

    def __init__(self, zgrid_path=DATA / "zgrid.csv.gz"):
        g = pd.read_csv(zgrid_path)
        self.zcols = [c for c in g.columns if c.startswith("z_")]
        self.zvals = np.array([float(c[2:]) for c in self.zcols])
        self.ages = np.sort(g.age.unique())
        self.regions = list(dict.fromkeys(g.region))
        self.tab = {(r, s): sub.sort_values("age")[self.zcols].to_numpy() for (r, s), sub in g.groupby(["region", "sex"])}

    def z(self, volumes, age, sex):
        """volumes: DataFrame (rows = participants, columns = regions, cm3); age, sex: array-like. Returns z DataFrame."""
        age = np.asarray(age, float); sex = np.asarray(sex)
        if np.any((age < self.ages[0]) | (age > self.ages[-1])):
            raise ValueError("age outside the reference range 5-21 years")
        if not set(np.unique(sex)) <= {"F", "M"}:
            raise ValueError("sex must be coded F or M")
        i = np.clip(np.searchsorted(self.ages, age) - 1, 0, len(self.ages) - 2)
        w = (age - self.ages[i]) / (self.ages[i + 1] - self.ages[i])
        out = pd.DataFrame(index=volumes.index, columns=[r for r in self.regions if r in volumes.columns], dtype=float)
        for r in out.columns:
            y = volumes[r].to_numpy(float)
            for s in ("F", "M"):
                m = sex == s
                if not m.any():
                    continue
                T = self.tab[(r, s)]
                curves = T[i[m]] * (1 - w[m, None]) + T[i[m] + 1] * w[m, None]   # volume at each z for each participant's age
                # beyond the grid (|z| > 4) the value is truncated at -4 or 4, so it still counts as out of range
                out.loc[m, r] = [np.interp(v, c, self.zvals, left=self.zvals[0], right=self.zvals[-1]) for v, c in zip(y[m], curves)]
        n_trunc = int((out.abs() >= self.zvals[-1]).sum().sum())
        if n_trunc:
            print(f"note: {n_trunc} values beyond |z| = 4 truncated at +/-4 (please check segmentation)")
        return out


# ---------------------------------------------------------------- local calibration and counts
def calibrate(z, site, is_control, method="shrinkage", eb_path=DATA / "eb_parameters.csv"):
    """Remove site offsets estimated from local controls. method: 'shrinkage' (empirical Bayes) or 'recentre'."""
    eb = pd.read_csv(eb_path).set_index("region")
    site = pd.Series(np.asarray(site), index=z.index); ctrl = pd.Series(np.asarray(is_control).astype(bool), index=z.index)
    out = z.copy()
    for s in site.unique():
        rows = site == s; ref = z[rows & ctrl]
        k = len(ref)
        if k == 0:
            print(f"note: site {s} has no local controls; z scores left uncalibrated")
            continue
        mean = ref.mean()
        if method == "shrinkage":
            tau2, sigma2 = eb.loc[z.columns, "tau2"], eb.loc[z.columns, "sigma2"]
            mean = mean * (k * tau2 / (k * tau2 + sigma2))
        elif method != "recentre":
            raise ValueError("method must be 'shrinkage' or 'recentre'")
        out.loc[rows] = z[rows] - mean
    return out


def count_out_of_range(z):
    """Numbers of regions below the 2.5th and above the 97.5th centile."""
    below = (z < Z_LOW).sum(axis=1); above = (z > Z_HIGH).sum(axis=1)
    return pd.DataFrame({"n_below": below, "n_above": above, "n_out_of_range": below + above})


# ---------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="shrinkage", choices=["shrinkage", "recentre", "none"])
    a = ap.parse_args()
    p = pd.read_csv(a.input)
    for col in ("id", "site", "age", "sex", "is_control"):
        if col not in p.columns:
            raise SystemExit(f"missing column: {col}")
    if "fastsurfer_dir" in p.columns:
        vol = build_regions(pd.DataFrame([read_fastsurfer(d) for d in p.fastsurfer_dir], index=p.index))
    else:
        vol = p
    ref = Reference()
    z = ref.z(vol, p.age, p.sex)
    if a.method != "none":
        z = calibrate(z, p.site, p.is_control, a.method)
    res = pd.concat([p[["id", "site", "age", "sex", "is_control"]], count_out_of_range(z), z.add_prefix("z_")], axis=1)
    res.to_csv(a.output, index=False)
    print(f"wrote {a.output}: {len(res)} participants, {z.shape[1]} regions, calibration = {a.method}")


if __name__ == "__main__":
    main()
