# paedvol-centiles

Paediatric reference centiles for automated MRI brain volumetry (FastSurfer, 5-21 years), with local site calibration
by empirical Bayes shrinkage and reference base rates of out-of-range regions.

Code and data accompanying: *Brain volume centiles across childhood and adolescence: transfer to new scanners and the base rate
of out-of-range regions across diagnoses* (Liu X, Zhao X, Jia X, Yang J, Zhang X, Wang X; submitted).

## Using the centiles (`paedvol/paedvol.py`)

Requirements: Python >= 3.10 with numpy and pandas.

1. Segment T1-weighted scans with **FastSurfer 2.6** in segmentation-only mode at 1-mm isotropic resolution
   (`--seg_only --vox_size 1`, CerebNet enabled), as in the article.
2. Prepare a CSV with one row per participant: `id, site, age` (years, 5-21), `sex` (`F`/`M`), `is_control`
   (1 = typically developing control scanned with the local protocol, else 0) and either `fastsurfer_dir`
   (path to the FastSurfer subject folder) or the 92 region columns in cm3 (names as in `data/eb_parameters.csv`).
3. Run

```
python paedvol/paedvol.py --input participants.csv --output results.csv            # empirical Bayes shrinkage (default)
python paedvol/paedvol.py --input participants.csv --output results.csv --method recentre
```

The output contains, for each participant, z scores for 92 regions after local calibration and the numbers of regions
below the 2.5th centile (`n_below`), above the 97.5th centile (`n_above`) and in total. A synthetic example
(`example/example_input.csv`, not real data) can be run with
`python paedvol/paedvol.py --input example/example_input.csv --output example/example_output.csv`.

**Calibration.** Each site's offset is estimated from its local controls and multiplied by the shrinkage factor
B = k*tau2/(k*tau2 + sigma2) (k = number of local controls; tau2 and sigma2 per region in `data/eb_parameters.csv`).
In the article's simulations shrinkage reduced site offsets with as few as five local controls, whereas plain re-centring on
five controls was worse than no calibration; rescaling by the local SD did not help with up to 40 controls. The variances were
estimated from research sites and may underestimate differences among clinical scanners.

**Interpretation.** In held-out typically developing controls of both datasets, 90.2% had at least one region outside the
2.5th-97.5th centiles; the median number was 4 and the 95th centile 23 of 92 regions (absolute volumes). Count below-range
and above-range regions separately: IQ was related to them in opposite directions.

**Accuracy of the tool.** Volumes are converted to z by interpolation on a dense grid (`data/zgrid.csv.gz`; z = -4 to 4 in steps
of 0.1; ages 5-21 years in steps of 0.25). Against exact GAMLSS z scores for 400 random participants x 92 regions the maximum absolute
error was 0.0016 and the |z| > 1.96 classification agreed for 99.997% of values; values beyond |z| = 4 are truncated at
+/-4 and still counted as out of range (`results/A31_tool_validation.json`).

## Data files

| File | Content |
|---|---|
| `data/centiles.csv` | 3rd-97th centiles per region, age (5-21 y, step 0.25) and sex (Supplementary Table S1) |
| `data/zgrid.csv.gz` | volume at z = -4 ... 4 per region, age and sex (used by the tool) |
| `data/eb_parameters.csv` | tau2, sigma2 and shrinkage factors B for k = 5, 10, 20, 40 (Supplementary Table S8) |
| `data/families.csv` | GAMLSS distribution family per region |
| `data/scanners.csv` | acquisition sites and MRI systems (Supplementary Table S2) |
| `results/` | group-level results reported in the article (no participant-level data) |

## Reproducing the analyses (`scripts/`)

Data: ABIDE I/II and HBN imaging data are openly available through the International Neuroimaging Data-sharing Initiative
(FCP-INDI). HBN phenotypic data (release 11) require a data use agreement with the Child Mind Institute and are **not** included;
participant-level data are not redistributed. R 4.6.1 (gamlss 5.5.5, data.table) and Python 3.12 (`requirements.txt`).

| Step | Script |
|---|---|
| Phenotypes and T1 listings | `01_abide_inventory.py`, `03_hbn_inventory.py` (T1 files downloaded from FCP-INDI) |
| Segmentation | FastSurfer 2.6 `--seg_only --vox_size 1` with CerebNet; for five HBN participants an alternative T1 run was used when the first failed segmentation |
| Volumes | `05_extract_volumes.py` |
| ABIDE-only centiles (supplementary analyses) | `A01_prepare_norm_input.py`, `A02_gamlss_norms.R`, `A03_norm_validation_and_asd.py`, `A07_raw_vs_z_algorithms_age_calibration.py` |
| HBN phenotypes (data use agreement) | `A10_extract_hbn_pheno.py` (password from environment variable), `A11_hbn_pheno_merge.py` |
| Joint input and centiles | `A12_prepare_joint_input.py`, `A13_gamlss_joint.R`, `A31_zgrid_centiles.R` |
| Validation and diagnostic groups | `A14_hbn_validation_and_dx.py`, `A15_joint_interpretation.py` |
| Local calibration | `A24_eb_site_adaptation.py` |
| IQ, direction and development | `A25_cognition_development.py` |
| Intracranial-volume sensitivity | `A26a_make_icv_input.py`, then `A13_gamlss_joint.R` into `derivatives/norms_joint_icv`, `A26_icv_sensitivity.py` |
| Other | `A19_scanner_params.py`, `A22_detectable_effects.py`, `A27_visual_qc_summary.py`, `A32_validate_paedvol_tool.py` |

Helper modules: `region_names.py`, `harmonize.py`, `07_feasibility_abide.py`. Random seeds: 42 (main), 7, 123, 2024.

## Citation and licence

See `CITATION.cff`. Code and data files are released under the MIT licence. Please also acknowledge ABIDE and the Child Mind
Institute Healthy Brain Network when using the centiles.
