"""Extract the encrypted HBN phenotype archive (R11, DUA-restricted).

Password is read from env var HBN_PHENO_PW (never stored in files / git).
Usage: HBN_PHENO_PW=... python -I scripts/A10_extract_hbn_pheno.py <archive.7z> <out_dir>
Writes results/A10_hbn_pheno_extract.json (file list + sizes + archive sha256).
"""
import hashlib, json, os, sys, datetime
from pathlib import Path
import py7zr

arc, out = Path(sys.argv[1]), Path(sys.argv[2])
pw = os.environ["HBN_PHENO_PW"]
out.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256(arc.read_bytes()).hexdigest()
with py7zr.SevenZipFile(arc, "r", password=pw) as z:
    z.extractall(out)
files = sorted(p for p in out.rglob("*") if p.is_file())
rep = {
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "script": "scripts/A10_extract_hbn_pheno.py",
    "archive": arc.name, "archive_sha256": sha,
    "n_files": len(files),
    "files": [{"path": str(p.relative_to(out)).replace("\\", "/"), "bytes": p.stat().st_size} for p in files],
}
Path("results/A10_hbn_pheno_extract.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(files), "files extracted")
