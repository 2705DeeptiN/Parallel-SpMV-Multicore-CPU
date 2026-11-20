import csv
import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "src" / "spmv.exe"
CONFIG = ROOT / "config" / "experiment.json"
MATRIX_CONFIG = ROOT / "config" / "matrices.csv"
METADATA = ROOT / "datasets" / "metadata" / "matrix_metadata.csv"

OUT = ROOT / "results" / "reference_validation"
REF_DIR = OUT / "references"
OUT.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

import json
cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

with open(MATRIX_CONFIG, newline="", encoding="utf-8-sig") as f:
    matrices = list(csv.DictReader(f))

with open(METADATA, newline="", encoding="utf-8-sig") as f:
    metadata = {
        row["matrix_id"]: float(row["ell_padding_ratio"])
        for row in csv.DictReader(f)
    }

summary_path = OUT / "correctness_summary.csv"
log_path = OUT / "validation.log"

results = []

def run_test(matrix_id, fmt, mode, schedule, matrix_path, vector_path):
    cmd = [
        str(EXE),
        "--matrix", str(matrix_path),
        "--vector", str(vector_path),
        "--dataset", matrix_id,
        "--format", fmt,
        "--mode", mode,
        "--schedule", schedule,
        "--threads", "4" if mode == "omp" else "1",
        "--chunk", "16",
        "--runs", "2",
        "--repetitions", "1"
    ]

    try:
        p = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            timeout=cfg.get("timeout_seconds", 3600)
        )
        output = p.stdout + p.stderr
        errors = re.findall(r"max_abs_error=([0-9.eE+-]+)", output)
        status = "PASS" if p.returncode == 0 and len(errors) >= 2 else "FAIL"
        max_error = max(map(float, errors)) if errors else ""

    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
        status = "TIMEOUT"
        max_error = ""

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*65}\n")
        log.write(f"{matrix_id} | {fmt} | {mode} | {schedule}\n")
        log.write(output + "\n")

    results.append({
        "matrix_id": matrix_id,
        "format": fmt,
        "mode": mode,
        "schedule": schedule,
        "threads": 4 if mode == "omp" else 1,
        "status": status,
        "max_abs_error": max_error
    })

    print(f"{status}: {matrix_id} | {fmt} | {mode} | {schedule}")

if not EXE.exists():
    raise SystemExit("ERROR: src/spmv.exe not found. Build it first.")

log_path.write_text("Team 5 independent-reference correctness validation\n",
                    encoding="utf-8")

for m in matrices:
    mid = m["matrix_id"]
    mp = ROOT / m["relative_path"]
    vp = ROOT / "datasets" / "vectors" / f"{mid}_x.txt"

    if not mp.exists() or not vp.exists():
        print(f"SKIP: missing matrix or vector: {mid}")
        results.append({
            "matrix_id": mid, "format": "all", "mode": "all",
            "schedule": "all", "threads": "",
            "status": "MISSING_INPUT", "max_abs_error": ""
        })
        continue

    # Save one independent reference vector per matrix.
    refpath = REF_DIR / f"{mid}_reference.txt"
    refcmd = [
        str(EXE), "--matrix", str(mp), "--vector", str(vp),
        "--dataset", mid, "--format", "csr", "--mode", "seq",
        "--threads", "1", "--runs", "2", "--repetitions", "1",
        "--reference-vector", str(refpath)
    ]

    try:
        p = subprocess.run(
            refcmd, cwd=ROOT, capture_output=True, text=True,
            timeout=cfg.get("timeout_seconds", 3600)
        )
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\nREFERENCE: {mid}\n{p.stdout}{p.stderr}\n")

        if p.returncode != 0 or not refpath.exists():
            print(f"REFERENCE FAILED: {mid}")
            results.append({
                "matrix_id": mid, "format": "reference", "mode": "seq",
                "schedule": "static", "threads": 1,
                "status": "FAIL", "max_abs_error": ""
            })
            continue

    except subprocess.TimeoutExpired:
        print(f"REFERENCE TIMEOUT: {mid}")
        continue

    for fmt in ("coo", "csr", "ell"):
        if fmt == "ell" and metadata.get(mid, float("inf")) > float(
            cfg.get("max_ell_padding_ratio", 20.0)
        ):
            print(f"SKIP ELL: {mid} (padding limit)")
            continue

        run_test(mid, fmt, "seq", "static", mp, vp)

        for schedule in ("static", "dynamic", "guided"):
            run_test(mid, fmt, "omp", schedule, mp, vp)

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    fields = [
        "matrix_id", "format", "mode", "schedule",
        "threads", "status", "max_abs_error"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

passed = sum(r["status"] == "PASS" for r in results)
failed = sum(r["status"] != "PASS" for r in results)

print("\nValidation finished.")
print(f"Passed: {passed}")
print(f"Failed/skipped: {failed}")
print(f"Summary: {summary_path}")
print(f"Log: {log_path}")
