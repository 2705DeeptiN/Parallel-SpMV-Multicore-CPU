import csv
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = ROOT / "config/matrices.csv"
meta = ROOT / "datasets/metadata"
output = meta / "static_partition_imbalance.csv"
chunk = 16

def read_row_counts(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        banner = f.readline().strip().lower()
        field = banner.split()[3]
        symmetry = banner.split()[4]
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, cols, nnz = map(int, line.split()[:3])
        row_indices = []
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            p = line.split()
            r, c = int(p[0]) - 1, int(p[1]) - 1
            row_indices.append(r)
            if symmetry in ("symmetric", "hermitian") and r != c:
                row_indices.append(c)
    return np.bincount(row_indices, minlength=rows)

records = []
with open(cfg, newline="", encoding="utf-8-sig") as f:
    for item in csv.DictReader(f):
        if not item.get("matrix_id"):
            continue
        path = ROOT / item["relative_path"]
        if not path.exists():
            print(f"SKIP missing: {path}")
            continue
        counts = read_row_counts(path)
        for threads in (1, 2, 4, 8, 16):
            work = np.zeros(threads, dtype=np.int64)
            for start in range(0, len(counts), chunk):
                thread = (start // chunk) % threads
                work[thread] += counts[start:start + chunk].sum()
            average = work.mean()
            records.append({
                "matrix_id": item["matrix_id"],
                "threads": threads,
                "chunk": chunk,
                "irregularity": counts.max() / counts.mean() if counts.mean() else 0,
                "max_thread_nnz": int(work.max()),
                "mean_thread_nnz": float(average),
                "static_partition_imbalance": float(work.max() / average) if average else 0
            })

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"Updated {output} with {len(records)} workload estimates.")
