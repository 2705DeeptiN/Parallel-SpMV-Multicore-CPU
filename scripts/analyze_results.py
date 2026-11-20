from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
raw=ROOT/"results/raw/team5_suitesparse_raw.csv"
out=ROOT/"results/processed";plots=ROOT/"results/plots"
out.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
if not raw.exists(): raise SystemExit("Run benchmarks first.")
d=pd.read_csv(raw)
d=d[d["warmup"]==0].copy()
d["throughput"]=pd.to_numeric(d["throughput"],errors="coerce")
d["total_ms"]=pd.to_numeric(d["total_ms"],errors="coerce")
group=["dataset","implementation","threads_or_ranks","schedule","chunk","proc_bind"]
summary=d.groupby(group,dropna=False).agg(mean_ms=("kernel_ms","mean"),std_ms=("kernel_ms","std"),mean_gflops=("throughput","mean"),std_gflops=("throughput","std"),runs=("run_id","count"),max_error=("error","max")).reset_index()
summary.to_csv(out/"team5_summary.csv",index=False)
# 1 GFLOP/s per format per matrix (best thread/schedule is explicitly labelled)
best=summary.sort_values("mean_gflops",ascending=False).groupby(["dataset","implementation"]).head(1)
for ds,g in best.groupby("dataset"):
    plt.figure(figsize=(8,4));plt.bar(g["implementation"],g["mean_gflops"],yerr=g["std_gflops"].fillna(0),capsize=4)
    plt.ylabel("GFLOP/s");plt.title(f"Best measured GFLOP/s – {ds}");plt.tight_layout();plt.savefig(plots/f"gflops_{ds}.png",dpi=180);plt.close()

# 2 Speedup vs threads: compare each OpenMP schedule separately
seq_rows = summary[summary.implementation.str.endswith("_seq")].copy()
omp_rows = summary[summary.implementation.str.endswith("_omp")].copy()

seq_rows["format"] = seq_rows.implementation.str.split("_").str[0]
omp_rows["format"] = omp_rows.implementation.str.split("_").str[0]

baseline_keys = ["dataset", "format", "proc_bind"]

seq_baseline = seq_rows[
    baseline_keys + ["mean_ms"]
].rename(columns={"mean_ms": "seq_ms"})

speedup_data = omp_rows.merge(
    seq_baseline,
    on=baseline_keys,
    how="inner"
)

speedup_data["speedup"] = (
    speedup_data["seq_ms"] / speedup_data["mean_ms"]
)

for ds, g in speedup_data.groupby("dataset"):
    plt.figure(figsize=(8, 5))

    for (fmt, schedule), h in g.groupby(["format", "schedule"]):
        h = h[h["chunk"] == 16]
        z = h.groupby("threads_or_ranks")["speedup"].mean()

        plt.plot(
            z.index,
            z.values,
            marker="o",
            label=f"{fmt.upper()} - {schedule}"
        )

    plt.axhline(1.0, linestyle="--", label="Sequential baseline")
    plt.xlabel("OpenMP threads")
    plt.ylabel("Speedup")
    plt.title(f"OpenMP speedup vs sequential – {ds}")
    plt.xticks([1, 4, 8])
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / f"speedup_{ds}.png", dpi=180)
    plt.close()


# 3 Fair scheduling comparison using matched configurations
omp_summary = summary[summary.implementation.str.endswith("_omp")].copy()
omp_summary["format"] = omp_summary.implementation.str.split("_").str[0]

match_keys = ["dataset", "format", "threads_or_ranks", "chunk", "proc_bind"]

sched = omp_summary.pivot_table(
    index=match_keys,
    columns="schedule",
    values="mean_gflops",
    aggfunc="mean"
)

required = ["static", "dynamic", "guided"]

if all(s in sched.columns for s in required):
    matched = sched.dropna(subset=required)
    normalized = matched[required].div(matched["static"], axis=0)

    means = normalized.mean()
    stds = normalized.std()

    plt.figure(figsize=(7, 4))
    plt.bar(required, means, yerr=stds.fillna(0), capsize=4)
    plt.axhline(1.0, linestyle="--", label="Static baseline")
    plt.ylabel("Relative GFLOP/s (static = 1.0)")
    plt.title(f"Matched OpenMP scheduling comparison (n={len(normalized)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots/"scheduling_comparison.png", dpi=180)
    plt.close()
else:
    print("WARNING: Insufficient matched scheduling data.")


# 4 static row-partition load imbalance vs matrix irregularity
imb=ROOT/"datasets/metadata/static_partition_imbalance.csv"
if imb.exists():
    q=pd.read_csv(imb)
    plt.figure(figsize=(7,4))
    for nt,h in q.groupby("threads"):
        plt.errorbar(h.irregularity,h.static_partition_imbalance,yerr=np.zeros(len(h)),fmt="o",capsize=2,label=f"{nt} threads (deterministic partition metric)")
    plt.xlabel("Matrix irregularity (max NNZ/row ÷ mean NNZ/row)");plt.ylabel("Static partition imbalance (max thread NNZ ÷ mean thread NNZ)")
    plt.title("Static row-partition load imbalance vs matrix irregularity");plt.legend();plt.tight_layout();plt.savefig(plots/"load_imbalance_vs_irregularity.png",dpi=180);plt.close()
# 5 Bandwidth roofline: measured STREAM bandwidth; compute ceiling unavailable.
rp=ROOT/"config/roofline.json"
meta=ROOT/"datasets/metadata/matrix_metadata.csv"
if rp.exists() and meta.exists():
    import json
    rc=json.loads(rp.read_text())
    bw=float(rc.get("stream_bandwidth_gbs",0))

    if bw>0:
        x=np.logspace(-3,3,300)
        y=bw*x

        plt.figure(figsize=(7,5))
        plt.loglog(x,y,label=f"STREAM bandwidth roof ({bw:.4f} GB/s)")

        md=pd.read_csv(meta)
        pts=summary.merge(md,left_on="dataset",right_on="matrix_id")

        for fmt,h in pts.groupby(pts.implementation.str.split("_").str[0]):
            if fmt=="coo":
                bytes_nz=np.full(len(h),24.0)
            elif fmt=="csr":
                bytes_nz=20.0+4.0*h.rows.to_numpy()/h.nnz.to_numpy()
            else:
                bytes_nz=4.0*h.ell_padding_ratio.to_numpy()+16.0

            ai=2.0/bytes_nz
            plt.errorbar(ai,h.mean_gflops.to_numpy(),
                         yerr=h.std_gflops.fillna(0).to_numpy(),
                         fmt="o",capsize=3,label=f"{fmt} measured")

        plt.xlabel("Arithmetic intensity (FLOP/byte)")
        plt.ylabel("GFLOP/s")
        plt.title("SpMV bandwidth roofline (compute ceiling unavailable)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots/"roofline.png",dpi=180)
        plt.close()
print("Processed results:",out/"team5_summary.csv")
print("Plots:",plots)
