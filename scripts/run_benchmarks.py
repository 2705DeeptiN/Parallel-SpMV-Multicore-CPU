import csv,json,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config/experiment.json").read_text())
exe=ROOT/"src/spmv.exe"
if not exe.exists(): raise SystemExit("Build src/spmv.exe first; see README.")
matcfg=list(csv.DictReader(open(ROOT/cfg["matrix_config"],newline="",encoding="utf-8-sig")))
metadata_path = ROOT/"datasets/metadata/matrix_metadata.csv"
metadata = {
    r["matrix_id"]: float(r["ell_padding_ratio"])
    for r in csv.DictReader(open(metadata_path, newline="", encoding="utf-8-sig"))
}
max_ell_padding = float(cfg.get("max_ell_padding_ratio", 20.0))
rawdir=ROOT/"results/raw";rawdir.mkdir(parents=True,exist_ok=True)
raw=rawdir/os.environ.get("TEAM5_RAW_CSV","team5_suitesparse_validation.csv")
header="cluster,problem,platform,implementation,dataset,size,threads_or_ranks,run_id,total_ms,kernel_ms,transfer_ms,throughput,error,power_w,warmup,schedule,chunk,first_touch,proc_bind\n"

if raw.exists():
    raise SystemExit(f"Refusing to append to existing CSV: {raw}. Choose a new TEAM5_RAW_CSV filename.")
raw.write_text(header, encoding="utf-8")
failures = []
for m in matcfg:
    mp=ROOT/m["relative_path"]
    vp=ROOT/"datasets/vectors"/(m["matrix_id"]+"_x.txt")
    if not mp.exists() or not vp.exists():
        print("SKIP: prepare/download missing",m["matrix_id"]);continue
    for fmt in cfg["formats"]:
      if fmt == "ell" and metadata.get(m["matrix_id"], float("inf")) > max_ell_padding:
       print("SKIP ELL:", m["matrix_id"], "padding ratio =", metadata[m["matrix_id"]], "limit =", max_ell_padding)
       continue
      for mode in cfg["modes"]:
       schedules=cfg["schedules"] if mode=="omp" else ["static"]
       for sch in schedules:
        for th in cfg["threads"] if mode=="omp" else [1]:
         for ch in cfg["chunks"] if mode=="omp" else [1]:
          for bind in cfg.get("proc_bind",["false"]):
           env=os.environ.copy();env["OMP_PROC_BIND"]=str(bind);env["OMP_PLACES"]="cores"
           cmd=[str(exe),"--matrix",str(mp),"--vector",str(vp),"--dataset",m["matrix_id"],"--format",fmt,"--mode",mode,
               "--schedule",sch,"--threads",str(th),"--chunk",str(ch),"--runs",str(cfg["runs"]),"--repetitions",str(cfg["repetitions"]),"--raw",str(raw)]
           print("RUN"," ".join(cmd),"OMP_PROC_BIND=",bind)
           try: subprocess.run(cmd,cwd=ROOT,check=True,timeout=cfg["timeout_seconds"],env=env)
           except subprocess.CalledProcessError as exc: failures.append((m["matrix_id"],fmt,mode,sch,th,ch,bind,"FAILED",exc.returncode))
           except subprocess.TimeoutExpired: failures.append((m["matrix_id"],fmt,mode,sch,th,ch,bind,"TIMEOUT",""))
print("Raw measurements:", raw)
if failures:
    print(f"BENCHMARK RUN FAILED: {len(failures)} configuration(s) failed or timed out.")
    for item in failures:
        print("FAILURE:", item)
    raise SystemExit(1)
print("BENCHMARK RUN COMPLETED: no failed configurations.")
