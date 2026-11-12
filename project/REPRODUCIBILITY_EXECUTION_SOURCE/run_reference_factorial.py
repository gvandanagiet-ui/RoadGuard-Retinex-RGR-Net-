#!/usr/bin/env python3
"""Enumerate or execute a 2^4 factorial study using the independent research
reconstruction. Execution DOES NOT reproduce the original manuscript model.
"""
import argparse,csv,itertools,json,subprocess,sys
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]/"INDEPENDENT_REFERENCE_NOT_ORIGINAL"/"reference_code"/"train_reference.py"

def jobs(seeds):
    for bits in itertools.product((0,1),repeat=4):
        cell=1+8*bits[0]+4*bits[1]+2*bits[2]+bits[3]
        for seed in seeds:
            yield {"cell":f"F{cell:02d}","G":bits[0],"E":bits[1],"S":bits[2],"C":bits[3],"seed":seed,"implementation":"independent_research_reconstruction"}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--train-dir")
    a.add_argument("--val-dir")
    a.add_argument("--out",required=True)
    a.add_argument("--execute",action="store_true",help="Run actual training; otherwise create job registry only")
    a.add_argument("--dry-run",action="store_true")
    a.add_argument("--epochs",type=int,default=100)
    a.add_argument("--batch-size",type=int,default=8)
    a.add_argument("--crop",type=int,default=256)
    a.add_argument("--width",type=int,default=12)
    a.add_argument("--max-batches",type=int,default=0)
    a.add_argument("--device",default="cpu")
    a.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    a.add_argument("--cells",nargs="+",default=None,help="Optional F01...F16 subset for a smoke test")
    args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    registry=list(jobs(args.seeds))
    if args.cells is not None:
        selected=set(args.cells)
        valid={f"F{i:02d}" for i in range(1,17)}
        if not selected.issubset(valid):a.error("--cells must be F01..F16")
        registry=[j for j in registry if j["cell"] in selected]
    with (out/"factorial_job_registry.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(registry[0]));wr.writeheader();wr.writerows(registry)
    if args.execute:
        if not args.train_dir or not args.val_dir: a.error("--train-dir and --val-dir are required with --execute")
        for j in registry:
            cmd=[sys.executable,str(SOURCE),"--train_dir",args.train_dir,"--val_dir",args.val_dir,"--out",str(out/j["cell"]/("seed_"+str(j["seed"]))),"--seed",str(j["seed"]),"--epochs",str(args.epochs),"--batch-size",str(args.batch_size),"--crop",str(args.crop),"--width",str(args.width),"--device",args.device]
            if args.max_batches: cmd.extend(["--max-batches",str(args.max_batches)])
            if j["G"]==0:cmd.append("--shared-gates")
            if j["E"]==0:cmd.append("--no-spatial-target")
            if j["S"]==0:cmd.append("--no-structure-loss")
            if j["C"]==0:cmd.append("--no-consistency")
            subprocess.run(cmd,check=True)
    summary={"jobs":len(registry),"executed":bool(args.execute),"distinct_cells":len(set(j['cell'] for j in registry)),"seeds":args.seeds,"model_provenance":"independent research reconstruction; not original RGR-Net"}
    (out/"execution_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
