#!/usr/bin/env python3
"""Benchmark original/exported TorchScript on a declared device. No benchmark data fabricated."""
import argparse,datetime,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgr_repro.evidence import sha256
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=1);p.add_argument('--warmup',type=int,default=200);p.add_argument('--repeats',type=int,default=1000);p.add_argument('--output',default='results/benchmark_measured.json');a=p.parse_args()
if a.device=='cuda' and not torch.cuda.is_available():raise SystemExit('No CUDA device: measurements cannot be reported as GPU latency.')
cp=Path(a.checkpoint).resolve();model=torch.jit.load(str(cp),map_location=a.device).eval()
rows=[]
with torch.inference_mode():
 for h,w in [(256,256),(400,600)]:
  x=torch.rand(a.batch,3,h,w,device=a.device)
  def sync():
   if a.device=='cuda':torch.cuda.synchronize()
  try:
   for _ in range(a.warmup):model(x)
   sync();vals=[]
   if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
   for _ in range(a.repeats):
    t=time.perf_counter();model(x);sync();vals.append((time.perf_counter()-t)*1000.)
  except RuntimeError as e:raise SystemExit(f'Execution failed at {h}×{w}: {e}')
  rows.append(dict(height=h,width=w,batch=a.batch,mean_ms=statistics.mean(vals),median_ms=statistics.median(vals),p95_ms=sorted(vals)[int(.95*(len(vals)-1))],peak_allocated_gb=(torch.cuda.max_memory_allocated()/(1024**3) if a.device=='cuda' else None)))
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(checkpoint=str(cp),checkpoint_sha256=sha256(cp),device=a.device,gpu=(torch.cuda.get_device_name(0) if a.device=='cuda' else None),torch=torch.__version__,date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),warmup=a.warmup,repeats=a.repeats,measurements=rows),indent=2))
print(out)
