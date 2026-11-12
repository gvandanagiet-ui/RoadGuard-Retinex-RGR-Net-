"""Execute short 5-seed reference training and 5-seed shared-gate PROXY.
No assertion that either matches published RGR-Net or UDCN. No paper metrics.
"""
import argparse,json,subprocess,sys,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rows=[]
for model in ['reference_rgr','reference_shared_proxy']:
 for seed in range(5):
    dest=out/model/f'seed_{seed}'
    cmd=[sys.executable,str(Path(__file__).with_name('train_reference.py')),'--train_dir',str(Path(a.data)/'train'/'low'),'--val_dir',str(Path(a.data)/'val'/'low'),'--out',str(dest),'--seed',str(seed),'--variant',model,'--epochs','1','--batch-size','2','--crop','32','--width','8','--max-batches','2']
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    m=json.loads((dest/'run_manifest.json').read_text())
    rows.append({'model':model,'seed':seed,'train_mode':'artificial diagnostic smoke only','train_images':len(m['train_files']),'val_images':len(m['val_files']),'unsup_val_loss':m['best_unsup_validation_loss'],'checkpoint':str(dest/'best_checkpoint.pt'),'params':m['parameter_count']})
with (out/'new_training_runs_DIAGNOSTIC_ONLY.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
(out/'NOT_FOR_PAPER_RESULTS.txt').write_text('10 actual software smoke-training jobs executed (5 reference RGR + 5 shared-gate proxy) using ARTIFICIAL images. These are NOT RGR-Net manuscript-model runs, NOT matched UDCN baselines, NOT a real benchmark and NOT acceptable empirical evidence for any PSNR/SSIM/mAP claim.\n')
print(json.dumps({'runs_completed':len(rows),'research_valid_training_runs':0,'diagnostic_only':True},indent=2))
