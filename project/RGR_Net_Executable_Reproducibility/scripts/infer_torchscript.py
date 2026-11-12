#!/usr/bin/env python3
"""Run a REAL user-supplied trained TorchScript model on locked held-out images.
Creates enhanced images and an output registry; no reference/surrogate model is included.
"""
import argparse,csv,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgr_repro.evidence import read_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--checkpoint',required=True);p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 if a.seed not in range(5):raise SystemExit('Seed must be 0,1,2,3 or 4')
 root=Path(a.root);cp=Path(a.checkpoint)
 if not cp.is_file():raise SystemExit('Trained original TorchScript checkpoint missing')
 rows=[r for r in read_csv(root/'manifests/image_ids.csv') if r.get('split')=='test']
 if not rows:raise SystemExit('Locked test manifest is empty')
 net=torch.jit.load(str(cp),map_location=a.device).eval();outrows=[]
 with torch.inference_mode():
  for r in rows:
   src=root/r['image_path'];im=np.array(Image.open(src).convert('RGB'),dtype=np.float32)/255
   x=torch.from_numpy(im.transpose(2,0,1))[None].to(a.device)
   y=net(x)
   if not isinstance(y,torch.Tensor):raise TypeError('TorchScript output must be an RGB tensor; supply author adapter for other output types')
   if y.shape!=x.shape:raise ValueError(f'Output dimensions mismatch for {r["image_id"]}')
   pixels=np.clip(y.detach().cpu().numpy()[0].transpose(1,2,0),0,1)
   dst=root/'results'/'enhanced_outputs'/a.method/f'seed{a.seed}'/r['dataset']/(r['image_id']+'.png')
   dst.parent.mkdir(parents=True,exist_ok=True)
   Image.fromarray((pixels*255+0.5).astype(np.uint8)).save(dst)
   outrows.append({'dataset':r['dataset'],'image_id':r['image_id'],'method':a.method,'seed':a.seed,'enhanced_path':str(dst.relative_to(root))})
 target=root/'runs'/'enhanced_outputs.csv';old=read_csv(target)
 old=[r for r in old if not (r['method']==a.method and r['seed']==str(a.seed))]
 with target.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['dataset','image_id','method','seed','enhanced_path']);writer.writeheader();writer.writerows(old+outrows)
 print('Wrote',len(outrows),'enhanced test images for',a.method,'seed',a.seed)
if __name__=='__main__':main()
