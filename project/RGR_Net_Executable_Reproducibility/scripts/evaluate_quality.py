#!/usr/bin/env python3
"""Evaluate genuine enhanced image files. Never generates images or dummy metrics.
Input runs/enhanced_outputs.csv: dataset,image_id,method,seed,enhanced_path
Original manifest: manifests/image_ids.csv: dataset,image_id,split,image_path,reference_path,...
LPIPS and NIQE are optional; stop when requested metric implementation unavailable.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgr_repro.evidence import read_csv

def open_rgb(p):return np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255.

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--lpips',action='store_true');p.add_argument('--niqe',action='store_true');a=p.parse_args();root=Path(a.root)
 ref={(r['dataset'],r['image_id']):r for r in read_csv(root/'manifests/image_ids.csv') if r.get('split')=='test'}
 outs=read_csv(root/'runs/enhanced_outputs.csv')
 if not ref or not outs:raise SystemExit('Real test manifest and enhanced_outputs.csv must be populated first.')
 lpips=None;niqe=None;torch=None
 if a.lpips:
  import torch as tt,lpips as pkg
  torch=tt;lpips=pkg.LPIPS(net='alex').eval()
 if a.niqe:
  import torch as tt,pyiqa
  torch=tt;niqe=pyiqa.create_metric('niqe',device='cpu')
 result=[]
 for r in outs:
  k=(r['dataset'],r['image_id'])
  if k not in ref:raise ValueError(f'Not locked test split: {k}')
  cand=root/r['enhanced_path']
  if not cand.is_file():raise FileNotFoundError(cand)
  img=open_rgb(cand); refpath=ref[k].get('reference_path','').strip()
  o=dict(dataset=k[0],image_id=k[1],method=r['method'],seed=r['seed'],psnr='',ssim='',lpips='',niqe='')
  if refpath:
   rr=open_rgb(root/refpath)
   if rr.shape!=img.shape:raise ValueError(f'Image/reference dimension mismatch: {k}')
   o['psnr']=float(peak_signal_noise_ratio(rr,img,data_range=1.0))
   o['ssim']=float(structural_similarity(rr,img,data_range=1.,channel_axis=2))
   if lpips is not None:
    aa=torch.from_numpy(rr.transpose(2,0,1))[None]*2-1
    bb=torch.from_numpy(img.transpose(2,0,1))[None]*2-1
    with torch.no_grad():o['lpips']=float(lpips(aa,bb))
  if niqe is not None:
   xx=torch.from_numpy(img.transpose(2,0,1))[None]
   with torch.no_grad():o['niqe']=float(niqe(xx))
  result.append(o)
 out=root/'results/per_image_quality_measured.csv';out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['dataset','image_id','method','seed','psnr','ssim','lpips','niqe']);w.writeheader();w.writerows(result)
 print(f'Evaluated {len(result)} real enhanced files; wrote {out}')
if __name__=='__main__':main()
