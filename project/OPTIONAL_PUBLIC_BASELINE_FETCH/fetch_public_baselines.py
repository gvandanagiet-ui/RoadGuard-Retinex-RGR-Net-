#!/usr/bin/env python3
"""Download project baseline sources only when network is available and licenses permit.

DO NOT treat official pretrained models as equally retrained baselines.
This script uses git clone and records exact commit SHA; it never supplies
original RGR-Net source, which is not publicly verified.
"""
from __future__ import annotations
import argparse,subprocess,json,hashlib
from pathlib import Path
SOURCES={
 'Zero-DCE++':'https://github.com/Li-Chongyi/Zero-DCE_extension.git',
 'UDCN':'https://github.com/myd945/UDCN.git',
 'EnlightenGAN':'https://github.com/VITA-Group/EnlightenGAN.git',
 'GLARE':'https://github.com/LowLevelAI/GLARE.git',
 'DI-Retinex':'https://github.com/sunshangquan/DI-retinex-low-light-enhancement.git',
 'RetinexFormer':'https://github.com/caiyuanhao1998/Retinexformer.git',
 'AGLLDiff':'https://github.com/LYL1015/AGLLDiff.git',
 'RetinexMamba':'https://github.com/YhuoyuH/RetinexMamba.git',
 'MambaLLIE':'https://github.com/wengjiangwei/MambaLLIE.git',
 'HVI-CIDNet':'https://github.com/Fediory/HVI-CIDNet.git',
 'Multinex':'https://github.com/albrateanu/multinex.git',
 'LightenDiffusion':'https://github.com/JianghaiSCU/LightenDiffusion.git',
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='downloaded_baselines');p.add_argument('--method',choices=['ALL',*SOURCES],default='ALL');p.add_argument('--depth',type=int,default=1);a=p.parse_args()
 root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
 selected=SOURCES.items() if a.method=='ALL' else [(a.method,SOURCES[a.method])]
 registry=[]
 for name,url in selected:
  folder=root/name
  entry={'method':name,'repo':url,'destination':str(folder),'status':'not_cloned'}
  if folder.exists():
   entry['status']='existing_directory_not_overwritten'
  else:
   try:
    # Manual review of each project's license is required before redistribution.
    proc=subprocess.run(['git','clone','--depth',str(a.depth),url,str(folder)],capture_output=True,text=True,timeout=180)
    if proc.returncode:entry['status']='failed';entry['error']=proc.stderr[-400:]
    else:entry['status']='cloned'
   except Exception as exc:entry['status']='failed';entry['error']=str(exc)
  if (folder/'.git').exists():
   proc=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],capture_output=True,text=True)
   if not proc.returncode:entry['git_commit']=proc.stdout.strip()
  registry.append(entry)
 (root/'baseline_source_registry.json').write_text(json.dumps(registry,indent=2))
 print(json.dumps(registry,indent=2))
if __name__=='__main__':main()
