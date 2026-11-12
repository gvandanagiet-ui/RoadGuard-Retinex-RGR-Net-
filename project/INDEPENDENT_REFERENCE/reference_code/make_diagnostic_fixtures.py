"""Generate ARTIFICIAL raster input for verifying software only.
DO NOT use as evidence of restoration quality on LOL/ExDark/LLVIP/HUE.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out)
for split,n in [('train',6),('val',4)]:
    folder=root/split/'low';folder.mkdir(parents=True,exist_ok=True)
    for i in range(n):
        rng=np.random.default_rng(i+({'train':10,'val':100}[split]))
        h=w=56
        xx,yy=np.meshgrid(np.linspace(0,1,w),np.linspace(0,1,h))
        im=np.stack([.04+.11*xx+.04*yy,.05+.06*xx+.06*yy,.03+.1*xx],axis=-1)
        im=im+rng.normal(0,.01,(h,w,1))
        im[12+i:16+i,18:22,:]=.97
        Image.fromarray(np.uint8(np.clip(im,0,1)*255)).save(folder/f'fixture_{split}_{i:02}.png')
print('Artificial diagnostic fixtures created:',root)
