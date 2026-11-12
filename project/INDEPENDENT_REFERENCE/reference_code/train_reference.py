"""A runnable reference *training system*, NOT the authors' original RGR-Net.
No LOL test pair or label is used for gradient updates. Select checkpoint on
unsupervised validation objective only, not normal-light test PSNR.
"""
import argparse,csv,hashlib,json,random,time,platform,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from model import ReferenceRGR,unsup_loss

class LowDataset(Dataset):
    def __init__(self,folder,crop):
        self.files=sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}]);self.crop=crop
        if len(self.files)<2:raise ValueError('At least two distinct low-light images per split required')
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        p=self.files[i];im=Image.open(p).convert('RGB');w,h=im.size
        if w<self.crop or h<self.crop: im=im.resize((max(w,self.crop),max(h,self.crop)))
        w,h=im.size
        x=random.randrange(w-self.crop+1);y=random.randrange(h-self.crop+1)
        arr=np.asarray(im.crop((x,y,x+self.crop,y+self.crop)),dtype=np.float32)/255
        return torch.from_numpy(arr.copy()).permute(2,0,1)

def load_args():
    a=argparse.ArgumentParser()
    for s in ['train_dir','val_dir','out']:a.add_argument('--'+s,required=True)
    a.add_argument('--seed',type=int,default=0);a.add_argument('--epochs',type=int,default=1)
    a.add_argument('--batch-size',type=int,default=2);a.add_argument('--crop',type=int,default=64)
    a.add_argument('--width',type=int,default=12);a.add_argument('--lr',type=float,default=1e-4)
    a.add_argument('--max-batches',type=int,default=0)
    a.add_argument('--device',default='cpu');a.add_argument('--variant',choices=['reference_rgr','reference_shared_proxy'],default='reference_rgr')
    a.add_argument('--no-illum-gate',action='store_true');a.add_argument('--no-noise-gate',action='store_true');a.add_argument('--no-spatial-target',action='store_true');a.add_argument('--no-structure-loss',action='store_true');a.add_argument('--no-consistency',action='store_true');a.add_argument('--no-q',action='store_true');a.add_argument('--shared-gates',action='store_true')
    return a.parse_args()

def main():
    a=load_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.epochs<1 or a.crop%8:raise ValueError('epochs>=1 and crop must be multiple of 8')
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    train=LowDataset(a.train_dir,a.crop);val=LowDataset(a.val_dir,a.crop)
    t_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0)
    v_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
    shared=a.shared_gates or a.variant=='reference_shared_proxy'
    use_q=not a.no_q and a.variant=='reference_rgr'
    model=ReferenceRGR(a.width,dual_gate=not shared,use_q=use_q,illum_gate=not a.no_illum_gate,noise_gate=not a.no_noise_gate).to(a.device)
    optim=torch.optim.Adam(model.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=a.epochs)
    best=float('inf');logs=[];tick=time.monotonic()
    for epoch in range(a.epochs):
        summary={}
        for phase,ldr in [('train',t_loader),('val',v_loader)]:
            model.train(phase=='train');losses=[]
            for j,x in enumerate(ldr):
                if a.max_batches and j>=a.max_batches:break
                x=x.to(a.device)
                with torch.set_grad_enabled(phase=='train'):
                    enhanced,aux=model(x)
                    perturbed=None
                    if phase=='train' and not a.no_consistency:
                        perturbed,_=model(x,perturb=True)
                    loss,parts=unsup_loss(x,enhanced,aux,not a.no_spatial_target,not a.no_structure_loss,perturbed)
                    if phase=='train':
                        optim.zero_grad();loss.backward();optim.step()
                    losses.append(float(loss.detach()))
            summary[phase]=float(np.mean(losses))
        scheduler.step()
        logs.append({'epoch':epoch+1,'seed':a.seed,'variant':a.variant,**summary})
        if summary['val']<best:
            best=summary['val']
            torch.save({'state_dict':model.state_dict(),'seed':a.seed,'architecture':'independent reference','args':vars(a)},out/'best_checkpoint.pt')
    with (out/'loss_log.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(logs[0]));wr.writeheader();wr.writerows(logs)
    conf={'architecture_notice':'INDEPENDENT REFERENCE IMPLEMENTATION; NOT ORIGINAL RGR-NET','evidence_class':'training-pipeline software smoke test' if 'smoke' in str(out).lower() else 'independent reference training—not paper model','args':vars(a),'train_files':[str(p.name) for p in train.files],'val_files':[str(p.name) for p in val.files],'pytorch':torch.__version__,'python':platform.python_version(),'parameter_count':sum(p.numel() for p in model.parameters()),'best_unsup_validation_loss':best,'wall_seconds':time.monotonic()-tick,'status':'completed'}
    with (out/'run_manifest.json').open('w') as f:json.dump(conf,f,indent=2)
    print(json.dumps({'seed':a.seed,'variant':a.variant,'val_unsup_loss':best,'params':conf['parameter_count'],'seconds':round(conf['wall_seconds'],2)},indent=2))
if __name__=='__main__':main()
