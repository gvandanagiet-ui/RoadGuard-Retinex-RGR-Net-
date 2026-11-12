#!/usr/bin/env python3
"""Run-level matched enhancement inference. No tests performed without actual inputs."""
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.stats import ttest_1samp

PRIMARY=('psnr','ssim','lpips')

def holm(pvalues):
    ranked=sorted(pvalues,key=pvalues.get)
    n=len(ranked);results={};prev=0.0
    for i,key in enumerate(ranked):
        prev=max(prev,min(1.0,(n-i)*pvalues[key]));results[key]=prev
    return results

def analyze(csvfile,dataset,model='RGR-Net',baseline='UDCN'):
    df=pd.read_csv(csvfile,dtype={'image_id':'string','camera_id':'string','dataset':'string','method':'string'})
    require={'seed','method','dataset','camera_id','image_id',*PRIMARY}
    missing=require-set(df.columns)
    if missing:raise ValueError('Missing metric fields '+str(missing))
    df=df[(df.dataset==dataset)&(df.method.isin([model,baseline]))].copy()
    if df.empty:raise ValueError('No metric observations: templates contain no results.')
    keys=['seed','dataset','camera_id','image_id']
    if df[keys+['method']].isna().any().any():raise ValueError('Missing identity keys')
    if df.duplicated(keys+['method']).any():raise ValueError('Duplicated run/image/method combinations')
    seeds=sorted(df.seed.unique().tolist())
    if len(seeds)<3:raise ValueError('Insufficient independent training seeds (at least 3).')
    reports={};pvals={}
    for metric in PRIMARY:
        work=df[keys+['method',metric]].copy()
        if work[metric].isna().any():raise ValueError(f'{metric}: missing paired scores; analyze each eligible dataset separately')
        paired=work.pivot(index=keys,columns='method',values=metric)
        if baseline not in paired or model not in paired:raise ValueError('Need both methods at every seed')
        if paired[[model,baseline]].isna().any().any():raise ValueError('Unmatched test IDs between models or seeds')
        per_run=paired[model].sub(paired[baseline]).groupby(level='seed').mean()
        supports=paired.groupby(level='seed').size()
        if supports.nunique()!=1:raise ValueError('Different image count across training seeds')
        ids_by_seed=[set(paired.xs(seed,level='seed').index) for seed in seeds]
        if any(ids_by_seed[0]!=s for s in ids_by_seed[1:]):raise ValueError('Image ID set varies across seeds')
        delta=per_run.to_numpy(float);n=len(delta);mean=float(np.mean(delta));sd=float(np.std(delta,ddof=1))
        se=sd/(n**0.5);half=float(tdist.ppf(.975,n-1)*se)
        p=float(ttest_1samp(delta,0).pvalue)
        pvals[metric]=p
        reports[metric]={'n_seeds':n,'test_images_per_run':int(supports.iloc[0]),'mean_matched_gain':mean,'sd_run_gain':sd,'ci95_run_t':[mean-half,mean+half],'unadjusted_p':p,'unit':'independent matched training run'}
    adj=holm(pvals)
    for metric in PRIMARY:reports[metric]['holm_adjusted_p']=adj[metric]
    return {'dataset':dataset,'model':model,'baseline':baseline,'results':reports,'note':'Inference uses independently trained runs, not pooled images. Requires verified training provenance.'}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--metrics',required=True);a.add_argument('--dataset',required=True);a.add_argument('--model',default='RGR-Net');a.add_argument('--baseline',default='UDCN');a.add_argument('--out',default='run_level_report.json')
    args=a.parse_args();result=analyze(args.metrics,args.dataset,args.model,args.baseline)
    with open(args.out,'w') as f:json.dump(result,f,indent=2)
    print('Saved',args.out)
