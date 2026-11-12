#!/usr/bin/env python3
import argparse,csv,sys
from pathlib import Path

def read(p,columns):
    with open(p,newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    if rows:
        absent=set(columns)-set(rows[0])
    else:
        with open(p,newline='',encoding='utf8') as f: absent=set(columns)-set(next(csv.reader(f)))
    if absent:raise ValueError(f"{p}: missing {sorted(absent)}")
    return rows

def validate(train,val,test,rcs):
    req=['dataset','image_id','camera_id','relative_input_path']
    groups={k:read(path,req) for k,path in [('train',train),('val',val),('test',test)]}
    if any(not groups[k] for k in groups):raise ValueError('All three manifests must contain real records; empty templates cannot validate a study.')
    seen={}
    for split,rows in groups.items():
        for r in rows:
            if not all(r.get(c,'').strip() for c in req):raise ValueError(f'Missing mandatory data in {split}: {r}')
            key=(r['dataset'],r['image_id'])
            if key in seen:raise ValueError(f'Duplicate/leaking dataset image {key}: {seen[key]} and {split}')
            seen[key]=split
    rcs_rows=read(rcs,['dataset','image_id','camera_id','condition'])
    for r in rcs_rows:
        if seen.get((r['dataset'],r['image_id']))!='test':raise ValueError('RCS must reference held-out test IDs only.')
    return {k:len(rows) for k,rows in groups.items()}|{'rcs':len(rcs_rows)}

if __name__=='__main__':
    a=argparse.ArgumentParser();
    for s in ('train','val','test','rcs'):a.add_argument('--'+s,required=True)
    x=a.parse_args()
    print('Verified manifests:',validate(x.train,x.val,x.test,x.rcs))
