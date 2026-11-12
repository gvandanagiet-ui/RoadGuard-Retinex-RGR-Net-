#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgr_repro.evidence import readiness
p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--strict',action='store_true');p.add_argument('--out',default='results/evidence_readiness.json');a=p.parse_args()
r=readiness(a.root)
out=Path(a.root)/a.out;out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps({'status':r['status'],'checks':{k:('PASS' if v['pass'] else 'REQUIRES EVIDENCE') for k,v in r['checks'].items()},'report':str(out)},indent=2))
if a.strict and r['status']!='COMPLETE_ORIGINAL_EVIDENCE':sys.exit(2)
