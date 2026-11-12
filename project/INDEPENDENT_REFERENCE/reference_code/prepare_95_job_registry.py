"""Generate ONLY a proposed job registry for a 16 x 5 + 3 x 5 design.
No experiment is run by this script; no result values are inserted.
'gate_mode' distinguishes independent vs shared gates in this reference code.
"""
import csv,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
rows=[]
for dual in [0,1]:
 for target in [0,1]:
  for structure in [0,1]:
   for consistency in [0,1]:
    tag=f'F_dual{dual}_target{target}_structure{structure}_consistency{consistency}'
    flags=[]
    if not dual:flags.append('--shared-gates')
    if not target:flags.append('--no-spatial-target')
    if not structure:flags.append('--no-structure-loss')
    if not consistency:flags.append('--no-consistency')
    for seed in range(5):rows.append(dict(job_id=f'{tag}_seed{seed}',type='factorial',seed=seed,gate_mode='independent' if dual else 'shared',spatial_target=bool(target),structure_loss=bool(structure),consistency=bool(consistency),flags=' '.join(flags),status='NOT_RUN',experiment_class='reference scaffold; not manuscript model',result_file=''))
for control,flag in [('without_q','--no-q'),('without_illum_gate','--no-illum-gate'),('without_noise_gate','--no-noise-gate')]:
 for seed in range(5):rows.append(dict(job_id=f'C_{control}_seed{seed}',type='targeted_control',seed=seed,gate_mode='independent',spatial_target=True,structure_loss=True,consistency=True,flags=flag,status='NOT_RUN',experiment_class='reference scaffold; not manuscript model',result_file=''))
assert len(rows)==95 and len({r['job_id'] for r in rows})==95
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f: wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('Wrote 95 planned independent-reference jobs; 0 executed:',out)
