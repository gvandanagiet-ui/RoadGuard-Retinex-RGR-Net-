#!/usr/bin/env python3
import csv,sys
from itertools import product
from pathlib import Path
root=Path(__file__).resolve().parents[1]
schedule=[]
for i,(g,e,s,c) in enumerate(product((0,1),repeat=4),1):
 for seed in range(5):schedule.append([f'factorial_{i:02d}_seed{seed}','factorial','rgr-net',f'F{i:02d}',g,e,s,c,seed,'(expected)','NOT EXECUTED'])
for mod in ['without_Q','shared_gate','without_noise_gate']:
 for seed in range(5):schedule.append([f'control_{mod}_seed{seed}','targeted_control','rgr-net',mod,'','','','',seed,'(expected)','NOT EXECUTED'])
for mod in ['rgr-net','udcn']:
 for seed in range(5):schedule.append([f'baseline_{mod}_seed{seed}','matched_baseline',mod,'','','','','',seed,'(expected)','NOT EXECUTED'])
with (root/'configs'/'training_schedule_105_expected.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['job_id','job_family','model','configuration','G','E','S','C','seed','evidence_class','execution_status']);w.writerows(schedule)
assert len(schedule)==105
print('Generated 105 planned jobs: 80 factorial + 15 controls + 10 matched-model runs. This is a schedule, not training output.')
