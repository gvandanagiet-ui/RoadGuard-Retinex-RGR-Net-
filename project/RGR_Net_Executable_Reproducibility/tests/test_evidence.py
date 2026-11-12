"""All numeric examples in this file are software unit-test fixtures, NOT RGR-Net observations."""
import csv,hashlib,json,math,sys
from itertools import product
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgr_repro.evidence import *

def putcsv(p,heads,records):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(heads);w.writerows(records)

def init_root(tmp):
 (tmp/'checkpoints').mkdir(exist_ok=True)
 cp=tmp/'checkpoints'/'unit_test_fixture_not_a_checkpoint.txt';cp.write_text('SOFTWARE TEST FIXTURE; NOT REAL TRAINED MODEL')
 sha=sha256(cp)
 for name,split,camera in [('train','train','camA'),('val','val','camB'),('test','test','camC')]:
  (tmp/'images').mkdir(exist_ok=True)
  (tmp/'images'/f'{name}.png').write_bytes(b'TEST ONLY NOT AN IMAGE')
 putcsv(tmp/'manifests'/'image_ids.csv',['image_id','dataset','source_camera','split','image_path','reference_path','annotation_path','rcs','condition'],[
 [name,'toy',camera,split,f'images/{name}.png','','','1' if split=='test' else '0','']
 for name,split,camera in [('train','train','camA'),('val','val','camB'),('test','test','camC')]])
 return 'checkpoints/'+cp.name,sha,sha256(tmp/'manifests'/'image_ids.csv')

def test_complete_manifest_source_disjoint(tmp_path):
 init_root(tmp_path);r=validate_manifest(tmp_path);assert r['pass'],r['issues'];assert r['counts']['test']==1

def test_manifest_reject_overlap(tmp_path):
 init_root(tmp_path)
 with (tmp_path/'manifests'/'image_ids.csv').open('a') as f:f.write('test,toy,camC,train,images/train.png,,,0,\n')
 assert not validate_manifest(tmp_path)['pass']

def test_matched_statistics_holm_from_genuine_file_contract(tmp_path):
 cp,sha,msha=init_root(tmp_path)
 records=[]
 for m,base in [('rgr-net',31.42),('udcn',25.21)]:
  for s in range(5):
   jitter=[-.11,.12,.05,-.06,0.][s]
   record=[m,s,msha,cp,sha,base+jitter*(1 if m=='rgr-net' else .5),.90+jitter*.01 if m=='rgr-net' else .74+jitter*.005,.12+jitter*.002 if m=='rgr-net' else .21+jitter*.004,1.76+jitter*.05 if m=='rgr-net' else 4.62+jitter*.07,82+jitter if m=='rgr-net' else 72+jitter*.8]
   records.append(record)
 putcsv(tmp_path/'runs'/'matched_baseline_seed_metrics.csv',['method','seed','manifest_sha256','checkpoint_path','checkpoint_sha256',*METRICS],records)
 r=audit_matched(tmp_path,msha);assert r['pass'],r['issues']
 assert pytest.approx(6.21,abs=1e-4)==r['analysis']['psnr']['difference']['mean']
 assert 0<=r['analysis']['psnr']['difference']['p_holm']<=1

def test_missing_runs_never_imputed(tmp_path):
 cp,sha,msha=init_root(tmp_path)
 assert not audit_matched(tmp_path,msha)['pass']
 assert not audit_factorial(tmp_path,msha)['pass']
 assert readiness(tmp_path)['status']=='ORIGINAL_EVIDENCE_REQUIRED'

def test_factorial_four_factors_and_interaction(tmp_path):
 cp,sha,msha=init_root(tmp_path);entries=[]
 for i,(G,E,S,C) in enumerate(product((0,1),repeat=4),1):
  for seed in range(5):
   score=20+.55*G+.45*E+.35*S+.25*C+.4*(1 if G==E else 0)+.01*seed
   entries.append([f'F{i:02d}',G,E,S,C,seed,msha,cp,sha,score,.87+.01*G+.005*E+.0002*seed])
 putcsv(tmp_path/'runs'/'factorial_seed_metrics.csv',['cell','G','E','S','C','seed','manifest_sha256','checkpoint_path','checkpoint_sha256','psnr','ssim'],entries)
 r=audit_factorial(tmp_path,msha)
 assert r['pass'],r['issues']
 assert r['analysis']['factorial_effects']['psnr']['G']['mean']==pytest.approx(.55)
 assert r['analysis']['factorial_effects']['psnr']['G×E']['mean']==pytest.approx(.4)
 assert r['analysis']['cell_summaries']['F16']['psnr']['n']==5

def test_zero_sd_does_not_yield_fake_p():
 r=paired_analysis([5]*5,[3]*5)
 assert r['p_raw'] is None

def test_holm_monotonic():
 r=holm({'a':.002,'b':.01,'c':.08})
 assert r['a']<=r['b']<=r['c']

def test_checkpoint_hash_tamper(tmp_path):
 cp,sha,msha=init_root(tmp_path)
 assert valid_artifact(tmp_path,cp,sha)
 (tmp_path/cp).write_text('tampered')
 assert not valid_artifact(tmp_path,cp,sha)

def test_five_seed_ci_uses_t4():
 r=five_seed_summary([1,2,3,4,5]);assert r['mean']==3
 assert r['sample_sd']==pytest.approx(math.sqrt(2.5))
 assert r['ci95_high']>r['mean']
 with pytest.raises(ValueError):five_seed_summary([1,2,3])
