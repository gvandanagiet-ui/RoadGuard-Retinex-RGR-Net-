import unittest,tempfile,subprocess,sys,csv,json
from pathlib import Path
import torch
from model import ReferenceRGR,condition_maps

class ReferencePipelineTests(unittest.TestCase):
 def test_cues_shapes_bounds(self):
  x=torch.rand(2,3,32,32)
  m=condition_maps(x)
  self.assertEqual(set(m),set('EGNSDQ'))
  for v in m.values():
   self.assertEqual(tuple(v.shape),(2,1,32,32))
   self.assertTrue(bool(torch.isfinite(v).all()))
   self.assertTrue(bool(((v>=0)&(v<=1)).all()))
 def test_variants_backward(self):
  for dual in [True,False]:
   for use_q in [True,False]:
    model=ReferenceRGR(width=8,dual_gate=dual,use_q=use_q)
    x=torch.rand(1,3,32,32)
    y,_=model(x)
    y.mean().backward()
    self.assertTrue(bool(torch.isfinite(y).all()))
 def test_registry_95(self):
  with tempfile.TemporaryDirectory() as tmp:
   subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_95_job_registry.py')),'--out',tmp+'/jobs.csv'],check=True,capture_output=True)
   with open(tmp+'/jobs.csv',newline='') as fh:
    rows=list(csv.DictReader(fh))
   self.assertEqual(len(rows),95)
   self.assertEqual(sum(x['type']=='factorial' for x in rows),80)
   self.assertTrue(all(x['status']=='NOT_RUN' for x in rows))
if __name__=='__main__':unittest.main()
