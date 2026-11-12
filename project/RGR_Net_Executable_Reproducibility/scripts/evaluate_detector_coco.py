#!/usr/bin/env python3
"""COCO AP@0.5 and class-wise AP from real frozen-detector predictions.
Does not train or download a detector. Registry CSV: method,seed,predictions_json.
Prediction JSON must contain COCO detections {image_id,category_id,bbox,score}; GT must be COCO format.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgr_repro.evidence import read_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--annotations',required=True);a=p.parse_args();root=Path(a.root)
 try:from pycocotools.coco import COCO;from pycocotools.cocoeval import COCOeval
 except ImportError:raise SystemExit('Install optional pycocotools to evaluate detector predictions.')
 reg=read_csv(root/'runs/detector_prediction_registry.csv')
 if not reg:raise SystemExit('Populate real detector_prediction_registry.csv before evaluation.')
 truth=COCO(str(root/a.annotations)); rows=[]
 for record in reg:
  pp=root/record['predictions_json']
  if not pp.is_file():raise FileNotFoundError(pp)
  predictions=json.loads(pp.read_text())
  if not predictions:raise ValueError(f'Empty predicted detections: {pp}')
  dt=truth.loadRes(predictions);ev=COCOeval(truth,dt,'bbox')
  ev.params.iouThrs=np.array([.5]);ev.params.maxDets=[1,10,100]
  ev.evaluate();ev.accumulate();ev.summarize()
  precision=ev.eval['precision'][0,:,:,:,2] # recall × class × area; maxDet=100
  # index area zero denotes 'all'; exclude missing class recall bins (-1).
  for idx,cid in enumerate(ev.params.catIds):
   vals=precision[:,idx,0];good=vals[vals>=0]
   annotations=truth.loadAnns(truth.getAnnIds(catIds=[cid]))
   gt_by={}
   for ann in annotations:
    if not ann.get('iscrowd',0):gt_by.setdefault(ann['image_id'],[]).append(ann)
   dt_by={}
   for det in predictions:
    if det['category_id']==cid and det['score']>=0.25:dt_by.setdefault(det['image_id'],[]).append(det)
   tp=fp=fn=0
   for image_id in set(gt_by)|set(dt_by):
    gt=gt_by.get(image_id,[]); dets=sorted(dt_by.get(image_id,[]),key=lambda x:-x['score'])
    matched=set()
    for det in dets:
     bx,by,bw,bh=det['bbox'];area_b=max(0,bw)*max(0,bh)
     best_i=-1;best_iou=0.
     for j,ann in enumerate(gt):
      if j in matched:continue
      ax,ay,aw,ah=ann['bbox']; interw=max(0.,min(ax+aw,bx+bw)-max(ax,bx));interh=max(0.,min(ay+ah,by+bh)-max(ay,by))
      intersection=interw*interh;union=area_b+max(0,aw)*max(0,ah)-intersection
      iou=intersection/union if union>0 else 0.
      if iou>best_iou:best_iou=iou;best_i=j
     if best_iou>=.5:tp+=1;matched.add(best_i)
     else:fp+=1
    fn+=len(gt)-len(matched)
   precision_at025=tp/(tp+fp) if tp+fp else 0.;recall_at025=tp/(tp+fn) if tp+fn else 0.
   f1=2*precision_at025*recall_at025/(precision_at025+recall_at025) if precision_at025+recall_at025 else 0.
   rows.append(dict(method=record['method'],seed=record['seed'],category_id=cid,category_name=truth.cats[cid]['name'],n_ground_truth=sum(len(v) for v in gt_by.values()),ap50=float(good.mean()) if len(good) else '',precision_conf025=precision_at025,recall_conf025=recall_at025,f1_conf025=f1,annotation_file=a.annotations,predictions_file=record['predictions_json']))
 out=root/'results/detector_classwise_ap50_measured.csv';out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('Wrote',out)
if __name__=='__main__':main()
