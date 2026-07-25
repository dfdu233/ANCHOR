#!/usr/bin/env python3
"""Fit one pooled-optimal threshold subject to source-domain non-degradation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from corrected_sgta.rule_source_preference import file_sha256,stable_json_sha256
VERSION='rule-source-pareto-margin-v1'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source-dev-result',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 groups=json.loads(a.source_dev_result.read_text())['records']; pts={d:[(r['sequence_log_probabilities']['identity']['Yes.']-r['sequence_log_probabilities']['identity']['No.'],r['ground_truth']=='Yes.') for r in rows] for d,rows in groups.items()}; vals=sorted({m for rows in pts.values() for m,y in rows}); candidates=[vals[0]-1]+[(x+y)/2 for x,y in zip(vals,vals[1:])]+[vals[-1]+1]; base={d:sum((m>=0)==y for m,y in rows)/len(rows) for d,rows in pts.items()}; ranked=[]
 for t in candidates:
  per={d:sum((m>=t)==y for m,y in rows)/len(rows) for d,rows in pts.items()}
  if all(per[d]+1e-12>=base[d] for d in per):
   allrows=sum(pts.values(),[]); micro=sum((m>=t)==y for m,y in allrows)/len(allrows); ranked.append((micro,min(per[d]-base[d] for d in per),sum(per.values())/len(per),-abs(t),t,per))
 best=max(ranked); fit={'threshold':best[4],'selection':'maximize pooled source accuracy subject to non-degradation on every source domain; ties by minimum-domain gain, macro accuracy, -abs(threshold)','identity_per_domain_accuracy':base,'per_domain_accuracy':best[5],'minimum_domain_delta_pp':100*best[1],'macro_accuracy':best[2],'micro_accuracy':best[0]}; payload={'version':VERSION,'source_dev_result':str(a.source_dev_result),'source_dev_sha256':file_sha256(a.source_dev_result),'target_labels_used':False,'fit':fit}; payload['fingerprint']=stable_json_sha256(payload); a.output.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
