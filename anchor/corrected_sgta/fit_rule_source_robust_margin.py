#!/usr/bin/env python3
"""Fit one worst-source robust complete-sequence margin threshold."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from corrected_sgta.rule_source_preference import file_sha256,stable_json_sha256
VERSION="rule-source-robust-margin-v1"

def rows_from_result(path:Path):
 p=json.loads(path.read_text()); return p["records"]
def fit(groups):
 pts={d:[(r["sequence_log_probabilities"]["identity"]["Yes."]-r["sequence_log_probabilities"]["identity"]["No."],r["ground_truth"]=="Yes.") for r in rows] for d,rows in groups.items()}
 values=sorted({m for rows in pts.values() for m,y in rows})
 candidates=[values[0]-1.0]+[(a+b)/2 for a,b in zip(values,values[1:])]+[values[-1]+1.0]
 ranked=[]
 for t in candidates:
  per={d:sum((m>=t)==y for m,y in rows)/len(rows) for d,rows in pts.items()}
  allrows=[x for rows in pts.values() for x in rows]
  micro=sum((m>=t)==y for m,y in allrows)/len(allrows)
  ranked.append((min(per.values()),sum(per.values())/len(per),micro,-abs(t),t,per))
 best=max(ranked)
 return {"threshold":best[4],"selection":"lexicographic max(min-domain accuracy, macro accuracy, micro accuracy, -abs(threshold))","per_domain_accuracy":best[5],"minimum_domain_accuracy":best[0],"macro_accuracy":best[1],"micro_accuracy":best[2]}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source-dev-result',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 fitted=fit(rows_from_result(a.source_dev_result))
 payload={"version":VERSION,"source_dev_result":str(a.source_dev_result),"source_dev_sha256":file_sha256(a.source_dev_result),"target_labels_used":False,"fit":fitted}
 payload["fingerprint"]=stable_json_sha256(payload); a.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
