"""Frozen contracts for the single-change exact-pair-only source pilot."""
from __future__ import annotations
from typing import Any,Mapping
import torch
from corrected_sgta.rule_source_exact_pair import (
    DEV_IMAGES_TOTAL,GRADIENT_CLIP,LEARNING_RATE,MAX_RELATIVE_UPDATE,
    PAIR_MANIFEST_VERSION,RANK,SEED,SOURCE_DOMAINS,STEPS,
    canonical_label,canonical_question,pair_manifest_identity,
    reference_relative_pair_loss,study_id,
)
from corrected_sgta.rule_source_preference import stable_json_sha256
VERSION='rule-source-exact-pair-only-v1'
DEV_EVAL_VERSION='rule-source-exact-pair-only-dev-eval-v1'

def common_bias_gradient(loss:torch.Tensor,positive_margin:torch.Tensor,negative_margin:torch.Tensor)->torch.Tensor:
    gp,gn=torch.autograd.grad(loss,(positive_margin,negative_margin),retain_graph=True)
    return gp+gn

def experiment_fingerprint(*,manifest_contract:Mapping[str,Any],pair_manifest:Mapping[str,Any],config:Mapping[str,Any],code_sha256:Mapping[str,str])->tuple[str,dict[str,Any]]:
    payload={'version':VERSION,'manifest_contract':dict(manifest_contract),'pair_manifest':dict(pair_manifest),'config':dict(config),'code_sha256':dict(sorted(code_sha256.items()))}
    return stable_json_sha256(payload),payload

def dev_gate(summary:Mapping[str,Any],margin_diagnostic:Mapping[str,Any])->dict[str,Any]:
    result=summary.get('source_pair_only',{}); micro=result.get('micro',{}); domains=result.get('per_domain',{})
    rescues=int(micro.get('rescues',0)); harms=int(micro.get('harms',0)); pos=int(margin_diagnostic.get('positive_delta_count',0)); neg=int(margin_diagnostic.get('negative_delta_count',0)); n=int(margin_diagnostic.get('n',0)); positive_rate=pos/n if n else 1.0; negative_rate=neg/n if n else 1.0; centered_rms=float(margin_diagnostic.get('centered_rms',0.0)); common_mean=float(margin_diagnostic.get('mean_delta',float('inf'))); common_ratio=abs(common_mean)/centered_rms if centered_rms>0 else float('inf')
    checks={
      'complete_source_dev':{'value':int(micro.get('n',-1)),'required':DEV_IMAGES_TOTAL,'passed':int(micro.get('n',-1))==DEV_IMAGES_TOTAL},
      'net_rescues':{'value':rescues-harms,'required':f'>=3/{DEV_IMAGES_TOTAL}','passed':rescues-harms>=3},
      'nondeclining_domains':{'value':sum(float(v.get('delta_pp',-float('inf')))>=0 for v in domains.values()),'required':'>=2/3','passed':len(domains)==SOURCE_DOMAINS and sum(float(v.get('delta_pp',-float('inf')))>=0 for v in domains.values())>=2},
      'harms_not_greater_than_rescues':{'value':{'rescues':rescues,'harms':harms},'required':'harms<=rescues','passed':harms<=rescues},
      'both_margin_delta_signs':{'value':{'positive':pos,'negative':neg},'required':'positive>0 and negative>0','passed':pos>0 and neg>0},
      'symmetric_sign_dominance':{'value':{'positive_rate':positive_rate,'negative_rate':negative_rate},'required':'max(positive_rate,negative_rate)<=0.90','passed':max(positive_rate,negative_rate)<=0.90},
      'common_shift_to_dispersion':{'value':{'mean':common_mean,'centered_rms':centered_rms,'absolute_ratio':common_ratio},'required':'centered_rms>0 and |mean|/centered_rms<=1','passed':centered_rms>0 and common_ratio<=1.0},
    }
    passed=all(v['passed'] for v in checks.values())
    return {'status':'passed' if passed else 'failed','target_evaluation_allowed':passed,'primary_variant':'source_pair_only','checks':checks,'selection_note':'Single frozen pair-only source-dev gate; failure terminates without tuning or target access.'}
