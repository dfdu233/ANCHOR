#!/usr/bin/env python3
"""Test source-prior-free robust intervals for small-expert/VLM collaboration.

Each source expert is converted from posterior log-odds to approximate visual
log likelihood ratio by subtracting the source pathology prevalence log-odds.
The convex hull across source domains is used as a distributionally robust
evidence interval.  The VLM decision changes only when the entire interval has
one sign.  No target labels or test-set tuning are used by the intervention.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from anchor.corrected_sgta.screen_external_visual_increment_v1 import load_claims, sha256_file
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import XRV_LABELS


# Restrict the preregistered primary screen to one-to-one label mappings.  A
# max over Nodule/Mass followed by subtraction of their union prevalence is
# not a valid posterior-to-evidence conversion, so nodule_mass is deliberately
# excluded rather than hidden behind an optimistic aggregation rule.
FINDINGS = ("cardiomegaly", "pleural_effusion", "pleural_thickening", "pulmonary_fibrosis")
TARGETS = {
    "nih": {
        "cardiomegaly": ("Cardiomegaly",), "nodule_mass": ("Nodule", "Mass"),
        "pleural_effusion": ("Effusion",), "pleural_thickening": ("Pleural_Thickening",),
        "pulmonary_fibrosis": ("Fibrosis",),
    },
    "pc": {
        "cardiomegaly": ("Cardiomegaly",), "nodule_mass": ("Nodule", "Mass"),
        "pleural_effusion": ("Effusion",), "pleural_thickening": ("Pleural_Thickening",),
        "pulmonary_fibrosis": ("Fibrosis",),
    },
    "chex": {
        "cardiomegaly": ("Cardiomegaly",), "nodule_mass": ("Lung Lesion",),
        "pleural_effusion": ("Effusion",),
    },
}
OP_THRESHOLDS = {
    "nih": [0.039117552, 0.0034529066, 0.11396341, 0.0057298196, 0.00045666535, 0.0018880932, 0.012037827, 0.038744126, 0.0037213727, 0.014730946, 0.016149804, 0.054241467, 0.037198864, 0.0004403434, np.nan, np.nan, np.nan, np.nan],
    "pc": [0.031012505, 0.013347598, 0.081435576, 0.001262615, 0.002587246, 0.0035944257, 0.0023071, 0.055412333, 0.044385884, 0.042766232, 0.043258056, 0.037629247, 0.005658899, 0.0091741895, np.nan, 0.026507627, np.nan, np.nan],
    "chex": [0.1988969, 0.05710573, np.nan, 0.0531293, 0.1435217, np.nan, np.nan, 0.27212676, 0.07749717, np.nan, 0.19712369, np.nan, np.nan, np.nan, 0.09932402, 0.09273402, 0.3270967, 0.10888247],
}


def logit(value: float) -> float:
    value = min(max(value, 1e-6), 1 - 1e-6)
    return float(np.log(value / (1 - value)))


def load_experts(path: Path):
    payload = np.load(path)
    if payload["labels"].astype(str).tolist() != list(XRV_LABELS):
        raise ValueError("label order drift")
    image_ids = payload["image_ids"].astype(str).tolist()
    domains = payload["domains"].astype(str).tolist()
    logits = payload["logits"].astype(np.float64)
    return domains, {image_id: logits[:, i, :] for i, image_id in enumerate(image_ids)}


def source_score(matrix: np.ndarray, domain_index: int, labels: tuple[str, ...]) -> float:
    indices = [XRV_LABELS.index(label) for label in labels]
    return float(np.max(matrix[domain_index, indices]))


def fit_balanced_likelihood_ratios(rows, domains, experts, findings):
    """Fit one-dimensional score calibrators on a balanced development split.

    Because each finding has equal positive/negative sampling, the fitted
    posterior log-odds estimates a score likelihood ratio rather than a
    prevalence-contaminated posterior.  This is a calibration probe, not
    end-to-end model training.
    """
    fitted = {}
    for domain_index, domain in enumerate(domains):
        for finding in findings:
            labels = TARGETS.get(domain, {}).get(finding)
            if not labels:
                continue
            selected = [row for row in rows if row["finding"] == finding]
            score = np.asarray(
                [source_score(experts[row["image_id"]], domain_index, labels) for row in selected],
                dtype=np.float64,
            )[:, None]
            target = np.asarray([row["label"] for row in selected], dtype=np.int64)
            if len(np.unique(target)) != 2:
                raise ValueError(f"Calibration split lacks both labels for {domain}/{finding}")
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(score, target)
            fitted[(domain, finding)] = (float(model.coef_[0, 0]), float(model.intercept_[0]))
    return fitted


def intervals(row, domains, experts, priors, calibrators, mode: str) -> tuple[float, float]:
    evidence = []
    matrix = experts[row["image_id"]]
    for d, domain in enumerate(domains):
        labels = TARGETS.get(domain, {}).get(row["finding"])
        if not labels:
            continue
        value = source_score(matrix, d, labels)
        if mode == "operating":
            thresholds = [OP_THRESHOLDS[domain][XRV_LABELS.index(label)] for label in labels]
            value -= logit(max(thresholds))
        elif mode == "source_prior":
            prevalence = priors["domains"][domain]["targets"][row["finding"]]["prevalence"]
            if prevalence is None:
                continue
            value -= logit(float(prevalence))
        elif mode == "balanced_lr":
            slope, intercept = calibrators[(domain, row["finding"])]
            value = slope * value + intercept
        elif mode != "raw":
            raise ValueError(mode)
        evidence.append(value)
    if len(evidence) < 2:
        raise ValueError(f"Need >=2 experts for {row['finding']}; got {len(evidence)}")
    return float(min(evidence)), float(max(evidence))


def predictions(rows, domains, experts, priors, calibrators, mode: str) -> np.ndarray:
    output = []
    for row in rows:
        original = row["margin"] > 0
        lower, upper = intervals(row, domains, experts, priors, calibrators, mode)
        if upper < 0:
            output.append(False)
        elif lower > 0:
            output.append(True)
        else:
            output.append(original)
    return np.asarray(output, dtype=bool)


def metrics(rows, predicted: np.ndarray) -> dict[str, Any]:
    label = np.asarray([row["label"] for row in rows], dtype=bool)
    tp, tn = int(np.sum(predicted & label)), int(np.sum(~predicted & ~label))
    fp, fn = int(np.sum(predicted & ~label)), int(np.sum(~predicted & label))
    return {
        "n": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        "fpr": fp / max(1, fp + tn), "recall": tp / max(1, tp + fn),
    }


def bootstrap(rows, base, candidate, draws: int, seed: int):
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[row["image_id"]].append(i)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    acc, bacc, fpr, recall = [], [], [], []
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        idx = np.asarray([i for image_id in sampled for i in groups[image_id]])
        sampled_rows = [rows[i] for i in idx]
        a, b = metrics(sampled_rows, base[idx]), metrics(sampled_rows, candidate[idx])
        acc.append(b["accuracy"] - a["accuracy"])
        bacc.append(b["balanced_accuracy"] - a["balanced_accuracy"])
        fpr.append(a["fpr"] - b["fpr"])
        recall.append(b["recall"] - a["recall"])
    def summary(values):
        x=np.asarray(values); return {"mean":float(x.mean()),"ci95":[float(np.quantile(x,.025)),float(np.quantile(x,.975))]}
    return {"accuracy_delta":summary(acc),"balanced_accuracy_delta":summary(bacc),"fpr_reduction":summary(fpr),"recall_delta":summary(recall)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--huatuo-confirmation",type=Path,required=True)
    parser.add_argument("--hulu-confirmation",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--expert-logits",type=Path,required=True)
    parser.add_argument("--source-priors",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap-draws",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--findings",nargs="+",choices=FINDINGS,default=list(FINDINGS))
    args=parser.parse_args()
    findings=tuple(args.findings)
    domains, experts=load_experts(args.expert_logits)
    priors=json.loads(args.source_priors.read_text())
    calibration_rows=[row for row in load_claims(args.calibration,"development","calibration") if row["finding"] in findings]
    calibrators=fit_balanced_likelihood_ratios(calibration_rows,domains,experts,findings)
    analyses={}
    for model,path in (("huatuo",args.huatuo_confirmation),("hulu",args.hulu_confirmation)):
        rows=[row for row in load_claims(path,"confirmation",model) if row["finding"] in findings]
        base=np.asarray([row["margin"]>0 for row in rows],dtype=bool)
        methods={"vlm":base}
        for mode in ("raw","operating","source_prior","balanced_lr"):
            methods[mode]=predictions(rows,domains,experts,priors,calibrators,mode)
        analyses[model]={
            "metrics":{name:metrics(rows,pred) for name,pred in methods.items()},
            "decision_changes":{
                name:{
                    "n":int(np.sum(pred != base)),
                    "negative_to_positive":int(np.sum(pred & ~base)),
                    "positive_to_negative":int(np.sum(~pred & base)),
                }
                for name,pred in methods.items() if name!="vlm"
            },
            "bootstrap_vs_vlm":{
                name:bootstrap(rows,base,pred,args.bootstrap_draws,args.seed)
                for name,pred in methods.items() if name!="vlm"
            },
            "bootstrap_balanced_lr_vs_controls":{
                name:bootstrap(rows,pred,methods["balanced_lr"],args.bootstrap_draws,args.seed+1)
                for name,pred in methods.items() if name in ("raw","operating","source_prior")
            },
        }
    passes=[]
    for analysis in analyses.values():
        p=analysis["metrics"]["balanced_lr"]; b=analysis["metrics"]["vlm"]; ci=analysis["bootstrap_vs_vlm"]["balanced_lr"]
        controls=analysis["bootstrap_balanced_lr_vs_controls"]
        passes.append(
            p["fpr"] <= .8*b["fpr"]
            and p["recall"] >= b["recall"]-.01
            and ci["balanced_accuracy_delta"]["ci95"][0]>0
            # The proposed source-prior removal must contribute more than the
            # already-known raw expert consensus and official operating-point
            # centering; otherwise this is just expert fusion under a new name.
            and all(value["balanced_accuracy_delta"]["ci95"][0]>0 for value in controls.values())
        )
    result={
        "status":"complete",
        "decision":"PASS_L0" if all(passes) else "NO_GO_L0",
        "command":shlex.join(sys.argv),
        "seed":args.seed,
        "bootstrap_draws":args.bootstrap_draws,
        "input_sha256":{
            "calibration":sha256_file(args.calibration),
            "huatuo_confirmation":sha256_file(args.huatuo_confirmation),
            "hulu_confirmation":sha256_file(args.hulu_confirmation),
            "expert_logits":sha256_file(args.expert_logits),
            "source_priors":sha256_file(args.source_priors),
        },
        "domains":domains,
        "findings":findings,
        "balanced_lr_calibrators":{
            f"{domain}/{finding}":{"slope":value[0],"intercept":value[1]}
            for (domain,finding),value in calibrators.items()
        },
        "preregistered_gate":{
            "models":"Huatuo and Hulu must both pass",
            "false_positive_reduction":"at least 20% relative",
            "recall_harm":"at most 1 percentage point",
            "paired_bacc_vs_vlm":"95% image-bootstrap CI lower bound > 0",
            "paired_bacc_vs_raw_operating_and_source_prior":"all three 95% image-bootstrap CI lower bounds > 0",
        },
        "analyses":analyses,
        "boundary":"The primary balanced-LR arm uses a small, finding-balanced development calibration split and never test labels. Source-logit minus source-prevalence log-odds remains only an approximate control because the discriminative experts are not guaranteed calibrated. One-to-many label mappings are excluded.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
