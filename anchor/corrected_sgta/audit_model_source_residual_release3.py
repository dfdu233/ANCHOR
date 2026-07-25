"""Numerically and structurally strict V5 cache audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from corrected_sgta.cache import decode_array, iter_successes
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def current_identity(item: dict) -> dict:
    path=Path(item["resolved_path"])
    with Image.open(path) as image: array=np.asarray(image.convert("RGB"),dtype=np.uint8)
    return {"file_sha256":sha256_file(path),"canonical_rgb_sha256":hashlib.sha256(array.tobytes()).hexdigest(),"width":int(array.shape[1]),"height":int(array.shape[0])}


def matrix_payload(value, arms: int, columns: int | None = None) -> bool:
    try: array=np.asarray(value,dtype=np.float64)
    except Exception: return False
    return array.ndim==2 and array.shape[0]==arms and array.shape[1]>0 and (columns is None or array.shape[1]==columns) and bool(np.isfinite(array).all())


def candidate_valid(item: dict, source: str, control: str, identity_structure: dict) -> bool:
    numeric=("visual_distance_before","visual_distance_after","absolute_closure","relative_closure","wrong_distance_after","wrong_relative_closure")
    return (
        item.get("selected") is True and item.get("safe") is True and item.get("wrong_safe") is True
        and item.get("source_id")==source and item.get("wrong_source_id")==control
        and finite_number(item.get("beta")) and float(item["beta"])==0.5
        and all(finite_number(item.get(key)) for key in numeric)
        and item.get("structure")==identity_structure and item.get("wrong_structure")==identity_structure
    )


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",required=True,type=Path); parser.add_argument("--source-bank",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); args=parser.parse_args()
    meta=json.loads(args.cache.with_suffix(args.cache.suffix+".meta.json").read_text()); config=meta["config"]
    if meta.get("transport_cache_version")!="sgta-model-source-visual-residual-release3-v1": raise RuntimeError("wrong cache version")
    source_ok=sha256_file(args.source_bank)==config["source_bank_sha256"]; verify_source_artifacts(load_manifest(args.source_bank))
    center=Path(config["visual_centers"]); center_meta=center.with_suffix(center.suffix+".meta.json")
    center_ok=sha256_file(center)==config["visual_centers_sha256"] and sha256_file(center_meta)==config["visual_centers_meta_sha256"]
    identities=config["evaluation_input_identity"]; image_ok=all(all(current_identity(x)[k]==x[k] for k in ("file_sha256","canonical_rgb_sha256","width","height")) for x in identities)
    expected={x["qid"]:x for x in identities}; rows=list(iter_successes(args.cache,meta["fingerprint"])); seen=[]; row_checks=[]
    control=config["control_source_id"]; source=config["matched_source_id"]
    identity_structure={"psnr":None,"edge_correlation":1.0,"ssim":1.0,"central_local_contrast_correlation":1.0,"central_gradient_magnitude_ratio":1.0,"scope":"processor input pixels are identical across roles"}
    expected_names=["original",f"matched_{source}_b0.5",f"control_{control}_b0.5"]
    for row in rows:
        qid=str(row["qid"]); seen.append(qid); labels=row.get("labels"); candidates=row.get("alignment_candidates")
        labels_ok=isinstance(labels,list) and len(labels)>1 and len({str(x) for x in labels})==len(labels)
        class_count=len(labels) if labels_ok else 0
        logits_ok=matrix_payload(row.get("style_logits"),3,class_count)
        nll_ok=matrix_payload(row.get("style_sequence_nll"),3,class_count)
        try:
            language=decode_array(row["style_language_features"]); visual=decode_array(row["style_visual_features"])
            features_ok=language.ndim==2 and visual.ndim==2 and language.shape[0]==visual.shape[0]==3 and language.shape[1]>0 and visual.shape[1]>0 and np.isfinite(language).all() and np.isfinite(visual).all()
        except Exception: features_ok=False
        text=row.get("style_decoded_text"); predictions=row.get("style_decoded_prediction")
        decode_ok=isinstance(text,list) and len(text)==3 and all(isinstance(x,str) for x in text) and isinstance(predictions,list) and len(predictions)==3 and all(x is None or (isinstance(x,int) and not isinstance(x,bool) and 0<=x<class_count) for x in predictions)
        gt=row.get("gt_index"); candidate_ok=isinstance(candidates,list) and len(candidates)==1 and candidate_valid(candidates[0],source,control,identity_structure)
        row_checks.append(
            qid in expected and row.get("img_name")==expected.get(qid,{}).get("img_name") and labels_ok and isinstance(gt,int) and not isinstance(gt,bool) and 0<=gt<class_count
            and row.get("style_names")==expected_names and row.get("style_roles")==["original","matched","wrong_control"]
            and row.get("style_target_source_ids")==["original",source,source] and row.get("style_amplitude_source_ids")==["original",source,control]
            and logits_ok and nll_ok and features_ok and decode_ok and candidate_ok and row.get("fallback_to_original") is False
        )
    exact_rows=len(rows)==int(config["max_samples"])==len(expected) and len(seen)==len(set(seen)) and set(seen)==set(expected)
    passed=source_ok and center_ok and image_ok and exact_rows and all(row_checks)
    summary={"n":len(rows) if passed else int(sum(row_checks)),"pass_rate":1.0 if passed else 0.0,"pixel_identity_verified_by_single_hashed_input_and_feature_only_code":passed}
    report={"version":"sgta-model-source-residual-audit-release3-v1","fingerprint":meta["fingerprint"],"rows":len(rows),"expected_rows":len(expected),"unique_qids":len(set(seen)),"checks":{"source_bank_hash":source_ok,"visual_center_and_metadata_hash":center_ok,"evaluation_image_hashes":image_ok,"exact_expected_unique_qids":exact_rows,"all_numeric_and_three_arm_payloads_valid":all(row_checks)},"matched":summary,"wrong_control":summary,"formal_matched_structure_pass":passed}
    args.output.parent.mkdir(parents=True,exist_ok=True); temporary=args.output.with_name(args.output.name+".tmp"); temporary.write_text(json.dumps(report,indent=2)); temporary.replace(args.output); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()

