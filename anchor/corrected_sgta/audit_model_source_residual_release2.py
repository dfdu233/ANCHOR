"""Strict V5 cache, input-image, source, and three-arm integrity audit."""

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


def current_identity(item: dict) -> dict:
    path=Path(item["resolved_path"])
    with Image.open(path) as image: array=np.asarray(image.convert("RGB"),dtype=np.uint8)
    return {"file_sha256":sha256_file(path),"canonical_rgb_sha256":hashlib.sha256(array.tobytes()).hexdigest(),"width":int(array.shape[1]),"height":int(array.shape[0])}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",required=True,type=Path); parser.add_argument("--source-bank",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); args=parser.parse_args()
    meta=json.loads(args.cache.with_suffix(args.cache.suffix+".meta.json").read_text()); config=meta["config"]
    if meta.get("transport_cache_version")!="sgta-model-source-visual-residual-release2-v1": raise RuntimeError("wrong cache version")
    source_ok=sha256_file(args.source_bank)==config["source_bank_sha256"]; verify_source_artifacts(load_manifest(args.source_bank))
    center=Path(config["visual_centers"]); center_meta=center.with_suffix(center.suffix+".meta.json")
    center_ok=sha256_file(center)==config["visual_centers_sha256"] and sha256_file(center_meta)==config["visual_centers_meta_sha256"]
    identities=config["evaluation_input_identity"]; image_ok=all(all(current_identity(x)[k]==x[k] for k in ("file_sha256","canonical_rgb_sha256","width","height")) for x in identities)
    expected={x["qid"]:x for x in identities}; rows=list(iter_successes(args.cache,meta["fingerprint"])); seen=[]; row_checks=[]
    control=config["control_source_id"]; source=config["matched_source_id"]
    for row in rows:
        qid=str(row["qid"]); seen.append(qid); candidate=row.get("alignment_candidates",[])
        arrays_ok=False
        try:
            arrays_ok=(decode_array(row["style_language_features"]).shape[0]==3 and decode_array(row["style_visual_features"]).shape[0]==3)
        except Exception:
            arrays_ok=False
        row_checks.append(
            qid in expected and row.get("img_name")==expected.get(qid,{}).get("img_name")
            and row.get("style_roles")==["original","matched","wrong_control"]
            and len(row.get("style_names",[]))==len(row.get("style_logits",[]))==len(row.get("style_sequence_nll",[]))==len(row.get("style_decoded_text",[]))==len(row.get("style_decoded_prediction",[]))==3
            and arrays_ok and row.get("style_target_source_ids")==["original",source,source]
            and row.get("style_amplitude_source_ids")==["original",source,control]
            and len(candidate)==1 and candidate[0].get("selected") is True and candidate[0].get("safe") is True and candidate[0].get("wrong_safe") is True
            and candidate[0].get("source_id")==source and candidate[0].get("wrong_source_id")==control and float(candidate[0].get("beta",-1))==0.5
            and row.get("fallback_to_original") is False
        )
    exact_rows=len(rows)==int(config["max_samples"])==len(expected) and len(seen)==len(set(seen)) and set(seen)==set(expected)
    passed=source_ok and center_ok and image_ok and exact_rows and all(row_checks)
    summary={"n":len(rows) if passed else int(sum(row_checks)),"pass_rate":1.0 if passed else 0.0,"pixel_identity_verified_by_single_hashed_input_and_feature_only_code":passed}
    report={"version":"sgta-model-source-residual-audit-release2-v1","fingerprint":meta["fingerprint"],"rows":len(rows),"expected_rows":len(expected),"unique_qids":len(set(seen)),"checks":{"source_bank_hash":source_ok,"visual_center_and_metadata_hash":center_ok,"evaluation_image_hashes":image_ok,"exact_expected_unique_qids":exact_rows,"all_three_arm_rows_complete":all(row_checks)},"matched":summary,"wrong_control":summary,"formal_matched_structure_pass":passed}
    args.output.parent.mkdir(parents=True,exist_ok=True); temporary=args.output.with_name(args.output.name+".tmp"); temporary.write_text(json.dumps(report,indent=2)); temporary.replace(args.output); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()

