#!/usr/bin/env python3
"""Build a CPU-only controlled source x polarity binding substrate.

This is explicitly not natural RAG.  VinDr query images are frozen by image
hash while reading only image and reader IDs.  Their finding votes are loaded
only in a second pass for post-selection ecological evaluation.  Controlled
state tags form a complete CURRENT/OTHER x present/absent factorial.  CURRENT
is an experimental binding symbol and is never interpreted as image truth.

Source-state provenance comes from strict, simple MIMIC train-only sentences.
The report text is never injected: it only certifies that each canonical state
has a real, cross-dataset source example with an auditable patient ID.  This
gives donor/query patient and image separation without relying on VinDr's
removed patient identifiers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from anchor.corrected_sgta.build_matched_retrieval_polarity_canary_v1 import (
    sentence_assertions,
)
from anchor.corrected_sgta.build_same_report_polarity_twin_pilot_v1 import (
    ATTRIBUTE,
    COMPOUND,
    UNCERTAINTY,
    finding_mentions,
    other_findings,
    sentence_spans,
    sha256_file,
    stable_hash,
)
from anchor.corrected_sgta.run_target_blind_canary_v1 import (
    load_target_blind_manifest,
    preflight_inputs,
)


LABELS_CSV = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv"
)
VINDR_IMAGE_ROOT = Path("/workspace/vinbigdata")
MIMIC_CORPUS = Path(
    "corrected_runs/paper_baselines_v1/full_matrix_v1/rag/combined_corpus/corpus.jsonl"
)
OUT_DIR = Path("corrected_runs/reader_grounded_controlled_source_injection_v1")
MANIFEST = OUT_DIR / "target_blind_manifest.json"
DISCOVERY_MANIFEST = OUT_DIR / "target_blind_discovery.json"
CONFIRMATION_MANIFEST = OUT_DIR / "target_blind_confirmation.json"
DONOR_AUDIT = OUT_DIR / "donor_provenance_audit.jsonl"
VOTE_AUDIT = OUT_DIR / "post_selection_vote_audit.jsonl"
SELECTION_FREEZE = OUT_DIR / "query_selection_freeze.json"
RESULT = OUT_DIR / "result.json"

PROTOCOL = "reader-grounded-controlled-source-injection-v1"
PANEL = ("R8", "R9", "R10")
CORE_FINDINGS = (
    "cardiomegaly",
    "lung_opacity",
    "pleural_effusion",
    "pneumothorax",
)
SOURCE_COLUMNS = {
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung Opacity",
    "pleural_effusion": "Pleural effusion",
    "pneumothorax": "Pneumothorax",
}
DISPLAY = {
    "cardiomegaly": "cardiomegaly",
    "lung_opacity": "lung opacity",
    "pleural_effusion": "pleural effusion",
    "pneumothorax": "pneumothorax",
}
KNOWLEDGE_CARDS = {
    "cardiomegaly": (
        "Assess whether the cardiac silhouette is enlarged relative to the thoracic width."
    ),
    "lung_opacity": (
        "Assess for focal or diffuse increased pulmonary density relative to expected aerated lung."
    ),
    "pleural_effusion": (
        "Assess for costophrenic angle blunting or dependent pleural fluid density."
    ),
    "pneumothorax": (
        "Assess for a visceral pleural line with absent peripheral lung markings."
    ),
}
QUERY_IMAGES_PER_FINDING = 32
PER_SPLIT = 16
SPLITS = ("discovery", "confirmation")
MODEL_TOKENIZERS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}

ARMS = (
    "current_present",
    "current_absent",
    "other_present",
    "other_absent",
    "current_uncertain",
    "other_uncertain",
    "plain",
    "random_unrelated_state",
)
STATE_TAGS = {
    "current_present": (
        "SOURCE_SCOPE: current. STATE_OBJECT: relevant. FINDING_STATE: present."
    ),
    "current_absent": (
        "SOURCE_SCOPE: current. STATE_OBJECT: relevant. FINDING_STATE: absent."
    ),
    "other_present": (
        "SOURCE_SCOPE: other. STATE_OBJECT: relevant. FINDING_STATE: present."
    ),
    "other_absent": (
        "SOURCE_SCOPE: other. STATE_OBJECT: relevant. FINDING_STATE: absent."
    ),
    "current_uncertain": (
        "SOURCE_SCOPE: current. STATE_OBJECT: relevant. FINDING_STATE: uncertain."
    ),
    "other_uncertain": (
        "SOURCE_SCOPE: other. STATE_OBJECT: relevant. FINDING_STATE: uncertain."
    ),
    "plain": (
        "SOURCE_SCOPE: none. STATE_OBJECT: withheld. FINDING_STATE: withheld."
    ),
    "random_unrelated_state": (
        "SOURCE_SCOPE: other. STATE_OBJECT: fracture. FINDING_STATE: present."
    ),
}

DONOR_TEMPORAL_OR_NONASSERTIVE = re.compile(
    r"\b(?:prior|previous|previously|interval|unchanged|stable|remains?|again|"
    r"persist\w*|increas\w*|decreas\w*|improv\w*|resolv\w*|resolution|"
    r"chang\w*|re[\s-]*demonstrat\w*|no change|known|chronic|new|old|healed|status post|"
    r"post[- ]?(?:operative|procedure)|after|recommend\w*|requires?|rule out|"
    r"evaluate|assess|follow[- ]?up)\b",
    re.I,
)
DONOR_UNCERTAINTY = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|may|might|could|cannot|"
    r"can't|uncertain|equivocal|concerning|suggest\w*|compatible|suspicious|"
    r"convincing|question|suspect\w*)\b",
    re.I,
)
DONOR_COMPOUND = re.compile(r"[,;]|\b(?:and|or|but|with|as well as)\b", re.I)
WORD = re.compile(r"\b[\w'-]+\b")
MAX_DONOR_SENTENCE_WORDS = 14


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_panel_image_universe(path: Path) -> list[str]:
    """First pass: intentionally read no finding vote column."""
    readers: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"image_id", "rad_id"} <= set(reader.fieldnames):
            raise ValueError("VinDr label file lacks image_id/rad_id")
        for row in reader:
            image_id, reader_id = row["image_id"].strip(), row["rad_id"].strip()
            if reader_id in readers[image_id]:
                raise ValueError(f"duplicate reader {reader_id} for {image_id}")
            readers[image_id].add(reader_id)
    return sorted(image for image, value in readers.items() if value == set(PANEL))


def freeze_queries(image_universe: list[str]) -> list[dict[str, str]]:
    """Global image-unique hash selection, with no target vote in scope."""
    used: set[str] = set()
    selected: list[dict[str, str]] = []
    for finding in CORE_FINDINGS:
        ordered = sorted(
            (image for image in image_universe if image not in used),
            key=lambda image: stable_hash(f"{PROTOCOL}:query:{finding}:{image}"),
        )
        chosen = ordered[:QUERY_IMAGES_PER_FINDING]
        if len(chosen) != QUERY_IMAGES_PER_FINDING:
            raise RuntimeError(f"{finding}: fewer than {QUERY_IMAGES_PER_FINDING} query images")
        for index, image in enumerate(chosen):
            split = SPLITS[index // PER_SPLIT]
            selected.append({"finding": finding, "image_id": image, "experiment_split": split})
            used.add(image)
    if len(used) != len(selected):
        raise AssertionError("query images are not globally unique")
    return selected


def post_selection_votes(path: Path, selected: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Second pass: votes become visible only after ordered query IDs freeze."""
    keys = {(row["image_id"], row["finding"]): row for row in selected}
    votes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("VinDr label file has no header")
        for row in reader:
            image = row["image_id"].strip()
            for finding, source_column in SOURCE_COLUMNS.items():
                key = (image, finding)
                if key not in keys:
                    continue
                value = row[source_column].strip()
                if value not in {"0", "1", "0.0", "1.0"}:
                    raise ValueError(f"non-binary vote for {key}")
                votes[key].append({"rad_id": row["rad_id"].strip(), "vote": int(float(value))})
    output = []
    for source in selected:
        key = (source["image_id"], source["finding"])
        reader_votes = sorted(votes[key], key=lambda row: row["rad_id"])
        if [row["rad_id"] for row in reader_votes] != sorted(PANEL):
            raise RuntimeError(f"selected image lacks exact fixed reader panel: {key}")
        positive = sum(row["vote"] for row in reader_votes)
        output.append({
            **source,
            "positive_votes": positive,
            "reader_count": 3,
            "vote_bin": f"{positive}/3",
            "reader_votes": reader_votes,
            "selection_used_this_vote": False,
        })
    return output


def clear_lung_negative(sentence: str) -> bool:
    value = sentence.strip().lower()
    return bool(
        re.fullmatch(
            r"(?:findings:\s*)?(?:the\s+)?(?:lungs|lung fields) (?:are|appear) clear\.?",
            value,
        )
    )


def donor_sentence_state(sentence: str, finding: str) -> str | None:
    """High-precision provenance parser; attributes are allowed only for presence."""
    if (
        len(WORD.findall(sentence)) > MAX_DONOR_SENTENCE_WORDS
        or re.match(r"\s*\d+\W", sentence)
        or DONOR_TEMPORAL_OR_NONASSERTIVE.search(sentence)
        or DONOR_UNCERTAINTY.search(sentence)
        or DONOR_COMPOUND.search(sentence)
    ):
        return None
    if finding == "lung_opacity" and clear_lung_negative(sentence):
        return "negative"
    if other_findings(sentence, finding):
        return None
    mentions = finding_mentions(sentence, finding)
    assertions = sentence_assertions(sentence, finding)
    if len(mentions) != 1 or len(assertions) != 1:
        return None
    state = assertions[0]
    if state not in {"positive", "negative"}:
        return None
    # "No large effusion" does not establish generic absence.  Any attribute
    # in a negative source is therefore excluded.  Positive attributes remain
    # valid provenance for generic presence because no attribute is injected.
    if state == "negative" and ATTRIBUTE.search(sentence):
        return None
    return state


def build_donor_pool(rows: list[dict[str, Any]], findings: set[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    pool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("dataset") != "mimic" or row.get("source_split") != "train":
            continue
        if not all(str(row.get(key, "")).strip() for key in ("patient_id", "study_id", "image_id")):
            continue
        for finding in findings:
            for begin, end, sentence in sentence_spans(str(row["report"])):
                state = donor_sentence_state(sentence, finding)
                if state is None:
                    continue
                key = (finding, state, str(row["patient_id"]))
                if key in seen:
                    continue
                seen.add(key)
                pool[(finding, state)].append({
                    "finding": finding,
                    "state": state,
                    "doc_id": row["doc_id"],
                    "patient_id": str(row["patient_id"]),
                    "study_id": str(row["study_id"]),
                    "image_id": str(row["image_id"]),
                    "report_sha256": row["report_sha256"],
                    "sentence": sentence,
                    "sentence_span": [begin, end],
                })
    for key in pool:
        pool[key].sort(
            key=lambda row: stable_hash(
                f"{PROTOCOL}:donor:{key[0]}:{key[1]}:{row['patient_id']}:{row['report_sha256']}"
            )
        )
    return pool


def select_donors(pool: dict[tuple[str, str], list[dict[str, Any]]]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    used_patients: set[str] = set()
    used_studies: set[str] = set()
    used_images: set[str] = set()
    shortage = {}
    requirements = [(finding, state) for finding in CORE_FINDINGS for state in ("positive", "negative")]
    requirements.append(("fracture", "positive"))
    for finding, state in requirements:
        candidates = pool.get((finding, state), [])
        chosen = []
        for row in candidates:
            if (
                row["patient_id"] in used_patients
                or row["study_id"] in used_studies
                or row["image_id"] in used_images
            ):
                continue
            chosen.append(row)
            used_patients.add(row["patient_id"])
            used_studies.add(row["study_id"])
            used_images.add(row["image_id"])
            if len(chosen) == len(SPLITS):
                break
        if len(chosen) < len(SPLITS):
            shortage[f"{finding}:{state}"] = {
                "eligible_unique_patient_candidates": len({row["patient_id"] for row in candidates}),
                "required_disjoint_split_donors": len(SPLITS),
                "selected": len(chosen),
            }
        for split, row in zip(SPLITS, chosen):
            selected[(finding, state, split)] = row
    leakage = {
        "selected_donors": len(selected),
        "unique_patient_ids": len({row["patient_id"] for row in selected.values()}),
        "unique_study_ids": len({row["study_id"] for row in selected.values()}),
        "unique_image_ids": len({row["image_id"] for row in selected.values()}),
        "all_selected_donor_patients_studies_images_disjoint": (
            len(selected)
            == len({row["patient_id"] for row in selected.values()})
            == len({row["study_id"] for row in selected.values()})
            == len({row["image_id"] for row in selected.values()})
        ),
        "shortage": shortage,
    }
    return selected, leakage


def prompt(finding: str, arm: str) -> str:
    return (
        "This is a controlled source-injection mechanism probe, not natural retrieval. "
        "The state line is an experimental symbol and may be false; current does not mean "
        "ground truth. Use the radiograph as primary evidence.\n"
        f"VISUAL_KNOWLEDGE_CARD: {KNOWLEDGE_CARDS[finding]}\n"
        f"CONTROLLED_STATE: {STATE_TAGS[arm]}\n"
        f"Question: Is {DISPLAY[finding]} present on the current chest X-ray?\n"
        "Answer with exactly one of: Yes, No, Uncertain."
    )


def donor_for_arm(
    donors: dict[tuple[str, str, str], dict[str, Any]], finding: str, split: str, arm: str
) -> dict[str, Any] | None:
    if arm.endswith("_present"):
        return donors[(finding, "positive", split)]
    if arm.endswith("_absent"):
        return donors[(finding, "negative", split)]
    if arm == "random_unrelated_state":
        return donors[("fracture", "positive", split)]
    return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    image_universe = fixed_panel_image_universe(LABELS_CSV)
    frozen_queries = freeze_queries(image_universe)
    selection_payload = {
        "protocol": PROTOCOL,
        "selection_stage": "completed_before_any_finding_vote_column_was_read",
        "labels_csv_sha256": sha256_file(LABELS_CSV),
        "fixed_reader_panel": list(PANEL),
        "fixed_panel_image_universe": len(image_universe),
        "selection_rule": (
            "fixed finding order; among globally unused fixed-panel images, ascending "
            "SHA256(protocol,query,finding,image_id); first 32; first 16 discovery, next 16 confirmation"
        ),
        "selection_uses_target_votes": False,
        "selected": frozen_queries,
        "ordered_selection_sha256": stable_hash(json.dumps(frozen_queries, sort_keys=True)),
    }
    votes = post_selection_votes(LABELS_CSV, frozen_queries)

    corpus_rows = read_jsonl(MIMIC_CORPUS)
    donor_findings = set(CORE_FINDINGS) | {"fracture"}
    donor_pool = build_donor_pool(corpus_rows, donor_findings)
    donors, donor_leakage = select_donors(donor_pool)
    required_donors = (len(CORE_FINDINGS) * 2 + 1) * len(SPLITS)
    gate_failures = []
    if len(frozen_queries) != len(CORE_FINDINGS) * QUERY_IMAGES_PER_FINDING:
        gate_failures.append("query_count")
    if len({row["image_id"] for row in frozen_queries}) != len(frozen_queries):
        gate_failures.append("query_image_disjoint")
    if len(donors) != required_donors or donor_leakage["shortage"]:
        gate_failures.append("mimic_donor_availability")
    if not donor_leakage["all_selected_donor_patients_studies_images_disjoint"]:
        gate_failures.append("mimic_donor_patient_study_image_disjoint")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SELECTION_FREEZE, selection_payload)
    VOTE_AUDIT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in votes), encoding="utf-8"
    )
    DONOR_AUDIT.write_text(
        "".join(
            json.dumps({"experiment_split": key[2], **row}, sort_keys=True) + "\n"
            for key, row in sorted(donors.items())
        ),
        encoding="utf-8",
    )

    pool_counts = {
        finding: {
            state: len({row["patient_id"] for row in donor_pool.get((finding, state), [])})
            for state in ("positive", "negative")
        }
        for finding in sorted(donor_findings)
    }
    vote_balance: dict[str, Any] = {}
    for finding in CORE_FINDINGS:
        vote_balance[finding] = {
            split: dict(
                sorted(
                    Counter(
                        row["vote_bin"]
                        for row in votes
                        if row["finding"] == finding and row["experiment_split"] == split
                    ).items()
                )
            )
            for split in SPLITS
        }

    if gate_failures:
        result = {
            "status": "stopped_pre_manifest_gate_failure",
            "protocol": PROTOCOL,
            "gate_failures": gate_failures,
            "gpu_execution": "not_run",
            "query_selection": selection_payload,
            "post_selection_vote_bin_evaluation": vote_balance,
            "donor_pool_unique_patient_counts": pool_counts,
            "donor_leakage": donor_leakage,
            "manifest_generated": False,
        }
        write_json(RESULT, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True
        )
        for name, path in MODEL_TOKENIZERS.items()
    }
    manifest_rows: list[dict[str, Any]] = []
    item_token_audits: list[dict[str, Any]] = []
    for source in frozen_queries:
        finding, image, split = source["finding"], source["image_id"], source["experiment_split"]
        questions = {arm: prompt(finding, arm) for arm in ARMS}
        token_counts = {
            model: {
                arm: len(tokenizer.encode(question, add_special_tokens=False))
                for arm, question in questions.items()
            }
            for model, tokenizer in tokenizers.items()
        }
        item_token_audits.append({
            "finding": finding,
            "image_id": image,
            "experiment_split": split,
            "counts": token_counts,
        })
        for model, counts in token_counts.items():
            if len(set(counts.values())) != 1:
                raise RuntimeError(f"{finding}/{image}: {model} arm token mismatch: {counts}")
        for arm in ARMS:
            donor = donor_for_arm(donors, finding, split, arm)
            row = {
                "arm": arm,
                "controlled_source_injection_not_natural_rag": True,
                "dataset": "vindr-cxr-1.0.0",
                "experiment_split": split,
                "finding": finding,
                "id": f"{finding}:{image}:{arm}",
                "img_name": f"train/{image}.dicom",
                "pair_id": f"{finding}:{image}",
                "prompt_contract": PROTOCOL,
                "qid": f"{finding}:{image}:{arm}",
                "question": questions[arm],
                "question_type": "binary_target_blinded",
                "selection_uses_target_vote": False,
                "task": "controlled_source_binding_generation",
            }
            if donor is not None:
                row.update({
                    "source_provenance_dataset": "mimic_train",
                    "source_provenance_report_sha256": donor["report_sha256"],
                    "source_provenance_patient_hash": stable_hash(donor["patient_id"]),
                })
            else:
                row["source_provenance_dataset"] = "canonical_control_no_donor"
            manifest_rows.append(row)

    write_json(MANIFEST, manifest_rows)
    write_json(
        DISCOVERY_MANIFEST,
        [row for row in manifest_rows if row["experiment_split"] == "discovery"],
    )
    write_json(
        CONFIRMATION_MANIFEST,
        [row for row in manifest_rows if row["experiment_split"] == "confirmation"],
    )

    loaded = load_target_blind_manifest(MANIFEST, limit=0)
    discovery_loaded = load_target_blind_manifest(DISCOVERY_MANIFEST, limit=0)
    confirmation_loaded = load_target_blind_manifest(CONFIRMATION_MANIFEST, limit=0)
    unique_image_rows = []
    seen_images = set()
    for row in loaded:
        if row["img_name"] not in seen_images:
            unique_image_rows.append(row)
            seen_images.add(row["img_name"])
    image_preflight = preflight_inputs(unique_image_rows, VINDR_IMAGE_ROOT)

    token_summary = {}
    for model in tokenizers:
        gaps = []
        counts = []
        for item in item_token_audits:
            values = list(item["counts"][model].values())
            gaps.append(max(values) - min(values))
            counts.extend(values)
        token_summary[model] = {
            "tokenizer_path": str(MODEL_TOKENIZERS[model]),
            "all_eight_arms_exact_within_each_query": all(gap == 0 for gap in gaps),
            "maximum_within_query_arm_gap": max(gaps),
            "minimum_prompt_tokens": min(counts),
            "maximum_prompt_tokens": max(counts),
        }

    discovery_images = {
        row["img_name"] for row in manifest_rows if row["experiment_split"] == "discovery"
    }
    confirmation_images = {
        row["img_name"] for row in manifest_rows if row["experiment_split"] == "confirmation"
    }
    discovery_patients = {
        row["patient_id"] for key, row in donors.items() if key[2] == "discovery"
    }
    confirmation_patients = {
        row["patient_id"] for key, row in donors.items() if key[2] == "confirmation"
    }
    result = {
        "status": "completed_cpu_only_target_blind_controlled_substrate",
        "protocol": PROTOCOL,
        "interpretation": {
            "natural_rag": False,
            "intended_use": "mechanism and attention-edge source-by-polarity binding experiment",
            "current_tag_is_target_truth": False,
            "canonical_state_may_be_false": True,
            "mimic_donor_text_injected": False,
            "mimic_donor_role": "cross-dataset ecological source-state provenance only",
        },
        "counts": {
            "query_images": len(frozen_queries),
            "query_images_per_finding": dict(Counter(row["finding"] for row in frozen_queries)),
            "query_images_per_split": dict(Counter(row["experiment_split"] for row in frozen_queries)),
            "arms_per_query": len(ARMS),
            "manifest_rows": len(manifest_rows),
            "rows_per_arm": dict(Counter(row["arm"] for row in manifest_rows)),
        },
        "factorial": {
            "primary_2x2": [
                "current_present", "current_absent", "other_present", "other_absent"
            ],
            "controls": [
                "current_uncertain", "other_uncertain", "plain", "random_unrelated_state"
            ],
            "state_tags": STATE_TAGS,
            "tag_whitespace_word_counts": {
                arm: len(value.split()) for arm, value in STATE_TAGS.items()
            },
            "tag_position_fixed": True,
            "knowledge_card_fixed_within_finding": True,
        },
        "query_selection": {
            "freeze_file": str(SELECTION_FREEZE),
            "freeze_sha256": sha256_file(SELECTION_FREEZE),
            "votes_visible_during_selection": False,
            "post_selection_vote_bin_evaluation": vote_balance,
        },
        "donor_provenance": {
            "source": str(MIMIC_CORPUS),
            "source_sha256": sha256_file(MIMIC_CORPUS),
            "source_split": "train_only",
            "unique_patient_candidate_counts": pool_counts,
            "selected": donor_leakage,
            "raw_patient_ids_only_in_separate_donor_audit": True,
        },
        "leakage_audit": {
            "query_images_globally_unique": len(frozen_queries) == len({row["image_id"] for row in frozen_queries}),
            "discovery_confirmation_query_image_intersection": len(discovery_images & confirmation_images),
            "selected_mimic_donor_patient_study_image_disjoint": donor_leakage[
                "all_selected_donor_patients_studies_images_disjoint"
            ],
            "discovery_confirmation_donor_patient_intersection": len(
                discovery_patients & confirmation_patients
            ),
            "donor_query_dataset_and_institution_disjoint": True,
            "donor_dataset": "MIMIC-CXR/Beth Israel Deaconess Medical Center",
            "query_dataset": "VinDr-CXR/H108 and HMUH",
            "target_vote_fields_in_runner_manifest": 0,
        },
        "token_length_audit": token_summary,
        "runner_validation": {
            "combined_rows": len(loaded),
            "discovery_rows": len(discovery_loaded),
            "confirmation_rows": len(confirmation_loaded),
            "target_blind_loader_passed": True,
            "unique_image_cpu_preflight": image_preflight,
            "image_root": str(VINDR_IMAGE_ROOT),
        },
        "gpu_execution": "not_run",
        "artifacts": {
            "combined_manifest": str(MANIFEST),
            "combined_manifest_sha256": sha256_file(MANIFEST),
            "discovery_manifest": str(DISCOVERY_MANIFEST),
            "discovery_manifest_sha256": sha256_file(DISCOVERY_MANIFEST),
            "confirmation_manifest": str(CONFIRMATION_MANIFEST),
            "confirmation_manifest_sha256": sha256_file(CONFIRMATION_MANIFEST),
            "donor_audit": str(DONOR_AUDIT),
            "donor_audit_sha256": sha256_file(DONOR_AUDIT),
            "post_selection_vote_audit": str(VOTE_AUDIT),
            "post_selection_vote_audit_sha256": sha256_file(VOTE_AUDIT),
        },
    }
    write_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
