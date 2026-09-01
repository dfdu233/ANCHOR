#!/usr/bin/env python3
"""Source-only broad-ontology preflight and blinded extractor review pack.

This discovery tool reads only Huatuo/PubMedVision source training text and
the frozen source index.  It never consumes VinDr labels/images, model outputs,
or GPU state.  Assistant assertions, question presuppositions, and source
captions remain separate channels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from anchor.corrected_sgta.build_pubmedvision_source_semantic_admission_v1 import (
    DEFAULT_ALIGNMENT,
    DEFAULT_INSTRUCTION,
    DEFAULT_SOURCE_INDEX,
    QUESTION_LABELS,
    STATES,
    FindingSpec,
    atomic_write,
    load_source_index,
    scan_stage,
    sha256_file,
)


VERSION = "pubmedvision-source-semantic-admission-v3.5"
REVIEW_RANDOMIZATION_VERSION = "pubmedvision-source-semantic-review-v3"
DEFAULT_ONTOLOGY = Path(
    "/home/dbw/ANCHOR/configs/pubmedvision_source_atomic_ontology_v3.json"
)
DEFAULT_OUTPUT = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr_v2/"
    "source_semantic_admission_v3_5"
)
PRIMARY_DOMAIN = "alignment_generic"
AUDIT_DOMAINS = (PRIMARY_DOMAIN, "instruction_tuning")
MIN_TRAIN_POSITIVE = 20
MIN_DEV_POSITIVE = 5
REVIEW_PER_PREDICTED_STATE_DOMAIN = 30
ONE_SIDED_ALPHA = 0.05
TARGET_POSITIVE_PRECISION = 0.90
TARGET_MACRO_F1 = 0.80

# Universal assertion-scope rules.  Aliases stay claim-specific in JSON, while
# polarity and epistemic commitment are handled identically for every claim.
_CLAUSE_BOUNDARIES = ".;!?\n"
_ADVERSATIVE = re.compile(r"\b(?:but|however|although|though|yet|nevertheless)\b", re.I)
_NEGATIVE_PREFIX = re.compile(
    r"\b(?:no|without|absen(?:ce|t)|free\s+of|negative\s+for|lack(?:ing)?|"
    r"neither|nor)\b"
    r"|\b(?:do|does|did|was|were|is|are|has|have|had)\s+not\b"
    r"|\bnot\s+(?:showing|demonstrating|revealing|identified|seen|present)\b",
    re.I,
)
_NEGATIVE_SUFFIX = re.compile(
    r"^\s*(?:is|are|was|were)?\s*(?:absent|not\s+present|not\s+seen|"
    r"not\s+identified|not\s+demonstrated)",
    re.I,
)
_HEDGED_NEGATION_PREFIX = re.compile(
    r"\bno\s+(?:definite|significant|large|new|obvious|clear|gross)\b", re.I
)
_STABILITY_PREFIX = re.compile(
    r"\b(?:no\s+(?:interval\s+)?(?:change|increase|worsening)\s+(?:in|of)|unchanged)\b",
    re.I,
)
_UNCERTAIN_PREFIX = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|perhaps|potential|potentially|"
    r"presumed|presumably|suspected|suspicious)\b"
    r"|\b(?:may|might|could|can)\s+(?:be|represent|reflect|indicate|suggest|include)\b"
    r"|\b(?:appears?|seems?)\s+to\s+(?:show|demonstrate|represent|be|have)\b"
    r"|\b(?:suggests?|suggesting)\b"
    r"|\b(?:suggestive\s+of|suspicious\s+for|concerning\s+for|compatible\s+with|"
    r"consistent\s+with|indicative\s+of|equivocal\s+for|questionable\s+for)\b"
    r"|\b(?:such\s+as|including|for\s+example|e\.g\.)\b"
    r"|\b(?:differential\s+diagnos(?:is|es)|possibilit(?:y|ies)\s+include)\b"
    r"|\b(?:history\s+of|previous|previously|prior|resolved|resolution\s+of)\b",
    re.I,
)
_UNCERTAIN_SUFFIX = re.compile(
    r"^\s*(?:(?:is|are|was|were)\s+)?(?:possible|probable|likely|suspected|"
    r"not\s+(?:entirely\s+)?excluded|not\s+(?:significantly|clearly|definitely|obviously)\s+present|"
    r"cannot\s+be\s+excluded|can\s+not\s+be\s+excluded|observed\s+previously|"
    r"previously\s+(?:seen|observed|noted)|has\s+resolved|have\s+resolved|resolved)",
    re.I,
)
_NEUTRAL_QUESTION = re.compile(
    r"\b(?:is|are)\s+there\b|\bdoes\s+(?:the\s+|this\s+)?(?:image|x-ray|radiograph)\s+show\b"
    r"|\bpresence\s+or\s+absence\b|\bwhether\b|\bevaluate\s+for\b|\bassess\s+for\b"
    r"|\bcan\s+(?:you\s+)?(?:identify|see|detect)\b",
    re.I,
)


def _local_clause(text: str, start: int, end: int, radius: int = 140) -> tuple[str, int, int]:
    left = max((text.rfind(mark, 0, start) for mark in _CLAUSE_BOUNDARIES), default=-1) + 1
    right_values = [text.find(mark, end) for mark in _CLAUSE_BOUNDARIES]
    right_values = [value for value in right_values if value >= 0]
    right = min(right_values) if right_values else len(text)
    left = max(left, start - radius)
    right = min(right, end + radius)
    return text[left:right], start - left, end - left


def _prefix_since_adversative(prefix: str, radius: int = 100) -> str:
    value = prefix[-radius:]
    matches = list(_ADVERSATIVE.finditer(value))
    return value[matches[-1].end() :] if matches else value


def classify_finding_v3(text: str, spec: FindingSpec) -> dict[str, Any]:
    """Conservative four-state extraction with list-aware generic scope.

    A bare alias is positive only if no scoped negative or epistemic cue is
    found.  Negation is allowed to distribute through comma/and/or lists; an
    adversative resets it.  Differential/example mentions are uncertain, not
    definite assertions.  Mixed local states fail conservatively to uncertain.
    """

    normalized = " ".join(str(text or "").split())
    mentions = list(spec.alias_pattern.finditer(normalized))
    if not mentions:
        return {"state": "unmentioned", "evidence": []}
    evidence = []
    local_states = []
    for mention in mentions:
        context, local_start, local_end = _local_clause(
            normalized, mention.start(), mention.end()
        )
        prefix = _prefix_since_adversative(context[:local_start])
        suffix = context[local_end : local_end + 75]
        hedge = _HEDGED_NEGATION_PREFIX.search(prefix)
        stability = _STABILITY_PREFIX.search(prefix)
        negative = None if stability else (_NEGATIVE_PREFIX.search(prefix) or _NEGATIVE_SUFFIX.search(suffix))
        uncertain_prefix = _UNCERTAIN_PREFIX.search(prefix[-100:])
        uncertain_suffix = _UNCERTAIN_SUFFIX.search(suffix)
        if hedge or uncertain_suffix:
            state = "uncertain"
            cue_match = hedge or uncertain_suffix
        elif negative:
            state = "negative"
            cue_match = negative
        elif uncertain_prefix:
            state = "uncertain"
            cue_match = uncertain_prefix
        else:
            state = "positive"
            cue_match = stability
        local_states.append(state)
        evidence.append(
            {
                "mention": mention.group(0),
                "context": context,
                "local_state": state,
                "cue": cue_match.group(0) if cue_match else None,
            }
        )
    unique = set(local_states)
    if "uncertain" in unique or len(unique) > 1:
        state = "uncertain"
    elif "positive" in unique:
        state = "positive"
    else:
        state = "negative"
    return {"state": state, "evidence": evidence}


def classify_question_v3(text: str, spec: FindingSpec) -> dict[str, Any]:
    surface = classify_finding_v3(text, spec)
    state = surface["state"]
    if state == "unmentioned":
        label = "none"
    elif state == "negative":
        label = "presupposes_negative"
    elif state == "uncertain":
        label = "uncertain"
    elif _NEUTRAL_QUESTION.search(str(text)):
        label = "neutral_query"
    else:
        label = "presupposes_positive"
    return {"surface_state": state, "presupposition": label, "evidence": surface["evidence"]}


def _literal_alias_regex(alias: str) -> str:
    """Turn an ontology phrase into an auditable whitespace-tolerant regex."""

    escaped = re.escape(alias.strip()).replace(r"\ ", r"\s+")
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def load_ontology(path: Path) -> tuple[dict[str, Any], tuple[FindingSpec, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise ValueError("ontology must be an object with a claims list")
    if not str(payload.get("schema_version", "")).endswith("-v3"):
        raise ValueError("ontology schema_version must be explicitly versioned v3")
    expected = int(payload.get("expected_claim_count", -1))
    if expected != len(payload["claims"]):
        raise ValueError(f"ontology expected {expected} claims but contains {len(payload['claims'])}")
    excluded = set(payload.get("excluded_non_atomic_labels", []))
    seen_claims: set[str] = set()
    seen_aliases: dict[str, str] = {}
    specs = []
    for index, row in enumerate(payload["claims"]):
        if not isinstance(row, dict):
            raise ValueError(f"ontology claim {index} is not an object")
        claim_id = str(row.get("claim_id", ""))
        aliases = row.get("aliases")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", claim_id):
            raise ValueError(f"invalid claim_id: {claim_id!r}")
        if claim_id in seen_claims or claim_id in excluded:
            raise ValueError(f"duplicate/excluded claim_id: {claim_id}")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"claim {claim_id} has no aliases")
        normalized = []
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"claim {claim_id} has invalid alias")
            canonical = " ".join(alias.lower().split())
            owner = seen_aliases.get(canonical)
            if owner is not None and owner != claim_id:
                raise ValueError(f"alias {alias!r} shared by {owner} and {claim_id}")
            seen_aliases[canonical] = claim_id
            normalized.append(_literal_alias_regex(canonical))
        seen_claims.add(claim_id)
        specs.append(FindingSpec(claim_id, tuple(normalized)))
    return payload, tuple(specs)


def response_domain(unit: Mapping[str, Any]) -> str | None:
    if unit["stage"] == "alignment" and unit["generic_alignment_prompt"]:
        return PRIMARY_DOMAIN
    if unit["stage"] == "instruction_tuning":
        return "instruction_tuning"
    return None


def deterministic_key(seed: int, *parts: object) -> str:
    value = ":".join(str(part) for part in (REVIEW_RANDOMIZATION_VERSION, seed, *parts))
    return hashlib.sha256(value.encode()).hexdigest()


def raw_counts(
    rows: Sequence[Mapping[str, Any]], claim_ids: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    domains = ("alignment_all", PRIMARY_DOMAIN, "instruction_tuning")
    for domain in domains:
        result[domain] = {}
        for claim_id in claim_ids:
            result[domain][claim_id] = {}
            for split in ("source_train", "source_dev", "source_review"):
                subset = [
                    row
                    for row in rows
                    if row["finding"] == claim_id
                    and row["source_split"] == split
                    and (
                        (domain == "alignment_all" and row["stage"] == "alignment")
                        or (domain == PRIMARY_DOMAIN and row["review_domain"] == PRIMARY_DOMAIN)
                        or (domain == "instruction_tuning" and row["stage"] == "instruction_tuning")
                    )
                ]
                counts = Counter(str(row["assistant_state"]) for row in subset)
                result[domain][claim_id][split] = {
                    state: int(counts[state]) for state in STATES
                }
    return result


def eligibility_from_counts(
    counts: Mapping[str, Any], claim_ids: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    primary = counts[PRIMARY_DOMAIN]
    for claim_id in claim_ids:
        train_positive = int(primary[claim_id]["source_train"]["positive"])
        dev_positive = int(primary[claim_id]["source_dev"]["positive"])
        eligible = train_positive >= MIN_TRAIN_POSITIVE and dev_positive >= MIN_DEV_POSITIVE
        result.append(
            {
                "claim_id": claim_id,
                "source_train_positive": train_positive,
                "source_dev_positive": dev_positive,
                "automatic_count_eligible": eligible,
                "rule": (
                    f"alignment_generic source_train positive >= {MIN_TRAIN_POSITIVE} "
                    f"and source_dev positive >= {MIN_DEV_POSITIVE}"
                ),
            }
        )
    return result


def raw_question_counts(
    rows: Sequence[Mapping[str, Any]], claim_ids: Sequence[str]
) -> dict[str, Any]:
    """Keep human-question presuppositions separate from assistant assertions."""

    result: dict[str, Any] = {}
    for stage in ("alignment", "instruction_tuning"):
        result[stage] = {}
        for claim_id in claim_ids:
            result[stage][claim_id] = {}
            for split in ("source_train", "source_dev", "source_review"):
                counts = Counter(
                    str(row["question_presupposition"])
                    for row in rows
                    if row["stage"] == stage
                    and row["finding"] == claim_id
                    and row["source_split"] == split
                )
                result[stage][claim_id][split] = {
                    label: int(counts[label]) for label in QUESTION_LABELS
                }
    return result


def select_blind_review(
    rows: Sequence[Mapping[str, Any]], eligible_claims: set[str], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample prediction strata and retain inverse-probability weights.

    Each stage/domain is audited separately.  Sparse strata are censused rather
    than padded with a different state.  That preserves raw zero/low counts and
    makes inability to validate a class explicit.
    """

    selected: list[dict[str, Any]] = []
    plan: dict[str, Any] = {}
    for claim_id in sorted(eligible_claims):
        plan[claim_id] = {}
        for domain in AUDIT_DOMAINS:
            plan[claim_id][domain] = {"predicted_state_strata": {}}
            domain_selected = []
            for state in STATES:
                population = [
                    dict(row)
                    for row in rows
                    if row["finding"] == claim_id
                    and row["review_domain"] == domain
                    and row["assistant_state"] == state
                ]
                population.sort(
                    key=lambda row: deterministic_key(
                        seed,
                        claim_id,
                        domain,
                        state,
                        row["source_split"],
                        row["response_unit_id"],
                    )
                )
                n = min(REVIEW_PER_PREDICTED_STATE_DOMAIN, len(population))
                chosen = population[:n]
                inclusion_probability = n / len(population) if population else 0.0
                design_weight = len(population) / n if n else None
                for row in chosen:
                    row["review_domain"] = domain
                    row["predicted_state_population_n"] = len(population)
                    row["predicted_state_sample_n"] = n
                    row["inclusion_probability"] = inclusion_probability
                    row["design_weight"] = design_weight
                selected.extend(chosen)
                domain_selected.extend(chosen)
                plan[claim_id][domain]["predicted_state_strata"][state] = {
                    "population_n": len(population),
                    "sample_n": n,
                    "target_n": REVIEW_PER_PREDICTED_STATE_DOMAIN,
                    "census": bool(population) and n == len(population),
                    "inclusion_probability": inclusion_probability,
                    "design_weight": design_weight,
                }
            positive_n = sum(
                row["assistant_state"] == "positive" for row in domain_selected
            )
            minimum_zero_error_n = math.ceil(
                math.log(ONE_SIDED_ALPHA) / math.log(TARGET_POSITIVE_PRECISION)
            )
            lower_if_zero_errors = (
                ONE_SIDED_ALPHA ** (1.0 / positive_n) if positive_n else 0.0
            )
            plan[claim_id][domain].update(
                {
                    "sample_n": len(domain_selected),
                    "positive_sample_n": positive_n,
                    "minimum_positive_n_for_zero_error_one_sided_95pct_lower_ge_0p90": minimum_zero_error_n,
                    "positive_precision_zero_error_bound_capable": positive_n >= minimum_zero_error_n,
                    "one_sided_95pct_lower_if_all_sampled_positives_are_correct": lower_if_zero_errors,
                    "macro_f1_point_validation_design": (
                        "inverse-probability-weighted confusion matrix over separately sampled "
                        "predicted-state strata; sparse strata are censused"
                    ),
                    "macro_f1_target": TARGET_MACRO_F1,
                }
            )
    selected.sort(
        key=lambda row: deterministic_key(
            seed,
            "final",
            row["finding"],
            row["review_domain"],
            row["assistant_state"],
            row["response_unit_id"],
        )
    )
    return selected, plan


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    final_names = (
        "preflight_counts.json",
        "question_presupposition_counts.json",
        "eligible_claims.json",
        "mention_records.jsonl",
        "blind_review.csv",
        "blind_review_key.jsonl",
        "review_sampling_plan.json",
        "stats.json",
        "_COMPLETE.json",
    )
    existing = [name for name in final_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError("v3 output is write-once; existing=" + ",".join(existing))

    ontology, specs = load_ontology(args.ontology)
    claim_ids = [spec.finding for spec in specs]
    source_rows, source_schema = load_source_index(args.source_index)
    stages = (("alignment", args.alignment), ("instruction_tuning", args.instruction))
    units = []
    stage_schema = {}
    for stage, path in stages:
        stage_units, schema = scan_stage(
            path, stage, source_rows, streaming=args.streaming_fallback
        )
        units.extend(stage_units)
        stage_schema[stage] = schema

    semantic_rows = []
    mentions = []
    for unit in units:
        domain = response_domain(unit)
        caption_text = "\n".join(unit["captions"])
        for spec in specs:
            assistant = classify_finding_v3(unit["assistant_response"], spec)
            question = classify_question_v3(unit["question"], spec)
            caption = classify_finding_v3(caption_text, spec)
            row = {
                **unit,
                "review_domain": domain,
                "finding": spec.finding,
                "assistant_state": assistant["state"],
                "assistant_evidence": assistant["evidence"],
                "question_surface_state": question["surface_state"],
                "question_presupposition": question["presupposition"],
                "question_evidence": question["evidence"],
                "caption_crosscheck_state": caption["state"],
                "caption_crosscheck_evidence": caption["evidence"],
            }
            semantic_rows.append(row)
            if (
                assistant["state"] != "unmentioned"
                or question["presupposition"] != "none"
                or caption["state"] != "unmentioned"
            ):
                mentions.append(row)

    counts = raw_counts(semantic_rows, claim_ids)
    question_counts = raw_question_counts(semantic_rows, claim_ids)
    eligibility = eligibility_from_counts(counts, claim_ids)
    eligible_set = {
        row["claim_id"] for row in eligibility if row["automatic_count_eligible"]
    }
    review_rows, review_plan = select_blind_review(semantic_rows, eligible_set, args.seed)
    if not review_rows:
        raise RuntimeError("no count-eligible claims; refusing to emit an empty review pack")

    public_rows = []
    private_rows = []
    for row in review_rows:
        review_id = hashlib.sha256(
            f"{REVIEW_RANDOMIZATION_VERSION}:{args.seed}:{row['finding']}:"
            f"{row['review_domain']}:{row['response_unit_id']}".encode()
        ).hexdigest()[:20]
        public_rows.append(
            {
                "review_id": review_id,
                "review_domain": row["review_domain"],
                "stage": row["stage"],
                "source_split": row["source_split"],
                "finding": row["finding"],
                "question": row["question"],
                "assistant_response": row["assistant_response"],
                "reviewer_assistant_state": "",
                "reviewer_question_presupposition": "",
                "reviewer_evidence_span": "",
                "reviewer_notes": "",
            }
        )
        private_rows.append(
            {
                "review_id": review_id,
                "response_unit_id": row["response_unit_id"],
                "vqa_id": row["vqa_id"],
                "review_domain": row["review_domain"],
                "stage": row["stage"],
                "source_split": row["source_split"],
                "source_group": row["source_group"],
                "finding": row["finding"],
                "automatic_assistant_state": row["assistant_state"],
                "automatic_assistant_evidence": row["assistant_evidence"],
                "automatic_question_surface_state": row["question_surface_state"],
                "automatic_question_presupposition": row["question_presupposition"],
                "caption_crosscheck_state": row["caption_crosscheck_state"],
                "predicted_state_population_n": row["predicted_state_population_n"],
                "predicted_state_sample_n": row["predicted_state_sample_n"],
                "inclusion_probability": row["inclusion_probability"],
                "design_weight": row["design_weight"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "preflight_counts.json", json.dumps(counts, indent=2, sort_keys=True) + "\n")
    atomic_write(
        output_dir / "question_presupposition_counts.json",
        json.dumps(question_counts, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(output_dir / "eligible_claims.json", json.dumps(eligibility, indent=2, sort_keys=True) + "\n")
    atomic_write(
        output_dir / "mention_records.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mentions),
    )
    _write_csv(output_dir / "blind_review.csv", public_rows)
    atomic_write(
        output_dir / "blind_review_key.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in private_rows),
    )
    (output_dir / "blind_review_key.jsonl").chmod(0o600)
    atomic_write(
        output_dir / "review_sampling_plan.json",
        json.dumps(review_plan, indent=2, sort_keys=True) + "\n",
    )

    stats = {
        "version": VERSION,
        "repair_note": (
            "v3.5 supersedes invalid v3-v3.4 drafts by completing generic assertion scope "
            "and serializing the separately stratified question-presupposition audit"
        ),
        "status": "discovery_source_only_preflight_and_blind_review_pack",
        "scope": (
            "Huatuo PubMedVision assistant source text only; questions and captions are "
            "separate audit channels; no VinDr, target labels, model outputs, or GPU"
        ),
        "ontology": {
            "path": str(args.ontology.resolve()),
            "sha256": sha256_file(args.ontology),
            "schema_version": ontology["schema_version"],
            "claim_count": len(specs),
            "claim_ids": claim_ids,
        },
        "inputs": {
            "source_index": {"path": str(args.source_index.resolve()), "sha256": sha256_file(args.source_index), "schema": source_schema},
            "alignment": {"path": str(args.alignment.resolve()), "sha256": sha256_file(args.alignment), "schema": stage_schema["alignment"]},
            "instruction": {"path": str(args.instruction.resolve()), "sha256": sha256_file(args.instruction), "schema": stage_schema["instruction_tuning"]},
        },
        "raw_count_policy": "no smoothing or pseudocounts; raw zeros retained",
        "unmentioned_is_negative": False,
        "eligibility_policy": {
            "domain": PRIMARY_DOMAIN,
            "source_train_min_positive": MIN_TRAIN_POSITIVE,
            "source_dev_min_positive": MIN_DEV_POSITIVE,
            "eligible_claim_count": len(eligible_set),
            "eligible_claims": sorted(eligible_set),
            "model_output_used": False,
        },
        "review_design": {
            "domains_separate": list(AUDIT_DOMAINS),
            "predicted_state_strata": list(STATES),
            "target_per_predicted_state_per_domain_per_claim": REVIEW_PER_PREDICTED_STATE_DOMAIN,
            "sparse_stratum_policy": "census; never pad from another state",
            "sampling_frame": "all source splits after automatic count eligibility; frozen extractor; no post-review alias tuning",
            "inverse_probability_weights_in_private_key": True,
            "target_positive_precision": TARGET_POSITIVE_PRECISION,
            "target_macro_f1": TARGET_MACRO_F1,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "minimum_zero_error_positive_reviews": math.ceil(math.log(ONE_SIDED_ALPHA) / math.log(TARGET_POSITIVE_PRECISION)),
        },
        "assistant_response_units": len(units),
        "semantic_rows": len(semantic_rows),
        "mention_rows": len(mentions),
        "blind_review_rows": len(public_rows),
        "blind_review_rows_by_claim": dict(sorted(Counter(row["finding"] for row in review_rows).items())),
        "blind_review_rows_by_domain": dict(sorted(Counter(str(row["review_domain"]) for row in review_rows).items())),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
    }
    stats["artifact_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in final_names
        if name not in {"stats.json", "_COMPLETE.json"}
    }
    atomic_write(output_dir / "stats.json", json.dumps(stats, indent=2, sort_keys=True) + "\n")
    completion_hashes = {
        name: sha256_file(output_dir / name)
        for name in final_names
        if name != "_COMPLETE.json"
    }
    completion = {
        "version": VERSION,
        "status": "complete_discovery_source_only",
        "eligible_claim_count": len(eligible_set),
        "blind_review_rows": len(public_rows),
        "artifact_sha256": completion_hashes,
    }
    atomic_write(output_dir / "_COMPLETE.json", json.dumps(completion, indent=2, sort_keys=True) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=83017)
    parser.add_argument("--streaming-fallback", action="store_true")
    args = parser.parse_args()
    for path in (args.ontology, args.source_index, args.alignment, args.instruction):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = build(args)
    print(
        json.dumps(
            {
                "version": result["version"],
                "eligible_claims": result["eligibility_policy"]["eligible_claims"],
                "blind_review_rows": result["blind_review_rows"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
