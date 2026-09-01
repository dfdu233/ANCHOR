#!/usr/bin/env python3
"""Fail-closed validator for a future CECD OE/report transfer pack.

The validator checks provenance, patient isolation, clinical-truth strata,
the content-frozen teacher-forcing orbit, and (optionally) structural equality
of paired intervention outputs.  It never computes an efficacy endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "cecd-oe-report-transfer-manifest-v1"
PAIR_SCHEMA_VERSION = "cecd-oe-report-content-conservation-v1"
MODELS = {"huatuo", "hulu"}
TASKS = {"oe_abnormality_listing", "report"}
SPLIT_MIN_PATIENTS = {"dev": 80, "test": 120}
STATE_MINIMA = {
    "dev": {"supported": 40, "refuted": 20, "undetermined": 20},
    "test": {"supported": 60, "refuted": 30, "undetermined": 30},
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(RuntimeError):
    """A pack cannot support the preregistered transfer claim."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValidationError(f"{path}:{line_number}: row must be object")
            rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSONL {path}: {error}") from error
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_hash(value: Any, field: str) -> str:
    text = str(value or "")
    require(bool(HEX64.fullmatch(text)), f"{field} must be lowercase sha256")
    return text


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_claim(value: Any, where: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{where}: normalized_claim must be object")
    finding = str(value.get("finding", "")).strip()
    polarity = str(value.get("polarity", ""))
    uncertainty = str(value.get("uncertainty", ""))
    anatomy = value.get("anatomy")
    attributes = value.get("attributes")
    require(bool(finding), f"{where}: empty finding")
    require(polarity in {"present", "absent"}, f"{where}: invalid polarity")
    require(
        uncertainty in {"definite", "uncertain", "unknown"},
        f"{where}: invalid uncertainty",
    )
    require(anatomy is None or isinstance(anatomy, str), f"{where}: invalid anatomy")
    require(
        isinstance(attributes, list)
        and all(isinstance(item, str) and item.strip() for item in attributes),
        f"{where}: attributes must be non-empty strings",
    )
    return {
        "finding": finding,
        "polarity": polarity,
        "uncertainty": uncertainty,
        "anatomy": anatomy,
        "attributes": list(attributes),
    }


def content_signature(claim: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "finding": claim["finding"],
            "anatomy": claim["anatomy"],
            "attributes": claim["attributes"],
        }
    )


def validate_truth(value: Any, where: str, *, required_claim: bool = False) -> str:
    require(isinstance(value, Mapping), f"{where}: truth must be object")
    state = str(value.get("state", ""))
    relevance = str(value.get("relevance", ""))
    require(
        state in {"supported", "refuted", "undetermined", "unobservable"},
        f"{where}: invalid truth state",
    )
    require(
        relevance in {"required", "optional", "out_of_scope"},
        f"{where}: invalid relevance",
    )
    if required_claim:
        require(state == "supported", f"{where}: required claim must be supported")
        require(relevance == "required", f"{where}: required claim relevance mismatch")
    require(value.get("reviewer_count") == 2, f"{where}: require two initial reviewers")
    require(value.get("independent_reviews") is True, f"{where}: reviews not independent")
    require(value.get("source_model_blinded") is True, f"{where}: model not blinded")
    require(value.get("reference_hidden") is True, f"{where}: reference not hidden")
    require(value.get("image_inspected") is True, f"{where}: image not inspected")
    require(value.get("adjudicated") is True, f"{where}: truth not adjudicated")
    require(
        value.get("adjudicator_is_third_radiologist") is True,
        f"{where}: no independent radiologist adjudicator",
    )
    require_hash(value.get("truth_record_sha256"), f"{where}.truth_record_sha256")
    return state


def validate_admission(payload: Mapping[str, Any]) -> dict[str, Any]:
    admission = payload.get("equivalence_admission")
    require(isinstance(admission, Mapping), "equivalence_admission missing")
    require(admission.get("frozen_before_orbit_scores") is True, "late admission freeze")
    require(admission.get("outcome_blind") is True, "admission is not outcome-blind")
    require(admission.get("clinically_admitted") is True, "equivalence not admitted")
    require(admission.get("reviewer_count") == 2, "admission needs two reviewers")
    require(admission.get("independent_reviews") is True, "admission reviews not independent")
    require(admission.get("adjudicated") is True, "admission lacks adjudication")
    require(
        admission.get("adjudicator_is_third_radiologist") is True,
        "admission adjudicator is not a third radiologist",
    )
    require(int(admission.get("unique_images", 0)) >= 60, "admission has <60 images")
    require_hash(admission.get("artifact_sha256"), "equivalence_admission.artifact_sha256")

    orbit = payload.get("orbit_contract")
    require(isinstance(orbit, Mapping), "orbit_contract missing")
    renders = orbit.get("science_render_ids")
    prompts = orbit.get("science_prompt_ids")
    require(
        isinstance(renders, list)
        and len(renders) >= 3
        and len(set(renders)) == len(renders),
        "need >=3 unique admitted science renders",
    )
    require(
        isinstance(prompts, list)
        and len(prompts) >= 3
        and len(set(prompts)) == len(prompts),
        "need >=3 unique admitted science prompts",
    )
    require(
        set(renders) == set(admission.get("admitted_render_ids", [])),
        "orbit render IDs differ from admission",
    )
    require(
        set(prompts) == set(admission.get("admitted_prompt_ids", [])),
        "orbit prompt IDs differ from admission",
    )
    require(
        orbit.get("baseline_render_id") in renders,
        "baseline render must be one of the science renders",
    )
    require(bool(orbit.get("identity_render_id")), "identity render missing")
    require(
        orbit.get("identity_render_id") not in renders,
        "identity control cannot alias a science render ID",
    )
    require(bool(orbit.get("duplicate_prompt_id")), "duplicate prompt missing")
    require(
        orbit.get("duplicate_prompt_id") not in prompts,
        "duplicate control cannot alias a science prompt ID",
    )
    require(
        orbit.get("score_definition")
        == "length_normalized_teacher_forced_logprob",
        "wrong teacher-forcing score",
    )
    require(
        orbit.get("target_policy") == "exact_same_atomic_assertion_all_cells",
        "claim target is not fixed across product orbit",
    )
    require(
        orbit.get("draft_policy") == "generate_once_canonical_before_orbit",
        "draft must be generated once before orbit scoring",
    )
    expected_cells = len(renders) * len(prompts) + len(prompts) + 1
    require(
        orbit.get("complete_cells_per_claim") == expected_cells,
        "orbit cell count does not include full science grid and controls",
    )
    return {
        "science_renders": len(renders),
        "science_prompts": len(prompts),
        "complete_cells_per_claim": expected_cells,
    }


def validate_pack(payload: Any) -> dict[str, Any]:
    require(isinstance(payload, Mapping), "pack must be an object")
    require(payload.get("schema_version") == SCHEMA_VERSION, "wrong schema_version")
    require(payload.get("frozen_before_orbit_scores") is True, "pack frozen late")
    require(payload.get("efficacy_outputs_inspected") is False, "efficacy outcome leakage")
    require(set(payload.get("models", [])) == MODELS, "pack must contain Huatuo and Hulu")
    claimed_tasks = set(payload.get("claimed_tasks", []))
    require(bool(claimed_tasks) and claimed_tasks <= TASKS, "invalid claimed_tasks")
    orbit_summary = validate_admission(payload)

    cases = payload.get("cases")
    require(isinstance(cases, list) and cases, "cases must be non-empty list")
    case_ids: set[str] = set()
    patient_split: dict[str, str] = {}
    patient_image: dict[str, tuple[str, str]] = {}
    patient_task: set[tuple[str, str]] = set()
    patients: dict[tuple[str, str], set[str]] = defaultdict(set)
    truth_counts: Counter[tuple[str, str, str, str]] = Counter()
    required_counts: Counter[tuple[str, str]] = Counter()
    draft_counts: Counter[tuple[str, str, str]] = Counter()
    nonempty_counts: Counter[tuple[str, str, str]] = Counter()
    refusal_counts: Counter[tuple[str, str, str]] = Counter()
    cap_counts: Counter[tuple[str, str, str]] = Counter()
    answer_texts: Counter[tuple[str, str, str, str]] = Counter()
    draft_index: dict[tuple[str, str], dict[str, Any]] = {}

    for case_number, raw_case in enumerate(cases):
        where = f"cases[{case_number}]"
        require(isinstance(raw_case, Mapping), f"{where}: case must be object")
        case = dict(raw_case)
        case_id = str(case.get("case_id", ""))
        patient_id = str(case.get("patient_id", ""))
        study_id = str(case.get("study_id", ""))
        image_id = str(case.get("image_id", ""))
        split = str(case.get("split", ""))
        task = str(case.get("task", ""))
        require(case_id and case_id not in case_ids, f"{where}: duplicate/empty case_id")
        case_ids.add(case_id)
        require(patient_id and study_id and image_id, f"{where}: missing patient/study/image identity")
        require(split in SPLIT_MIN_PATIENTS, f"{where}: invalid split")
        require(task in claimed_tasks, f"{where}: undeclared task")
        require_hash(case.get("image_sha256"), f"{where}.image_sha256")
        require(case.get("required_set_reviewed") is True, f"{where}: omission universe not reviewed")
        require(
            patient_split.setdefault(patient_id, split) == split,
            f"patient {patient_id} leaks across dev/test",
        )
        identity = (study_id, image_id)
        require(
            patient_image.setdefault(patient_id, identity) == identity,
            f"patient {patient_id} has multiple studies/images in minimum pack",
        )
        require((patient_id, task) not in patient_task, f"duplicate patient/task {patient_id}/{task}")
        patient_task.add((patient_id, task))
        patients[(task, split)].add(patient_id)

        required_claims = case.get("required_claims")
        require(isinstance(required_claims, list), f"{where}: required_claims not list")
        required_ids: set[str] = set()
        for claim_number, raw_claim in enumerate(required_claims):
            claim_where = f"{where}.required_claims[{claim_number}]"
            require(isinstance(raw_claim, Mapping), f"{claim_where}: not object")
            claim_id = str(raw_claim.get("claim_id", ""))
            require(claim_id and claim_id not in required_ids, f"{claim_where}: duplicate/empty ID")
            required_ids.add(claim_id)
            normalized_claim(raw_claim.get("normalized_claim"), claim_where)
            validate_truth(raw_claim.get("truth"), claim_where, required_claim=True)
            required_counts[(task, split)] += 1

        drafts = case.get("drafts")
        require(isinstance(drafts, list), f"{where}: drafts not list")
        require(
            {str(draft.get("model_id", "")) for draft in drafts if isinstance(draft, Mapping)}
            == MODELS,
            f"{where}: drafts must contain exactly Huatuo and Hulu",
        )
        require(len(drafts) == len(MODELS), f"{where}: duplicate draft model")
        for draft_number, raw_draft in enumerate(drafts):
            draft_where = f"{where}.drafts[{draft_number}]"
            require(isinstance(raw_draft, Mapping), f"{draft_where}: not object")
            draft = dict(raw_draft)
            model = str(draft.get("model_id"))
            draft_id = str(draft.get("draft_id", ""))
            require(bool(draft_id), f"{draft_where}: empty draft_id")
            require(
                draft.get("generated_once_on_canonical_cell") is True,
                f"{draft_where}: draft not canonical-once",
            )
            require(
                draft.get("generated_before_orbit_scoring") is True,
                f"{draft_where}: draft generated after orbit outcome",
            )
            answer_text = str(draft.get("answer_text", ""))
            refused = draft.get("refused") is True
            cap_hit = draft.get("cap_hit") is True
            key = (model, task, split)
            draft_counts[key] += 1
            nonempty_counts[key] += int(bool(answer_text.strip()))
            refusal_counts[key] += int(refused)
            cap_counts[key] += int(cap_hit)
            answer_texts[(model, task, split, answer_text.strip())] += 1
            covered = draft.get("covered_required_claim_ids")
            require(
                isinstance(covered, list) and set(map(str, covered)) <= required_ids,
                f"{draft_where}: invalid required-coverage IDs",
            )

            claims = draft.get("claims")
            require(isinstance(claims, list), f"{draft_where}: claims not list")
            claim_ids: set[str] = set()
            slots: list[int] = []
            parsed_claims: list[dict[str, Any]] = []
            for claim_number, raw_claim in enumerate(claims):
                claim_where = f"{draft_where}.claims[{claim_number}]"
                require(isinstance(raw_claim, Mapping), f"{claim_where}: not object")
                claim_id = str(raw_claim.get("claim_id", ""))
                require(claim_id and claim_id not in claim_ids, f"{claim_where}: duplicate/empty ID")
                claim_ids.add(claim_id)
                try:
                    slot = int(raw_claim.get("slot"))
                except (TypeError, ValueError) as error:
                    raise ValidationError(f"{claim_where}: invalid slot") from error
                slots.append(slot)
                require(bool(str(raw_claim.get("text_span", "")).strip()), f"{claim_where}: empty text span")
                target = str(raw_claim.get("target_assertion", ""))
                require(bool(target.strip()), f"{claim_where}: empty target assertion")
                require(
                    hashlib.sha256(target.encode("utf-8")).hexdigest()
                    == require_hash(
                        raw_claim.get("target_assertion_sha256"),
                        f"{claim_where}.target_assertion_sha256",
                    ),
                    f"{claim_where}: target assertion hash mismatch",
                )
                claim = normalized_claim(raw_claim.get("normalized_claim"), claim_where)
                require(
                    raw_claim.get("content_signature") == content_signature(claim),
                    f"{claim_where}: content signature mismatch",
                )
                require(
                    raw_claim.get("claim_type") in {"visual", "knowledge", "unobservable"},
                    f"{claim_where}: invalid claim_type",
                )
                require(
                    raw_claim.get("commitment") in {"definite", "uncertain", "unknown"},
                    f"{claim_where}: invalid commitment",
                )
                atomization = raw_claim.get("atomization")
                require(isinstance(atomization, Mapping), f"{claim_where}: missing atomization")
                require(atomization.get("human_confirmed") is True, f"{claim_where}: automatic-only atom")
                require(atomization.get("source_model_blinded") is True, f"{claim_where}: atomizer unblinded")
                state = validate_truth(raw_claim.get("truth"), claim_where)
                if raw_claim.get("claim_type") == "visual" and state != "unobservable":
                    truth_counts[(model, task, split, state)] += 1
                parsed_claims.append({**claim, "claim_id": claim_id, "slot": slot})
            require(sorted(slots) == list(range(len(slots))), f"{draft_where}: slots not contiguous from zero")
            draft_index[(case_id, model)] = {
                "task": task,
                "split": split,
                "draft_id": draft_id,
                "answer_text": answer_text,
                "refused": refused,
                "cap_hit": cap_hit,
                "covered_required_claim_ids": sorted(map(str, covered)),
                "claims": sorted(parsed_claims, key=lambda row: row["slot"]),
            }

    for task in claimed_tasks:
        for split, minimum in SPLIT_MIN_PATIENTS.items():
            require(
                len(patients[(task, split)]) >= minimum,
                f"{task}/{split}: need >= {minimum} unique patients",
            )
            require(
                required_counts[(task, split)] >= minimum // 2,
                f"{task}/{split}: too few independently reviewed required claims",
            )
            for model in MODELS:
                key = (model, task, split)
                total = draft_counts[key]
                require(total >= minimum, f"{model}/{task}/{split}: incomplete drafts")
                require(nonempty_counts[key] / total >= 0.95, f"{model}/{task}/{split}: nonempty <0.95")
                require(refusal_counts[key] / total <= 0.05, f"{model}/{task}/{split}: refusal >0.05")
                require(cap_counts[key] / total <= 0.05, f"{model}/{task}/{split}: cap-hit >0.05")
                largest = max(
                    count
                    for (m, t, s, _), count in answer_texts.items()
                    if (m, t, s) == key
                )
                require(largest / total <= 0.20, f"{model}/{task}/{split}: template share >0.20")
                for state, state_minimum in STATE_MINIMA[split].items():
                    require(
                        truth_counts[(model, task, split, state)] >= state_minimum,
                        f"{model}/{task}/{split}/{state}: need >= {state_minimum} claims",
                    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "transfer_pack_structurally_admissible",
        "outcome_blind": True,
        "claimed_tasks": sorted(claimed_tasks),
        "cases": len(cases),
        "unique_patients": len(patient_split),
        "patient_counts": {
            task: {split: len(patients[(task, split)]) for split in SPLIT_MIN_PATIENTS}
            for task in sorted(claimed_tasks)
        },
        "orbit": orbit_summary,
        "truth_counts": {
            "/".join(key): value for key, value in sorted(truth_counts.items())
        },
        "formal_orbit_scoring_authorized": True,
        "intervention_authorized": False,
        "efficacy_claim_authorized": False,
        "draft_index": draft_index,
    }


def validate_pairs(pack_result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = pack_result["draft_index"]
    require(isinstance(index, Mapping), "internal draft index missing")
    expected = set(index)
    observed: set[tuple[str, str]] = set()
    max_length_delta = 0.0
    for number, row in enumerate(rows):
        where = f"pairs[{number}]"
        require(row.get("schema_version") == PAIR_SCHEMA_VERSION, f"{where}: wrong schema")
        key = (str(row.get("case_id", "")), str(row.get("model_id", "")))
        require(key in expected and key not in observed, f"{where}: unknown/duplicate pair")
        observed.add(key)
        baseline = index[key]
        require(row.get("draft_id") == baseline["draft_id"], f"{where}: draft mismatch")
        require(row.get("generated_with_content_lock") is True, f"{where}: no content lock")
        require(row.get("refused") is baseline["refused"], f"{where}: refusal changed")
        require(row.get("cap_hit") is baseline["cap_hit"], f"{where}: cap state changed")
        require(
            sorted(map(str, row.get("covered_required_claim_ids", [])))
            == baseline["covered_required_claim_ids"],
            f"{where}: omission coverage changed",
        )
        claims = row.get("claims")
        require(isinstance(claims, list), f"{where}: claims missing")
        require(len(claims) == len(baseline["claims"]), f"{where}: claim count/K changed")
        positive_before = sum(c["polarity"] == "present" for c in baseline["claims"])
        positive_after = 0
        for base, raw_claim in zip(baseline["claims"], claims):
            require(isinstance(raw_claim, Mapping), f"{where}: malformed intervention claim")
            claim = normalized_claim(raw_claim.get("normalized_claim"), where)
            require(str(raw_claim.get("claim_id")) == base["claim_id"], f"{where}: claim identity changed")
            require(int(raw_claim.get("slot")) == base["slot"], f"{where}: claim slot changed")
            require(content_signature(claim) == content_signature(base), f"{where}: clinical content changed")
            require(claim["polarity"] == base["polarity"], f"{where}: polarity changed")
            positive_after += int(claim["polarity"] == "present")
        require(positive_after == positive_before, f"{where}: positive K changed")
        original_words = len(str(baseline["answer_text"]).split())
        new_words = len(str(row.get("answer_text", "")).split())
        if original_words == 0:
            require(new_words == 0, f"{where}: empty refusal gained content")
            ratio = 1.0
        else:
            ratio = new_words / original_words
        require(0.90 <= ratio <= 1.10, f"{where}: length outside [0.90,1.10]")
        max_length_delta = max(max_length_delta, abs(ratio - 1.0))
    require(observed == expected, "paired intervention rows are incomplete")
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "status": "content_conservation_pass",
        "pairs": len(rows),
        "claim_identity_preserved": True,
        "polarity_preserved": True,
        "positive_k_preserved": True,
        "required_coverage_preserved": True,
        "refusal_and_cap_state_preserved": True,
        "length_ratio_within_0p90_1p10": True,
        "max_absolute_length_ratio_delta": max_length_delta,
        "efficacy_claim_authorized": False,
    }


def public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "draft_index"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intervention-pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        pack = validate_pack(load_json(args.manifest))
        result: dict[str, Any] = {"pack": public_result(pack)}
        if args.intervention_pairs:
            rows = load_jsonl(args.intervention_pairs)
            result["content_conservation"] = validate_pairs(pack, rows)
        else:
            result["content_conservation"] = {
                "status": "not_evaluated",
                "intervention_authorized": False,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValidationError(f"refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (ValidationError, OSError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"status": result["pack"]["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
