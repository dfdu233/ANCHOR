#!/usr/bin/env python3
"""Compile the physician-admitted Specificity Ratchet mechanism manifest.

No output is written unless the entire two-reviewer adjudication passes the
fail-closed validator.  Model text supplies candidate strings only; all
scientific roles below are functions of final physician adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
        AdjudicationValidationError,
        SOURCE_REQUIRED_STATES,
        validate_adjudication,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` entry point.
    from validate_specificity_ratchet_adjudication_v1 import (  # type: ignore[no-redef]
        AdjudicationValidationError,
        SOURCE_REQUIRED_STATES,
        validate_adjudication,
    )


MANIFEST_PROTOCOL_ID = "specificity-ratchet-mechanism-v1"
SPLIT_SEED = "specificity-ratchet-image-disjoint-v1"
PRIMARY_ERROR_STATES = {"refuted", "undetermined"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_constraint_spans(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every exact child-text occurrence introduced by the edge.

    Modifier proposals may introduce the same constraint at multiple positions
    (for example, two occurrences of ``right``).  Those are a frozen union of
    spans, never silently reduced to the easiest occurrence.
    """

    child = str(candidate["child_proposal"])
    proposal = str(candidate["added_constraint_proposal"])
    components = proposal.split(" | ") if candidate["edge_type"] in {"laterality", "size_morph"} else [proposal]
    spans: list[dict[str, Any]] = []
    for component in components:
        component = component.strip()
        if not component:
            raise ValueError(f"{candidate['edge_id']}: empty constraint component")
        pattern = re.compile(r"(?<!\w)" + re.escape(component) + r"(?!\w)", re.IGNORECASE)
        matches = list(pattern.finditer(child))
        if not matches:
            raise ValueError(
                f"{candidate['edge_id']}: constraint component is not an exact child span: {component!r}"
            )
        if candidate["edge_type"] in {"subtype", "etiology"} and len(matches) != 1:
            raise ValueError(
                f"{candidate['edge_id']}: clause constraint must have one exact occurrence"
            )
        for match in matches:
            exact = child[match.start() : match.end()]
            spans.append(
                {
                    "char_start": match.start(),
                    "char_end_exclusive": match.end(),
                    "text": exact,
                    "utf8_sha256": _sha256_bytes(exact.encode("utf-8")),
                }
            )
    spans.sort(key=lambda row: (row["char_start"], row["char_end_exclusive"]))
    for left, right in zip(spans, spans[1:]):
        if left["char_end_exclusive"] > right["char_start"]:
            raise ValueError(f"{candidate['edge_id']}: overlapping constraint spans")
    return spans


def exact_observed_child_span(candidate: dict[str, Any]) -> dict[str, Any]:
    """Bind the scored child to one exact span in the observed OE generation.

    The generated span is provenance for spontaneous occurrence only.  It may
    never define clinical support, which remains exclusively physician-admitted.
    """

    observed = str(candidate["answer_span"])
    child = str(candidate["child_proposal"])
    starts = [match.start() for match in re.finditer(re.escape(child), observed)]
    if len(starts) != 1:
        raise ValueError(
            f"{candidate['edge_id']}: child target must occur exactly once in observed answer span"
        )
    start = starts[0]
    return {
        "char_start": start,
        "char_end_exclusive": start + len(child),
        "utf8_sha256": _sha256_bytes(child.encode("utf-8")),
        "observed_answer_span_sha256": _sha256_bytes(observed.encode("utf-8")),
    }


def _scientific_role(final: dict[str, str]) -> tuple[str | None, str]:
    if final["final_edge_entailment_admitted"] != "yes":
        return None, "edge_not_admitted"
    if final["final_parent_visual_support"] != "supported":
        return None, "parent_not_visually_supported"
    child = final["final_child_visual_support"]
    source = final["final_increment_observability"]
    if child == "supported" and source == "observable_on_supplied_image":
        return "supported_specificity_control", "admitted_supported_child"
    if child in PRIMARY_ERROR_STATES and source == "observable_on_supplied_image":
        return "causal_escalation_error", f"admitted_child_{child}"
    if child == "unobservable" and source in SOURCE_REQUIRED_STATES:
        return "evidence_source_boundary", f"admitted_{source}"
    return None, "uncertain_or_incoherent_evidence_boundary"


def _case_feature_tokens(rows: list[dict[str, Any]]) -> Counter[str]:
    features: Counter[str] = Counter()
    features["case"] = 1
    for row in rows:
        for prefix, value in (
            ("role", row["scientific_role"]),
            ("edge", row["edge_type"]),
            ("modality", row["modality_stratum"]),
            ("anatomy", row["anatomy_stratum"]),
        ):
            features[f"{prefix}:{value}"] += 1
    return features


def assign_grouped_splits(
    rows: list[dict[str, Any]], *, seed: str = SPLIT_SEED
) -> dict[str, str]:
    """Greedily balance mechanism roles/strata while grouping entire images."""

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    if len(by_case) < 2:
        raise ValueError("at least two admitted image groups are required for dev/test")
    feature_by_case = {case: _case_feature_tokens(items) for case, items in by_case.items()}
    totals: Counter[str] = Counter()
    for features in feature_by_case.values():
        totals.update(features)

    def rarity(case: str) -> float:
        return sum(value / totals[key] for key, value in feature_by_case[case].items())

    def tie_hash(case: str) -> str:
        return hashlib.sha256(f"{seed}|{case}".encode()).hexdigest()

    order = sorted(
        by_case,
        key=lambda case: (-len(by_case[case]), -rarity(case), tie_hash(case)),
    )
    assigned_total: Counter[str] = Counter()
    test_total: Counter[str] = Counter()
    assignment: dict[str, str] = {}
    for case in order:
        incoming = feature_by_case[case]
        next_total = assigned_total + incoming

        def cost(send_to_test: bool) -> float:
            proposed = test_total + incoming if send_to_test else test_total
            return sum(
                abs(proposed[key] - 0.5 * next_total[key]) / max(1.0, totals[key])
                for key in totals
            )

        dev_cost, test_cost = cost(False), cost(True)
        if test_cost == dev_cost:
            send_to_test = int(tie_hash(case), 16) % 2 == 0
        else:
            send_to_test = test_cost < dev_cost
        assignment[case] = "test" if send_to_test else "dev"
        assigned_total.update(incoming)
        if send_to_test:
            test_total.update(incoming)
    if set(assignment.values()) != {"dev", "test"}:
        raise ValueError("grouped split assignment collapsed to one split")
    return assignment


def compile_rows(validated: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    scientific: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for candidate in validated.candidates:
        edge_id = candidate["edge_id"]
        final = validated.final_rows[edge_id]
        role, reason = _scientific_role(final)
        if role is None:
            exclusions.append({"case_id": candidate["case_id"], "edge_id": edge_id, "reason": reason})
            continue
        spans = exact_constraint_spans(candidate)
        observed_child = exact_observed_child_span(candidate)
        scientific.append(
            {
                "manifest_protocol_id": MANIFEST_PROTOCOL_ID,
                "sample_id": "SRM1-" + hashlib.sha256(edge_id.encode()).hexdigest()[:16],
                "case_id": candidate["case_id"],
                "edge_id": edge_id,
                "image_relpath": candidate["image_relpath"],
                "question": candidate["question"],
                "parent_target": candidate["parent_proposal"],
                "child_target": candidate["child_proposal"],
                "constraint_char_spans_in_child": spans,
                "child_target_span_in_observed_generation": observed_child,
                "child_target_exact_observed_substring": True,
                "observed_generation_role": (
                    "spontaneous-occurrence provenance only; never clinical truth"
                ),
                "constraint_occurrences": len(spans),
                "constraint_whitespace_tokens": sum(len(span["text"].split()) for span in spans),
                "parent_whitespace_tokens": len(candidate["parent_proposal"].split()),
                "child_whitespace_tokens": len(candidate["child_proposal"].split()),
                "edge_type": candidate["edge_type"],
                "modality_stratum": candidate["modality_stratum"],
                "anatomy_stratum": candidate["anatomy_stratum"],
                "prompt_requested_increment": candidate["prompt_requested_increment"],
                "scientific_role": role,
                "adjudicated_parent_visual_support": final["final_parent_visual_support"],
                "adjudicated_child_visual_support": final["final_child_visual_support"],
                "adjudicated_increment_observability": final["final_increment_observability"],
                "mitigation_nearest_ancestor": candidate["parent_proposal"],
                "mitigation_claim_count_delta": 0,
            }
        )
    if not scientific:
        raise ValueError("no physician-admitted, supported-parent scientific samples")
    assignment = assign_grouped_splits(scientific)
    for row in scientific:
        row["split"] = assignment[row["case_id"]]
    dev_images = {row["case_id"] for row in scientific if row["split"] == "dev"}
    test_images = {row["case_id"] for row in scientific if row["split"] == "test"}
    if dev_images & test_images:
        raise AssertionError("image leakage across grouped splits")
    return scientific, exclusions


def _atomic_write(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compile_manifest(
    pack: Path,
    output: Path,
    metadata_output: Path,
    attestations: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate everything first; only then atomically emit both artifacts."""

    validated = validate_adjudication(pack, attestations)
    rows, exclusions = compile_rows(validated)
    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    role_counts = Counter(row["scientific_role"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    split_image_counts = Counter()
    for split in ("dev", "test"):
        split_image_counts[split] = len({row["case_id"] for row in rows if row["split"] == split})
    metadata = {
        "manifest_protocol_id": MANIFEST_PROTOCOL_ID,
        "status": "physician_admitted",
        "claim_ceiling": "causal child-over-parent specificity escalation on admitted OE edges",
        "n_scientific_edges": len(rows),
        "n_excluded_edges": len(exclusions),
        "scientific_role_counts": dict(sorted(role_counts.items())),
        "split_edge_counts": dict(sorted(split_counts.items())),
        "split_image_counts": dict(sorted(split_image_counts.items())),
        "image_disjoint": True,
        "split_seed": SPLIT_SEED,
        "input_sha256": validated.input_sha256,
        "manifest_sha256": _sha256_bytes(jsonl),
        "estimands": {
            "primary_teacher_forcing": (
                "At every decoder layer, compare mean log probability on the exact union of "
                "added-constraint token spans in child_target against the same-image "
                "parent_target token mean. Fit the child-minus-parent contrast with "
                "scientific_role interaction on dev only; test coefficients once on test."
            ),
            "observed_generation_anchor": (
                "child_target is a unique exact UTF-8 span of the frozen OE generation. "
                "This anchors spontaneous occurrence only; physicians alone define support."
            ),
            "token_boundary_rule": (
                "The runtime tokenizer must map offsets back to every frozen UTF-8 span; "
                "score their union only. Boundary mismatch is an error, never sentence fallback."
            ),
            "length_frequency_controls": (
                "Pre-register target token count and text-only mean token NLL for parent and "
                "constraint spans as nuisance covariates. Text-only NLL is a lexical-frequency "
                "control, not clinical evidence."
            ),
            "image_null_secondary_only": (
                "Image-null and image-swap contrasts are sensitivity analyses after the "
                "same-image parent-controlled primary result; they cannot admit the mechanism."
            ),
            "mitigation": (
                "For each exact matched unsupported child, replace it one-for-one with its "
                "physician-admitted nearest parent. Hold clinical claim count K fixed; no "
                "deletion, refusal, extra hedge, or ontology insertion is credited."
            ),
        },
        "exclusions": exclusions,
        "truth_prohibitions": [
            "model outputs",
            "VQA-RAD reference answers",
            "LLM judges",
            "RadGraph",
            "cross-model agreement",
        ],
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    # Validate both destinations before the first mutation.
    for destination in (output, metadata_output):
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    _atomic_write(output, jsonl, overwrite=overwrite)
    try:
        _atomic_write(metadata_output, metadata_bytes, overwrite=overwrite)
    except Exception:
        # The manifest is scientifically unusable without its contract.  Remove
        # only the file created by this invocation.
        output.unlink(missing_ok=True)
        raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"),
    )
    parser.add_argument("--attestations", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/mechanism_manifest_v1/samples.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/mechanism_manifest_v1/metadata.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = compile_manifest(
            args.pack,
            args.output,
            args.metadata_output,
            attestations=args.attestations,
            overwrite=args.overwrite,
        )
    except (AdjudicationValidationError, ValueError, FileExistsError) as exc:
        issues = exc.issues if isinstance(exc, AdjudicationValidationError) else [str(exc)]
        print(json.dumps({"status": "refused", "issues": issues}, indent=2))
        raise SystemExit(2)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
