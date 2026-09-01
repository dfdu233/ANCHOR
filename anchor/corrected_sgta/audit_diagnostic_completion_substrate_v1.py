#!/usr/bin/env python3
"""Audit a reader-grounded substrate for spontaneous diagnostic completion.

This is a label-aware *substrate* audit, not a hallucination evaluator.  It
looks only for patient-specific, within-sentence transitions in which a model
first states a radiographic observation and then adds a diagnostic impression.
The diagnosis is scored against the three independent VinDr image-level
reader labels.  Generic teaching text (for example, "may indicate conditions
such as ...") is deliberately excluded.

The audit answers two questions before any hidden-state experiment is run:

1. Does the frozen output contain repeated observation-to-diagnosis events
   whose diagnostic token is shared across reader-support states?
2. Does the exact R8/R9/R10 panel contain enough eligible images to justify a
   larger, outcome-independent native generation pass?

Absence of a VinDr diagnosis label is not treated as proof about laboratory or
clinical-context truth.  All claims are limited to the radiologists' image-only
diagnostic impressions recorded by VinDr-CXR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


VERSION = "diagnostic-completion-substrate-audit-v1"
FROZEN_PANEL = frozenset({"R8", "R9", "R10"})
DEFAULT_LABELS = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "annotations/image_labels_train.csv"
)
DEFAULT_GENERATIONS = Path(
    "/home/dbw/ANCHOR/corrected_runs/vindr_v2/"
    "clinical_presupposition_huatuo_generation_v1/generations.jsonl"
)


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: str
    parent_label: str
    child_label: str
    parent_pattern: re.Pattern[str]
    child_pattern: re.Pattern[str]


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


EDGE_SPECS = (
    EdgeSpec(
        "opacity_to_pneumonia",
        "Lung Opacity",
        "Pneumonia",
        _rx(r"\b(?:lung\s+|pulmonary\s+|airspace\s+)?opacit(?:y|ies)\b"),
        _rx(r"\b(?:pneumonia|pneumonic)\b"),
    ),
    EdgeSpec(
        "infiltration_to_pneumonia",
        "Infiltration",
        "Pneumonia",
        _rx(r"\b(?:pulmonary\s+)?infiltrat(?:e|es|ion|ions)\b"),
        _rx(r"\b(?:pneumonia|pneumonic)\b"),
    ),
    EdgeSpec(
        "consolidation_to_pneumonia",
        "Consolidation",
        "Pneumonia",
        _rx(r"\bconsolidat(?:ion|ions|ive)\b"),
        _rx(r"\b(?:pneumonia|pneumonic)\b"),
    ),
    EdgeSpec(
        "opacity_to_edema",
        "Lung Opacity",
        "Edema",
        _rx(r"\b(?:lung\s+|pulmonary\s+|airspace\s+)?opacit(?:y|ies)\b"),
        _rx(r"\b(?:pulmonary\s+)?edema\b"),
    ),
    EdgeSpec(
        "opacity_to_atelectasis",
        "Lung Opacity",
        "Atelectasis",
        _rx(r"\b(?:lung\s+|pulmonary\s+|airspace\s+)?opacit(?:y|ies)\b"),
        _rx(r"\batelecta(?:sis|tic)\b"),
    ),
    EdgeSpec(
        "nodule_mass_to_lung_tumor",
        "Nodule/Mass",
        "Lung tumor",
        _rx(r"\b(?:pulmonary\s+|lung\s+)?(?:nodule|mass)\b"),
        _rx(r"\b(?:lung\s+)?(?:tumou?r|neoplasm|malignan(?:cy|t))\b"),
    ),
)

# These connectors express a case-specific inference.  Broad educational
# continuations such as "may indicate underlying conditions such as" are not
# included, because they do not assert a diagnosis about the current image.
CONNECTOR = _rx(
    r"(?:suggest(?:s|ing|ive\s+of)?|consistent\s+with|compatible\s+with|"
    r"suspicious\s+for|concerning\s+for|indicative\s+of|"
    r"could\s+(?:represent|reflect|be\s+due\s+to)|"
    r"may\s+(?:represent|reflect)|likely\s+(?:represents?|reflects?|due\s+to)|"
    r"raise(?:s|d)?\s+concern\s+for)"
)
GENERIC_EXPLANATION = _rx(
    r"(?:underlying\s+(?:medical\s+)?conditions?|causes?)\s+such\s+as|"
    r"can\s+be\s+clinically\s+(?:relevant|significant)\s+as"
)
NEGATED_CHILD = _rx(
    r"(?:no\s+(?:radiographic\s+)?(?:evidence|signs?)\s+of|without|absence\s+of|"
    r"negative\s+for|not\s+(?:consistent|compatible)\s+with|unlikely\s+to\s+(?:be|represent))"
)
UNCERTAINTY = _rx(
    r"\b(?:possible|possibly|may|might|could|suspected|suspicious|concerning|"
    r"cannot\s+exclude|indeterminate|differential)\b"
)
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_exact_panel_votes(path: Path) -> dict[str, dict[str, int]]:
    required_labels = {spec.parent_label for spec in EDGE_SPECS} | {
        spec.child_label for spec in EDGE_SPECS
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_labels - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"VinDr label CSV is missing fields: {sorted(missing)}")
        for row in reader:
            grouped[str(row["image_id"])].append(row)

    votes: dict[str, dict[str, int]] = {}
    for image_id, rows in grouped.items():
        panel = {str(row["rad_id"]) for row in rows}
        if len(rows) != 3 or panel != FROZEN_PANEL:
            continue
        votes[image_id] = {
            label: sum(int(row[label]) for row in rows) for label in required_labels
        }
    return votes


def normalized_target(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def normalized_observation_prefix(sentence: str, parent_end: int) -> str:
    prefix = normalized_target(sentence[:parent_end])
    for boilerplate in (
        "the chest x ray shows ",
        "this chest x ray shows ",
        "the radiograph shows ",
        "this radiograph shows ",
        "there is ",
        "there are ",
    ):
        if prefix.startswith(boilerplate):
            return prefix[len(boilerplate) :]
    return prefix


def _child_status(sentence: str, child: re.Match[str]) -> str | None:
    left = sentence[max(0, child.start() - 70) : child.start()]
    if NEGATED_CHILD.search(left):
        return None
    window = sentence[max(0, child.start() - 90) : child.end() + 20]
    return "uncertain" if UNCERTAINTY.search(window) else "definite"


def extract_events(text: str) -> list[dict[str, str]]:
    """Extract strict patient-specific observation-to-diagnosis transitions."""

    events: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sentence_match in SENTENCE.finditer(text):
        sentence = sentence_match.group(0).strip()
        if not sentence or GENERIC_EXPLANATION.search(sentence):
            continue
        for spec in EDGE_SPECS:
            for parent in spec.parent_pattern.finditer(sentence):
                child = spec.child_pattern.search(sentence, parent.end())
                if child is None:
                    continue
                between = sentence[parent.end() : child.start()]
                connector = CONNECTOR.search(between)
                if connector is None:
                    continue
                status = _child_status(sentence, child)
                if status is None:
                    continue
                key = (spec.edge_id, normalized_target(child.group(0)))
                if key in seen:
                    continue
                seen.add(key)
                events.append(
                    {
                        "edge_id": spec.edge_id,
                        "parent_label": spec.parent_label,
                        "child_label": spec.child_label,
                        "parent_surface": parent.group(0),
                        "child_surface": child.group(0),
                        "target_key": key[1],
                        "observation_key": normalized_observation_prefix(
                            sentence, parent.end()
                        ),
                        "commitment": status,
                        "sentence": sentence,
                    }
                )
    return events


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "low": None, "high": None}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": p,
        "low": max(0.0, center - radius),
        "high": min(1.0, center + radius),
    }


def vote_role(parent_votes: int, child_votes: int) -> str:
    if parent_votes >= 1 and child_votes == 0:
        return "parent_seen_child_0of3"
    if parent_votes >= 1 and child_votes == 1:
        return "parent_seen_child_1of3"
    if parent_votes >= 1 and child_votes == 2:
        return "parent_seen_child_2of3"
    if parent_votes >= 1 and child_votes == 3:
        return "parent_seen_child_3of3"
    return "parent_0of3"


def _edge_lookup() -> dict[str, EdgeSpec]:
    return {spec.edge_id: spec for spec in EDGE_SPECS}


def audit(
    generations: Iterable[Mapping[str, Any]],
    votes: Mapping[str, Mapping[str, int]],
    *,
    condition: str,
    minimum_events_per_extreme: int,
    minimum_edge_types: int,
) -> dict[str, Any]:
    selected = [row for row in generations if str(row.get("prompt_condition")) == condition]
    if not selected:
        raise ValueError(f"no generations found for condition {condition!r}")
    if len({str(row["image_id"]) for row in selected}) != len(selected):
        raise ValueError("condition contains duplicate image IDs")

    specs = _edge_lookup()
    event_rows: list[dict[str, Any]] = []
    opportunity: Counter[tuple[str, str]] = Counter()
    observed: Counter[tuple[str, str]] = Counter()
    for row in selected:
        image_id = str(row["image_id"])
        if image_id not in votes:
            raise ValueError(f"generation image is outside exact R8/R9/R10 panel: {image_id}")
        for spec in EDGE_SPECS:
            pv = int(votes[image_id][spec.parent_label])
            cv = int(votes[image_id][spec.child_label])
            opportunity[(spec.edge_id, vote_role(pv, cv))] += 1
        for event in extract_events(str(row["text"])):
            spec = specs[event["edge_id"]]
            pv = int(votes[image_id][spec.parent_label])
            cv = int(votes[image_id][spec.child_label])
            role = vote_role(pv, cv)
            observed[(spec.edge_id, role)] += 1
            event_rows.append(
                {
                    "image_id": image_id,
                    "prompt_condition": condition,
                    **event,
                    "parent_votes": pv,
                    "child_votes": cv,
                    "reader_role": role,
                }
            )

    global_counts: Counter[tuple[str, str]] = Counter()
    generation_union: set[str] = set()
    for image_id, image_votes in votes.items():
        for spec in EDGE_SPECS:
            pv = int(image_votes[spec.parent_label])
            cv = int(image_votes[spec.child_label])
            role = vote_role(pv, cv)
            global_counts[(spec.edge_id, role)] += 1
            if pv >= 1:
                generation_union.add(image_id)

    edge_results: dict[str, Any] = {}
    eligible_edge_types: list[str] = []
    for spec in EDGE_SPECS:
        role_rows = {}
        for child_votes in range(4):
            role = f"parent_seen_child_{child_votes}of3"
            n = opportunity[(spec.edge_id, role)]
            k = observed[(spec.edge_id, role)]
            interval = wilson(k, n)
            global_n = global_counts[(spec.edge_id, role)]
            projected = None
            if interval["rate"] is not None:
                projected = {
                    "point": float(interval["rate"]) * global_n,
                    "low": float(interval["low"]) * global_n,
                    "high": float(interval["high"]) * global_n,
                }
            role_rows[role] = {
                "current_opportunities": n,
                "current_events": k,
                "current_event_rate_wilson95": interval,
                "global_eligible_images": global_n,
                "projected_events_if_all_eligible_generated": projected,
            }
        extreme_counts = (
            role_rows["parent_seen_child_0of3"]["current_events"],
            role_rows["parent_seen_child_3of3"]["current_events"],
        )
        repeated_extremes = all(value >= minimum_events_per_extreme for value in extreme_counts)
        if repeated_extremes:
            eligible_edge_types.append(spec.edge_id)
        edge_results[spec.edge_id] = {
            "parent_label": spec.parent_label,
            "child_label": spec.child_label,
            "reader_roles": role_rows,
            "current_extreme_event_gate": repeated_extremes,
        }

    target_overlap = Counter(
        (str(row["target_key"]), str(row["reader_role"])) for row in event_rows
    )
    target_summary: dict[str, dict[str, int]] = defaultdict(dict)
    for (target, role), count in sorted(target_overlap.items()):
        target_summary[target][role] = count

    gates = {
        "at_least_minimum_semantic_edges_have_repeated_0of3_and_3of3_events": len(
            eligible_edge_types
        )
        >= minimum_edge_types,
        "minimum_events_per_extreme": minimum_events_per_extreme,
        "minimum_edge_types": minimum_edge_types,
        "eligible_edge_types": eligible_edge_types,
    }
    gates["confirmatory_hidden_state_replay_authorized"] = bool(
        gates["at_least_minimum_semantic_edges_have_repeated_0of3_and_3of3_events"]
    )

    return {
        "version": VERSION,
        "scope": (
            "reader-grounded VinDr image-only diagnostic impression; not clinical-context "
            "or pathology truth and not a hallucination efficacy claim"
        ),
        "prompt_condition": condition,
        "generation_images": len(selected),
        "exact_reader_panel_images": len(votes),
        "strict_transition_events": len(event_rows),
        "strict_transition_images": len({row["image_id"] for row in event_rows}),
        "generation_union_parent_at_least_1of3": len(generation_union),
        "edge_results": edge_results,
        "target_key_role_overlap": dict(target_summary),
        "events": event_rows,
        "gates": gates,
        "next_action": (
            "Run an outcome-independent native generation pass over the frozen parent>=1/3 "
            "union before replay. Do not run confirmatory replay yet."
            if not gates["confirmatory_hidden_state_replay_authorized"]
            else "Freeze the repeated event manifest and open hidden-state replay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--condition", default="neutral")
    parser.add_argument("--minimum-events-per-extreme", type=int, default=12)
    parser.add_argument("--minimum-edge-types", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_events_per_extreme <= 0 or args.minimum_edge_types <= 0:
        raise ValueError("minimum gates must be positive")

    result = audit(
        load_jsonl(args.generations),
        load_exact_panel_votes(args.labels),
        condition=args.condition,
        minimum_events_per_extreme=args.minimum_events_per_extreme,
        minimum_edge_types=args.minimum_edge_types,
    )
    result["generations"] = str(args.generations.resolve())
    result["generations_sha256"] = sha256_file(args.generations)
    result["labels"] = str(args.labels.resolve())
    result["labels_sha256"] = sha256_file(args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
