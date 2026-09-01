#!/usr/bin/env python3
"""Fail-closed CPU audit of the VinDr substrate for AR-SoS.

AR-SoS requires a target image with two independently unanimous findings
(``A=3/3`` and ``B=3/3``) and a third unanimous-negative finding ``C=0/3``.
The A--C language-prior association is fitted *only* on the pre-existing dev
split.  Confirmation labels are subsequently opened only to count whether the
already frozen triples have enough target images; they never rank associations
or triples.  This program does not load a model, invent clinical phrases, or
assign truth with text matching/LLMs.

The original decisive design called for six A--B pairs with forty confirmation
images per pair.  A scientific manifest is deliberately not implemented here:
if the audit fails, the only valid artifact is the no-go audit.  If a future
substrate passes, a separate versioned builder must implement unique-image
allocation and the prefix-admission contract before any model run is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from corrected_sgta.clinical_claims import normalize_term
from corrected_sgta.prepare_vindr_reader_manifest import (
    build_records,
    load_ontology_findings,
    read_votes,
    select_ontology_columns,
    sha256_file,
    stable_key,
)
from corrected_sgta.prepare_vindr_reader_manifest_v2 import (
    fixed_panel_records,
    three_way_split,
)


PROTOCOL_ID = "ar-sos-vindr-substrate-audit-v1"
FROZEN_PANEL = ("R8", "R9", "R10")
FROZEN_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
FROZEN_SEED = 42
FIT_SPLIT = "dev"
LOCKED_SPLIT = "confirmation"

# Frozen before looking at confirmation target availability.  The Bonferroni
# family contains all 8*7 ordered A->C tests.
MIN_DEV_AC_JOINT_POSITIVE = 10
MIN_DEV_AC_LIFT = 2.0
MAX_BONFERRONI_P = 0.05
MIN_DEV_TARGETS_PER_AB_PAIR = 8
MINIMUM_SCIENTIFIC_AB_PAIRS = 4
ORIGINAL_DESIGN_AB_PAIRS = 6
MIN_CONFIRMATION_TARGETS_PER_PAIR = 40


class SubstrateError(RuntimeError):
    """The independent-reader substrate violates a frozen data contract."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value for positive A--C association."""

    if min(a, b, c, d) < 0:
        raise ValueError("contingency counts must be non-negative")
    population = a + b + c + d
    row_one = a + b
    col_one = a + c
    if population == 0:
        return 1.0
    upper = min(row_one, col_one)
    log_denominator = _log_choose(population, col_one)
    terms = [
        _log_choose(row_one, x)
        + _log_choose(population - row_one, col_one - x)
        - log_denominator
        for x in range(a, upper + 1)
    ]
    pivot = max(terms)
    return min(1.0, math.exp(pivot) * sum(math.exp(value - pivot) for value in terms))


def vectorize_rows(
    rows: Iterable[Mapping[str, Any]],
    findings: Sequence[str] = FROZEN_FINDINGS,
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """Create one exact eight-dimensional vote vector per image."""

    expected = set(findings)
    vectors: dict[str, dict[str, int]] = defaultdict(dict)
    splits: dict[str, str] = {}
    for row in rows:
        image_id = str(row["image_id"])
        finding = str(row["finding"])
        if finding not in expected:
            continue
        if finding in vectors[image_id]:
            raise SubstrateError(f"duplicate image/finding row: {image_id}/{finding}")
        panel = tuple(str(value) for value in row.get("reader_panel", FROZEN_PANEL))
        if panel != FROZEN_PANEL:
            raise SubstrateError(f"non-frozen reader panel for {image_id}/{finding}: {panel}")
        votes = int(row["positive_votes"])
        if votes not in {0, 1, 2, 3}:
            raise SubstrateError(f"invalid vote count for {image_id}/{finding}: {votes}")
        vectors[image_id][finding] = votes
        split = str(row.get("experiment_split") or three_way_split(image_id, FROZEN_SEED))
        previous = splits.setdefault(image_id, split)
        if previous != split:
            raise SubstrateError(f"image crosses splits: {image_id}")
    incomplete = {
        image_id: sorted(expected - set(vector))
        for image_id, vector in vectors.items()
        if set(vector) != expected
    }
    if incomplete:
        image_id = sorted(incomplete)[0]
        raise SubstrateError(f"incomplete eight-finding vector {image_id}: {incomplete[image_id]}")
    if not vectors:
        raise SubstrateError("no exact-panel image vectors")
    return dict(vectors), splits


def association_rows(
    vectors: Mapping[str, Mapping[str, int]],
    splits: Mapping[str, str],
    *,
    fit_split: str = FIT_SPLIT,
    findings: Sequence[str] = FROZEN_FINDINGS,
) -> list[dict[str, Any]]:
    """Fit every ordered A->C association without reading confirmation rows."""

    fit_ids = sorted(image for image, split in splits.items() if split == fit_split)
    family_size = len(findings) * (len(findings) - 1)
    output: list[dict[str, Any]] = []
    for finding_a in findings:
        for finding_c in findings:
            if finding_a == finding_c:
                continue
            counts = Counter(
                (vectors[image][finding_a], vectors[image][finding_c])
                for image in fit_ids
                if vectors[image][finding_a] in {0, 3}
                and vectors[image][finding_c] in {0, 3}
            )
            n11 = counts[(3, 3)]
            n10 = counts[(3, 0)]
            n01 = counts[(0, 3)]
            n00 = counts[(0, 0)]
            probability_given_a = (n11 + 0.5) / (n11 + n10 + 1.0)
            probability_given_not_a = (n01 + 0.5) / (n01 + n00 + 1.0)
            p_value = fisher_greater(n11, n10, n01, n00)
            bonferroni = min(1.0, family_size * p_value)
            log_odds = math.log(
                ((n11 + 0.5) * (n00 + 0.5))
                / ((n10 + 0.5) * (n01 + 0.5))
            )
            lift = probability_given_a / probability_given_not_a
            admitted = (
                n11 >= MIN_DEV_AC_JOINT_POSITIVE
                and lift >= MIN_DEV_AC_LIFT
                and bonferroni <= MAX_BONFERRONI_P
            )
            output.append(
                {
                    "finding_a": finding_a,
                    "finding_c": finding_c,
                    "fit_split": fit_split,
                    "n_a3_c3": n11,
                    "n_a3_c0": n10,
                    "n_a0_c3": n01,
                    "n_a0_c0": n00,
                    "haldane_log_odds_ratio": log_odds,
                    "smoothed_lift": lift,
                    "fisher_greater_p": p_value,
                    "bonferroni_p_56": bonferroni,
                    "association_admitted": admitted,
                }
            )
    output.sort(
        key=lambda row: (
            not row["association_admitted"],
            -int(row["n_a3_c3"]),
            -float(row["haldane_log_odds_ratio"]),
            str(row["finding_a"]),
            str(row["finding_c"]),
        )
    )
    return output


def freeze_ab_pairs_on_dev(
    vectors: Mapping[str, Mapping[str, int]],
    splits: Mapping[str, str],
    associations: Sequence[Mapping[str, Any]],
    *,
    required_pairs: int = ORIGINAL_DESIGN_AB_PAIRS,
    min_dev_targets: int = MIN_DEV_TARGETS_PER_AB_PAIR,
    findings: Sequence[str] = FROZEN_FINDINGS,
) -> list[dict[str, Any]]:
    """Rank A/B/C triples using dev data only and freeze distinct A--B pairs."""

    fit_ids = sorted(image for image, split in splits.items() if split == FIT_SPLIT)
    candidates: list[dict[str, Any]] = []
    for association in associations:
        if not association["association_admitted"]:
            continue
        finding_a = str(association["finding_a"])
        finding_c = str(association["finding_c"])
        for finding_b in findings:
            if finding_b in {finding_a, finding_c}:
                continue
            target_ids = [
                image
                for image in fit_ids
                if vectors[image][finding_a] == 3
                and vectors[image][finding_b] == 3
                and vectors[image][finding_c] == 0
            ]
            candidates.append(
                {
                    "finding_a": finding_a,
                    "finding_b": finding_b,
                    "finding_c": finding_c,
                    "dev_target_count": len(target_ids),
                    "dev_target_id_digest": hashlib.sha256(
                        "\n".join(target_ids).encode()
                    ).hexdigest(),
                    "association_dev_n_a3_c3": int(association["n_a3_c3"]),
                    "association_dev_lift": float(association["smoothed_lift"]),
                    "association_dev_bonferroni_p": float(association["bonferroni_p_56"]),
                }
            )
    candidates.sort(
        key=lambda row: (
            -int(row["dev_target_count"]),
            -int(row["association_dev_n_a3_c3"]),
            -float(row["association_dev_lift"]),
            stable_key(
                FROZEN_SEED,
                "ar-sos-dev-pair-rank-v1",
                str(row["finding_a"]),
                str(row["finding_b"]),
                str(row["finding_c"]),
            ),
        )
    )
    frozen: list[dict[str, Any]] = []
    used_ab: set[tuple[str, str]] = set()
    for row in candidates:
        if int(row["dev_target_count"]) < min_dev_targets:
            continue
        ab = (str(row["finding_a"]), str(row["finding_b"]))
        if ab in used_ab:
            continue
        frozen.append(dict(row))
        used_ab.add(ab)
        if len(frozen) == required_pairs:
            break
    return frozen


def confirmation_availability(
    frozen_pairs: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, int]],
    splits: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Count locked-split targets after pair identities have been frozen."""

    locked_ids = sorted(image for image, split in splits.items() if split == LOCKED_SPLIT)
    output: list[dict[str, Any]] = []
    for source in frozen_pairs:
        row = dict(source)
        target_ids = [
            image
            for image in locked_ids
            if vectors[image][str(row["finding_a"])] == 3
            and vectors[image][str(row["finding_b"])] == 3
            and vectors[image][str(row["finding_c"])] == 0
        ]
        row.update(
            {
                "confirmation_target_count": len(target_ids),
                "confirmation_target_id_digest": hashlib.sha256(
                    "\n".join(target_ids).encode()
                ).hexdigest(),
                "meets_40_image_gate": len(target_ids) >= MIN_CONFIRMATION_TARGETS_PER_PAIR,
                "confirmation_used_for_pair_ranking": False,
            }
        )
        output.append(row)
    return output


def cardinality_envelope(
    vectors: Mapping[str, Mapping[str, int]],
    splits: Mapping[str, str],
    associations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove a sample-size no-go without using confirmation to select a pair.

    This enumerates all six possible B findings for every dev-admitted A->C
    association.  Confirmation counts are descriptive upper bounds only.  They
    cannot nominate a formal pair and are never joined to a model outcome.
    """

    split_ids = {
        split: sorted(image for image, value in splits.items() if value == split)
        for split in (FIT_SPLIT, LOCKED_SPLIT)
    }
    rows: list[dict[str, Any]] = []
    for association in associations:
        if not association["association_admitted"]:
            continue
        finding_a = str(association["finding_a"])
        finding_c = str(association["finding_c"])
        for finding_b in FROZEN_FINDINGS:
            if finding_b in {finding_a, finding_c}:
                continue
            counts = {}
            for split, image_ids in split_ids.items():
                counts[split] = sum(
                    vectors[image][finding_a] == 3
                    and vectors[image][finding_b] == 3
                    and vectors[image][finding_c] == 0
                    for image in image_ids
                )
            rows.append(
                {
                    "finding_a": finding_a,
                    "finding_b": finding_b,
                    "finding_c": finding_c,
                    "dev_target_count": counts[FIT_SPLIT],
                    "confirmation_target_count": counts[LOCKED_SPLIT],
                }
            )
    rows.sort(
        key=lambda row: (
            -int(row["confirmation_target_count"]),
            -int(row["dev_target_count"]),
            str(row["finding_a"]),
            str(row["finding_b"]),
            str(row["finding_c"]),
        )
    )
    return {
        "purpose": "cardinality upper bound only; never formal pair selection",
        "strong_association_abc_combinations": len(rows),
        "combinations_with_dev_at_least_8": sum(
            int(row["dev_target_count"]) >= MIN_DEV_TARGETS_PER_AB_PAIR for row in rows
        ),
        "combinations_with_confirmation_at_least_40": sum(
            int(row["confirmation_target_count"]) >= MIN_CONFIRMATION_TARGETS_PER_PAIR
            for row in rows
        ),
        "maximum_confirmation_target_count": max(
            (int(row["confirmation_target_count"]) for row in rows), default=0
        ),
        "all_combinations": rows,
    }


def audit_vectors(
    vectors: Mapping[str, Mapping[str, int]],
    splits: Mapping[str, str],
) -> dict[str, Any]:
    associations = association_rows(vectors, splits)
    frozen = freeze_ab_pairs_on_dev(vectors, splits, associations)
    checked = confirmation_availability(frozen, vectors, splits)
    passing = [row for row in checked if row["meets_40_image_gate"]]
    envelope = cardinality_envelope(vectors, splits, associations)
    split_counts = Counter(splits.values())
    prevalence = {
        split: {
            finding: {
                "0/3": sum(
                    splits[image] == split and vector[finding] == 0
                    for image, vector in vectors.items()
                ),
                "3/3": sum(
                    splits[image] == split and vector[finding] == 3
                    for image, vector in vectors.items()
                ),
            }
            for finding in FROZEN_FINDINGS
        }
        for split in (FIT_SPLIT, LOCKED_SPLIT)
    }
    reasons = []
    reasons.append(
        "F6: VinDr supplies finding votes but no native report wording; the same-support-other-image prefix has no admitted semantic stimulus without independent physician review"
    )
    if len(frozen) < ORIGINAL_DESIGN_AB_PAIRS:
        reasons.append(
            f"dev froze only {len(frozen)}/{ORIGINAL_DESIGN_AB_PAIRS} original-design distinct A-B pairs "
            f"with at least {MIN_DEV_TARGETS_PER_AB_PAIR} A3/B3/C0 images"
        )
    if len(passing) < MINIMUM_SCIENTIFIC_AB_PAIRS:
        reasons.append(
            f"confirmation has only {len(passing)}/{MINIMUM_SCIENTIFIC_AB_PAIRS} minimum scientific pairs "
            f"with at least {MIN_CONFIRMATION_TARGETS_PER_PAIR} A3/B3/C0 images"
        )
    status = "admitted_for_separate_manifest_builder" if not reasons else "no_go"
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "truth_contract": (
            "A/B/C states are exact independent R8/R9/R10 vote sums only; no generated "
            "phrase, regex, automatic labeler, or LLM defines clinical truth"
        ),
        "fatal_flaw_mapping": {
            "F6_construct_admission": {
                "status": "failed",
                "reason": (
                    "VinDr has no native report sentences. Model/automatic wording may propose "
                    "a stimulus but cannot certify that supported-A, A-prime, neutral, and donor "
                    "prefixes contain only their intended propositions; two independent clinical "
                    "reviewers are absent."
                ),
            },
            "F7_substrate_cardinality": {
                "status": "failed",
                "reason": (
                    "No dev-frozen pair reaches forty locked-confirmation A3/B3/C0 images; "
                    "the exhaustive strong-association maximum is 39."
                ),
            },
        },
        "split_contract": {
            "cooccurrence_fit_split": FIT_SPLIT,
            "formal_test_split": LOCKED_SPLIT,
            "global_image_hash_seed": FROZEN_SEED,
            "confirmation_used_for_association_fit_or_pair_ranking": False,
            "confirmation_label_access": "availability count after dev identities frozen only",
        },
        "image_count": len(vectors),
        "split_image_counts": dict(sorted(split_counts.items())),
        "finding_prevalence": prevalence,
        "association_gate": {
            "ordered_family_size": len(FROZEN_FINDINGS) * (len(FROZEN_FINDINGS) - 1),
            "minimum_dev_a3_c3": MIN_DEV_AC_JOINT_POSITIVE,
            "minimum_smoothed_lift": MIN_DEV_AC_LIFT,
            "maximum_bonferroni_p": MAX_BONFERRONI_P,
            "admitted_count": sum(row["association_admitted"] for row in associations),
            "admitted": [row for row in associations if row["association_admitted"]],
        },
        "pair_gate": {
            "minimum_scientific_distinct_ordered_ab_pairs": MINIMUM_SCIENTIFIC_AB_PAIRS,
            "original_design_distinct_ordered_ab_pairs": ORIGINAL_DESIGN_AB_PAIRS,
            "minimum_dev_targets_per_pair": MIN_DEV_TARGETS_PER_AB_PAIR,
            "minimum_confirmation_targets_per_pair": MIN_CONFIRMATION_TARGETS_PER_PAIR,
            "dev_frozen_pair_count": len(frozen),
            "confirmation_passing_pair_count": len(passing),
            "dev_frozen_then_confirmation_counted": checked,
        },
        "cardinality_envelope": envelope,
        "manifest_emitted": False,
        "no_go_reasons": reasons,
        "next_action": (
            "Do not run a VLM on this substrate. Expand the independently labelled finding "
            "ontology or use a larger multi-label source, then rerun the same frozen audit."
            if reasons
            else "Implement a new fail-closed manifest builder with unique target/donor images and physician-admitted prefix wording."
        ),
    }


def load_exact_panel_vectors(labels_csv: Path, ontology: Path) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    votes, source_findings, _, _ = read_votes(labels_csv)
    selected, _ = select_ontology_columns(source_findings, load_ontology_findings(ontology))
    selected = [name for name in selected if normalize_term(name) in set(FROZEN_FINDINGS)]
    if {normalize_term(name) for name in selected} != set(FROZEN_FINDINGS):
        raise SubstrateError("source/ontology does not contain the frozen eight findings")
    records, _ = build_records(votes, selected, "local-only")
    fixed = fixed_panel_records(records, FROZEN_PANEL)
    for row in fixed:
        row["experiment_split"] = three_way_split(str(row["image_id"]), FROZEN_SEED)
        row["reader_panel"] = list(FROZEN_PANEL)
    return vectorize_rows(fixed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vectors, splits = load_exact_panel_vectors(args.labels_csv, args.ontology)
    audit = audit_vectors(vectors, splits)
    audit["inputs"] = {
        "labels_csv": str(args.labels_csv.resolve()),
        "labels_csv_sha256": sha256_file(args.labels_csv),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "reader_panel": list(FROZEN_PANEL),
        "findings": list(FROZEN_FINDINGS),
    }
    _atomic_json(args.output, audit)
    print(json.dumps({
        "status": audit["status"],
        "image_count": audit["image_count"],
        "admitted_ac": audit["association_gate"]["admitted_count"],
        "dev_frozen_pairs": audit["pair_gate"]["dev_frozen_pair_count"],
        "confirmation_passing_pairs": audit["pair_gate"]["confirmation_passing_pair_count"],
        "output": str(args.output),
    }, sort_keys=True))
    raise SystemExit(0 if audit["status"] != "no_go" else 2)


if __name__ == "__main__":
    main()
