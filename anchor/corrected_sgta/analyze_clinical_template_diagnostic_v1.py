#!/usr/bin/env python3
"""Reader-linked, fail-closed diagnostic for prompt-conditioned templates.

This analysis never parses generated clinical claims.  It asks only whether
exact/prefix template identity varies with an independently fixed reader-vote
bin.  It cannot establish clinical correctness or a hidden-state mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from corrected_sgta.run_clinical_presupposition_generation_v1 import (
    FROZEN_FINDINGS,
    full_fixed_panel_universe,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    atomic_json,
    dicom_to_pil,
    sha256_file,
)


VERSION = "clinical-template-reader-diagnostic-v1"
CONDITIONS = ("neutral", "existential", "negative_obligation")


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _t_fraction(counter: Counter[Any], fraction: float) -> int:
    threshold = fraction * sum(counter.values())
    cumulative = 0
    for rank, count in enumerate(sorted(counter.values(), reverse=True), 1):
        cumulative += count
        if cumulative >= threshold:
            return rank
    raise ValueError("empty template counter")


def concentration(keys: Iterable[Any]) -> dict[str, Any]:
    counts = Counter(keys)
    n = sum(counts.values())
    _require(n > 0, "template concentration needs observations")
    top_key, top_count = counts.most_common(1)[0]
    return {
        "n": n,
        "unique": len(counts),
        "unique_rate": len(counts) / n,
        "top1_count": top_count,
        "top1_share": top_count / n,
        "t80": _t_fraction(counts, 0.8),
        "top1_key": list(top_key) if isinstance(top_key, tuple) else top_key,
    }


def mutual_information(templates: list[str], bins: list[int]) -> float:
    _require(len(templates) == len(bins) > 0, "MI inputs are empty or misaligned")
    joint = Counter(zip(templates, bins))
    template_counts = Counter(templates)
    bin_counts = Counter(bins)
    n = len(templates)
    return float(
        sum(
            (count / n)
            * math.log((count * n) / (template_counts[template] * bin_counts[vote]))
            for (template, vote), count in joint.items()
        )
    )


def extreme_collision_rate(templates: list[str], bins: list[int]) -> float | None:
    zero = Counter(template for template, vote in zip(templates, bins) if vote == 0)
    three = Counter(template for template, vote in zip(templates, bins) if vote == 3)
    denominator = sum(zero.values()) * sum(three.values())
    if denominator == 0:
        return None
    return sum(count * three[template] for template, count in zero.items()) / denominator


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    _require(pvalues and all(0 <= value <= 1 for value in pvalues), "invalid p-values")
    order = sorted(range(len(pvalues)), key=lambda index: (pvalues[index], index))
    adjusted = [1.0] * len(pvalues)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), 1):
        rank = len(pvalues) - reverse_rank + 1
        running = min(running, pvalues[index] * len(pvalues) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def association(
    templates: list[str],
    bins: list[int],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    observed = mutual_information(templates, bins)
    rng = np.random.default_rng(seed)
    bin_array = np.asarray(bins, dtype=int)
    permuted = [
        mutual_information(templates, rng.permutation(bin_array).tolist())
        for _ in range(draws)
    ]
    collision = extreme_collision_rate(templates, bins)
    boot = []
    for _ in range(draws):
        indices = rng.integers(0, len(templates), size=len(templates))
        value = extreme_collision_rate(
            [templates[int(index)] for index in indices],
            [bins[int(index)] for index in indices],
        )
        if value is not None:
            boot.append(value)
    return {
        "mutual_information_nats": observed,
        "permutation_mean_mi_nats": float(np.mean(permuted)),
        "permutation_adjusted_mi_nats": observed - float(np.mean(permuted)),
        "permutation_p_ge_observed": (1 + sum(value >= observed for value in permuted))
        / (draws + 1),
        "reader_bins": dict(sorted(Counter(bins).items())),
        "extreme_0v3_pairwise_exact_collision_rate": collision,
        "extreme_collision_cluster_bootstrap_ci": None
        if not boot
        else [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "valid_bootstrap_draws": len(boot),
    }


def audit_image_integrity(
    rows: list[dict[str, Any]], image_root: Path
) -> dict[str, Any]:
    try:
        import pydicom
    except ImportError as error:  # Runtime-only dependency; CPU statistics stay importable.
        raise RuntimeError(
            "DICOM integrity audit requires pydicom in the runtime environment"
        ) from error
    by_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), row)
    file_hashes = []
    render_hashes = []
    identifiers: dict[str, list[str]] = {
        name: []
        for name in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
    }
    for image_id, row in sorted(by_image.items()):
        path = image_root / str(row["dicom_relpath"]).removeprefix("train/")
        _require(path.is_file(), f"missing DICOM: {path}")
        file_hashes.append(sha256_file(path))
        header = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        for name in identifiers:
            identifiers[name].append(str(getattr(header, name, "") or ""))
        image = dicom_to_pil(path)
        render_hashes.append(
            hashlib.sha256(
                f"{image.mode}:{image.size}".encode() + image.tobytes()
            ).hexdigest()
        )
    identifier_summary = {
        name: {
            "present": sum(bool(value) for value in values),
            "unique_nonempty": len({value for value in values if value}),
        }
        for name, values in identifiers.items()
    }
    linkage_available = all(
        identifier_summary[name]["present"] == len(by_image)
        for name in ("PatientID", "StudyInstanceUID")
    )
    return {
        "images": len(by_image),
        "unique_dicom_file_hashes": len(set(file_hashes)),
        "unique_rendered_pixel_hashes": len(set(render_hashes)),
        "dicom_file_identity_passed": len(set(file_hashes)) == len(by_image),
        "render_identity_passed": len(set(render_hashes)) == len(by_image),
        "dicom_identifiers": identifier_summary,
        "patient_and_study_linkage_available": linkage_available,
        "cross_patient_study_collision_exclusion_passed": linkage_available,
    }


def analyze(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    image_integrity: dict[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    _require(draws > 0, "draws must be positive")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    fingerprints = set()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        condition = str(row.get("prompt_condition", ""))
        _require(image_id and condition in CONDITIONS, "invalid generation identity")
        _require(condition not in grouped[image_id], f"duplicate {image_id}/{condition}")
        _require(row.get("clinical_claim_evaluation_status") == "pending_shared_audit", "generation assigned clinical truth")
        grouped[image_id][condition] = row
        fingerprints.add(str(row.get("fingerprint", "")))
    _require(len(fingerprints) == 1 and "" not in fingerprints, "generation fingerprints differ")
    _require(all(set(values) == set(CONDITIONS) for values in grouped.values()), "incomplete prompt triplet")

    votes: dict[tuple[str, str], int] = {}
    for row in reference_rows:
        key = (str(row["image_id"]), str(row["finding"]))
        if key[0] in grouped:
            _require(key not in votes, f"duplicate reference {key}")
            votes[key] = int(row["positive_votes"])
    findings = sorted(FROZEN_FINDINGS)
    _require(
        set(votes) == {(image_id, finding) for image_id in grouped for finding in findings},
        "reference universe does not exactly cover selected images/findings",
    )
    _require(image_integrity.get("images") == len(grouped), "image-integrity count mismatch")

    conditions: dict[str, Any] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        subset = [grouped[image_id][condition] for image_id in sorted(grouped)]
        exact = [normalize_text(str(row["text"])) for row in subset]
        prefix = [tuple(int(value) for value in row["generated_token_ids"][:10]) for row in subset]
        exact_counts = Counter(exact)
        top_templates = []
        for template, count in exact_counts.most_common(10):
            template_images = [
                str(row["image_id"])
                for row in subset
                if normalize_text(str(row["text"])) == template
            ]
            top_templates.append(
                {
                    "normalized_text": template,
                    "count": count,
                    "share": count / len(subset),
                    "reader_vote_bins_by_finding": {
                        finding: dict(
                            sorted(Counter(votes[(image_id, finding)] for image_id in template_images).items())
                        )
                        for finding in findings
                    },
                }
            )
        conditions[condition] = {
            "exact_report": concentration(exact),
            "first_10_generated_token_ids": concentration(prefix),
            "cap_hits": sum(bool(row.get("hit_max_new_tokens")) for row in subset),
            "surface_refusal_matches": sum(bool(row.get("surface_refusal_match")) for row in subset),
            "prompt_echoes": sum(
                normalize_text(str(row["prompt"])) in normalize_text(str(row["text"]))
                for row in subset
            ),
            "top_exact_templates": top_templates,
            "reader_association_by_finding": {
                finding: association(
                    exact,
                    [votes[(str(row["image_id"]), finding)] for row in subset],
                    draws=draws,
                    seed=seed + 1000 * condition_index + finding_index,
                )
                for finding_index, finding in enumerate(findings)
            },
        }

    association_paths = [
        (condition, finding)
        for condition in CONDITIONS
        for finding in findings
    ]
    raw_pvalues = [
        float(conditions[condition]["reader_association_by_finding"][finding]["permutation_p_ge_observed"])
        for condition, finding in association_paths
    ]
    adjusted_pvalues = benjamini_hochberg(raw_pvalues)
    for (condition, finding), qvalue in zip(association_paths, adjusted_pvalues):
        conditions[condition]["reader_association_by_finding"][finding][
            "benjamini_hochberg_q_across_24_tests"
        ] = qvalue

    transitions = {}
    for first, second in (("neutral", "existential"), ("neutral", "negative_obligation")):
        pairs = []
        same_exact = 0
        same_prefix = 0
        for image_id in sorted(grouped):
            left = grouped[image_id][first]
            right = grouped[image_id][second]
            left_exact = normalize_text(str(left["text"]))
            right_exact = normalize_text(str(right["text"]))
            left_prefix = tuple(left["generated_token_ids"][:10])
            right_prefix = tuple(right["generated_token_ids"][:10])
            same_exact += left_exact == right_exact
            same_prefix += left_prefix == right_prefix
            pairs.append((left_exact, right_exact))
        transitions[f"{first}_to_{second}"] = {
            "same_exact_rate": same_exact / len(grouped),
            "same_prefix10_rate": same_prefix / len(grouped),
            "unique_exact_transitions": len(set(pairs)),
            "top_exact_transitions": [
                {"from": pair[0], "to": pair[1], "count": count}
                for pair, count in Counter(pairs).most_common(10)
            ],
        }

    return {
        "version": VERSION,
        "status": "complete_model_specific_diagnostic",
        "items": len(grouped),
        "generations": len(rows),
        "conditions": conditions,
        "reader_association_multiple_testing": {
            "family": "3 prompt conditions x 8 frozen findings",
            "tests": len(association_paths),
            "method": "Benjamini-Hochberg",
            "fdr_0p05_rejections": [
                {"prompt_condition": condition, "finding": finding, "q": qvalue}
                for (condition, finding), qvalue in zip(association_paths, adjusted_pvalues)
                if qvalue <= 0.05
            ],
        },
        "within_image_transitions": transitions,
        "image_integrity": image_integrity,
        "gates": {
            "complete_triplets": len(rows) == 3 * len(grouped),
            "unique_dicom_files": image_integrity.get("dicom_file_identity_passed") is True,
            "unique_rendered_images": image_integrity.get("render_identity_passed") is True,
            "cross_patient_study_collision_exclusion": image_integrity.get("cross_patient_study_collision_exclusion_passed") is True,
            "second_medical_vlm_replication": False,
            "clinical_autoregressive_lock_in_causal_transition": False,
        },
        "paper_mechanism_authorized": False,
        "interpretation": (
            "Exact-output and prefix identity are diagnostic behavior only. Reader-bin association "
            "does not parse claims, establish clinical correctness, or localize a mechanism."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.generations.read_text().splitlines() if line.strip()]
    reference_rows = full_fixed_panel_universe(args.labels_csv, args.ontology, args.seed)
    integrity = audit_image_integrity(rows, args.image_root)
    result = analyze(rows, reference_rows, integrity, draws=args.draws, seed=args.seed)
    result["provenance"] = {
        "generations": str(args.generations.resolve()),
        "generations_sha256": sha256_file(args.generations),
        "labels_csv": str(args.labels_csv.resolve()),
        "labels_csv_sha256": sha256_file(args.labels_csv),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "draws": args.draws,
        "seed": args.seed,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "conditions"}, indent=2))


if __name__ == "__main__":
    main()
