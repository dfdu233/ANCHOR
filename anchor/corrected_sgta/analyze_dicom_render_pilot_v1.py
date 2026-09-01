#!/usr/bin/env python3
"""Analyze paired DICOM-render perturbations in reader-vote units.

This file is deliberately isolated from the common evaluation stack.  The
primary estimand is the within-DICOM continuous-render orbit diameter, divided
by the robust change in baseline polarity associated with one additional
VinDr reader vote.  Viewer polarity toggles and inversions are always reported
as secondary controls and can never contribute to a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


VERSION = "dicom-render-reader-equivalent-analysis-v1"
METHOD = "reader-grounded-dicom-render-orbit-analysis-v1"
SECONDARY_NAME_MARKERS = (
    "polarity_toggle",
    "polarity-toggle",
    "inversion",
    "invert",
    "content_loss",
    "content-loss",
    "identity_lossless_duplicate",
)
IDENTITY_CONTROL_CANDIDATES = (
    "identity_lossless_duplicate",
    "lossless_identity_duplicate",
    "identity_duplicate",
    "lossless_duplicate",
)
BASELINE_CANDIDATES = (
    "baseline_percentile",
    "percentile_baseline",
    "current_percentile",
    "baseline",
)


class AnalysisInputError(ValueError):
    """Raised when a run cannot support the predeclared analysis."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2**32)


def finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def quantile_ci(values: Sequence[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def nested_get(mapping: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if value is not None:
                return value
    return None


def is_secondary_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in SECONDARY_NAME_MARKERS)


def normalize_logits(view: Mapping[str, Any]) -> dict[str, float]:
    source = nested_get(view, ("logits",), ("scores", "logits"), ("fp32_logits",))
    if not isinstance(source, Mapping):
        return {}
    aliases = {
        "yes": ("yes", "Yes", "supported", "present"),
        "no": ("no", "No", "refuted", "absent"),
        "maybe": ("maybe", "Maybe", "undetermined", "uncertain"),
    }
    output: dict[str, float] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = finite_float(source.get(candidate))
            if value is not None:
                output[target] = value
                break
    return output


def audit_pass(view: Mapping[str, Any]) -> tuple[bool, str]:
    audit = nested_get(view, ("audit",), ("clinical_audit",))
    if not isinstance(audit, Mapping):
        return False, "missing_clinical_audit"
    pixel_hash = nested_get(
        view,
        ("pixel_sha256",),
        ("pixel_hash",),
        ("audit", "pixel_sha256"),
    )
    if not isinstance(pixel_hash, str) or len(pixel_hash) < 16:
        return False, "missing_pixel_hash"
    finite_fraction = finite_float(audit.get("finite_fraction"))
    if finite_fraction is None or finite_fraction < 1.0:
        return False, "nonfinite_render_pixels"
    bbox = nested_get(
        audit,
        ("bbox_retained",),
        ("bbox_retention",),
        ("all_bboxes_retained",),
    )
    if not isinstance(bbox, (bool, np.bool_)) or not bool(bbox):
        return False, "bbox_retention_missing_or_failed"
    if finite_float(audit.get("roi_saturation_fraction")) is None:
        return False, "missing_roi_saturation_audit"
    if finite_float(audit.get("display_edge_correlation_with_baseline")) is None:
        return False, "missing_edge_consistency_audit"
    explicit = nested_get(
        view,
        ("clinical_guard_pass",),
        ("audit", "clinical_guard_pass"),
        ("audit", "guard_pass"),
        ("clinical_audit", "pass"),
    )
    if isinstance(explicit, (bool, np.bool_)):
        return bool(explicit), "complete_explicit_clinical_guard"
    # Thresholds are runner-frozen in config; absence of its final decision is
    # a contract violation rather than an invitation to tune post hoc.
    return False, "missing_explicit_clinical_guard_decision"


def normalize_view(name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    logits = normalize_logits(raw)
    polarity = finite_float(
        nested_get(raw, ("polarity",), ("scores", "polarity"), ("coordinates", "polarity"))
    )
    commitment = finite_float(
        nested_get(
            raw,
            ("commitment",),
            ("scores", "commitment"),
            ("coordinates", "commitment"),
        )
    )
    if polarity is None and {"yes", "no"} <= logits.keys():
        polarity = logits["yes"] - logits["no"]
    if commitment is None and {"yes", "no", "maybe"} <= logits.keys():
        commitment = max(logits["yes"], logits["no"]) - logits["maybe"]
    if polarity is None or commitment is None:
        raise AnalysisInputError(f"view {name!r} lacks finite FP32 polarity/commitment")
    prediction = nested_get(raw, ("prediction",), ("scores", "prediction"))
    if prediction is None and logits:
        prediction = max(logits, key=logits.get)
    passed, audit_source = audit_pass(raw)
    declared_track = str(nested_get(raw, ("track",), ("render_track",)) or "").lower()
    declared_primary = nested_get(raw, ("is_primary",), ("primary",))
    hard_secondary = is_secondary_name(name)
    is_primary = not hard_secondary and declared_track != "secondary" and declared_primary is not False
    return {
        "name": name,
        "polarity": polarity,
        "commitment": commitment,
        "prediction": str(prediction) if prediction is not None else None,
        "logits": logits,
        "clinical_guard_pass": passed,
        "clinical_guard_source": audit_source,
        "is_primary": bool(is_primary),
        "hard_secondary_exclusion": hard_secondary,
    }


def normalize_shard(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "ok"))
    if status != "ok":
        raise AnalysisInputError(f"non-ok shard status={status}")
    image_id = nested_get(
        payload,
        ("image_id",),
        ("case", "image_id"),
        ("sample", "image_id"),
        ("metadata", "image_id"),
    )
    finding = nested_get(
        payload,
        ("finding",),
        ("case", "finding"),
        ("sample", "finding"),
        ("metadata", "finding"),
    )
    votes = nested_get(
        payload,
        ("positive_votes",),
        ("case", "positive_votes"),
        ("sample", "positive_votes"),
        ("metadata", "positive_votes"),
        ("reader_votes",),
        ("vote_count",),
        ("case", "reader_votes"),
        ("case", "vote_count"),
        ("sample", "reader_votes"),
        ("metadata", "reader_votes"),
    )
    if image_id is None or finding is None or votes is None:
        raise AnalysisInputError("shard lacks image_id, finding, or reader_votes")
    votes_int = int(votes)
    if votes_int not in {0, 1, 2, 3}:
        raise AnalysisInputError(f"reader_votes must be 0..3, got {votes_int}")
    raw_views = nested_get(payload, ("views",), ("renders",), ("results", "views"))
    if isinstance(raw_views, Mapping):
        items = [(str(name), raw) for name, raw in raw_views.items()]
    elif isinstance(raw_views, Sequence) and not isinstance(raw_views, (str, bytes)):
        items = []
        for raw in raw_views:
            if not isinstance(raw, Mapping):
                raise AnalysisInputError("view list contains a non-object")
            name = nested_get(raw, ("name",), ("view_name",), ("render_name",))
            if name is None:
                raise AnalysisInputError("view list entry lacks name")
            items.append((str(name), raw))
    else:
        raise AnalysisInputError("shard lacks views/renders")
    views = {name: normalize_view(name, raw) for name, raw in items}
    if len(views) != len(items):
        raise AnalysisInputError("duplicate view name")
    return {
        "image_id": str(image_id),
        "finding": str(finding),
        "reader_votes": votes_int,
        "record_key": payload.get("record_key"),
        "config_fingerprint": payload.get("config_fingerprint"),
        "shard_path": str(path),
        "views": views,
    }


def load_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    if config.get("split") != "pilot":
        raise AnalysisInputError(
            "v1 is an exploratory pilot analyzer; dev/confirmation requires a separately frozen contract"
        )
    expected_count = config.get("selected_claims")
    expected_keys = config.get("selection_keys")
    config_fingerprint = config.get("fingerprint")
    if (
        not isinstance(expected_count, int)
        or expected_count <= 0
        or not isinstance(expected_keys, list)
        or len(expected_keys) != expected_count
        or len(set(expected_keys)) != expected_count
        or not isinstance(config_fingerprint, str)
        or not config_fingerprint
    ):
        raise AnalysisInputError("config lacks a complete frozen selection/fingerprint contract")
    expected_key_set = set(expected_keys)
    run_state_path = run_dir / "run_state.json"
    if not run_state_path.is_file():
        raise AnalysisInputError("run_state.json is absent; the collector has not completed")
    run_state = json.loads(run_state_path.read_text())
    if not (
        run_state.get("config_fingerprint") == config_fingerprint
        and run_state.get("selected_claims") == expected_count
        and run_state.get("complete_shards") == expected_count
        and run_state.get("error_shards_this_invocation") == 0
    ):
        raise AnalysisInputError("run_state does not certify an error-free complete collection")
    shard_paths = sorted((run_dir / "shards").glob("*.json"))
    if len(shard_paths) != expected_count:
        raise AnalysisInputError(
            f"expected exactly {expected_count} shards, found {len(shard_paths)}"
        )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    digest = hashlib.sha256()
    for path in shard_paths:
        file_hash = sha256_file(path)
        digest.update(path.name.encode())
        digest.update(file_hash.encode())
        try:
            row = normalize_shard(path, json.loads(path.read_text()))
            if row["config_fingerprint"] != config_fingerprint:
                raise AnalysisInputError("shard/config fingerprint mismatch")
            if row["record_key"] not in expected_key_set:
                raise AnalysisInputError("shard record key is outside the frozen selection")
            key = (row["image_id"], row["finding"])
            if key in seen:
                raise AnalysisInputError(f"duplicate case {key}")
            seen.add(key)
            rows.append(row)
        except Exception as error:  # malformed shards are auditable, not silently lost
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    observed_keys = {row["record_key"] for row in rows}
    if failures or len(rows) != expected_count or observed_keys != expected_key_set:
        raise AnalysisInputError(
            f"formal analysis requires the exact complete shard set; invalid={len(failures)} "
            f"valid={len(rows)}/{expected_count}"
        )
    provenance = {
        "config_sha256": sha256_file(config_path),
        "run_state_sha256": sha256_file(run_state_path),
        "shard_set_sha256": digest.hexdigest(),
        "shard_files": len(shard_paths),
        "valid_shards": len(rows),
        "invalid_shards": failures,
    }
    return config, rows, provenance


def resolve_baseline(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    configured = nested_get(
        config,
        ("baseline_view",),
        ("rendering", "baseline_view"),
        ("render_config", "baseline_view"),
    )
    common = set.intersection(*(set(row["views"]) for row in rows))
    candidates = ([str(configured)] if configured is not None else []) + list(BASELINE_CANDIDATES)
    for candidate in candidates:
        if candidate in common:
            return candidate
    raise AnalysisInputError(
        f"cannot resolve a common baseline view; configured={configured!r}, common={sorted(common)}"
    )


def clinical_eligibility(
    rows: Sequence[Mapping[str, Any]], baseline: str, minimum_rate: float
) -> tuple[dict[str, Any], list[str]]:
    names = sorted({name for row in rows for name in row["views"]})
    audits: dict[str, Any] = {}
    eligible: list[str] = []
    total = len(rows)
    for name in names:
        available = [row["views"][name] for row in rows if name in row["views"]]
        passed = sum(view["clinical_guard_pass"] for view in available)
        primary = all(view["is_primary"] for view in available) and not is_secondary_name(name)
        availability_rate = len(available) / total
        joint_pass_rate = passed / total
        gate_eligible = (
            name != baseline
            and primary
            and availability_rate >= minimum_rate
            and joint_pass_rate >= minimum_rate
        )
        audits[name] = {
            "n_cases_total": total,
            "n_available": len(available),
            "n_clinical_guard_pass": passed,
            "availability_rate": availability_rate,
            "joint_audit_pass_rate": joint_pass_rate,
            "primary_continuous_track": primary,
            "hard_secondary_exclusion": is_secondary_name(name),
            "gate_eligible": gate_eligible,
            "exclusion_reasons": [
                reason
                for condition, reason in (
                    (name == baseline, "baseline_not_a_perturbation"),
                    (not primary, "secondary_or_inversion"),
                    (availability_rate < minimum_rate, "insufficient_coverage"),
                    (joint_pass_rate < minimum_rate, "insufficient_clinical_audit_pass_rate"),
                )
                if condition
            ],
            "audit_source_counts": dict(Counter(view["clinical_guard_source"] for view in available)),
        }
        if gate_eligible:
            eligible.append(name)
    return audits, eligible


def theil_sen_cross_bin(rows: Sequence[Mapping[str, Any]], score_key: str = "baseline_polarity") -> float | None:
    slopes: list[float] = []
    by_vote: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get(score_key))
        if value is not None:
            by_vote[int(row["reader_votes"])].append(value)
    for low in range(4):
        for high in range(low + 1, 4):
            for left in by_vote.get(low, []):
                for right in by_vote.get(high, []):
                    slopes.append((right - left) / (high - low))
    return float(np.median(slopes)) if slopes else None


def cluster_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    return groups


def bootstrap_stat(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[list[Mapping[str, Any]]], float | None],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    groups = cluster_groups(rows)
    image_ids = sorted(groups)
    estimate = statistic(list(rows))
    if not image_ids or repetitions <= 0:
        return {"estimate": estimate, "ci95": None, "valid_replicates": 0}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        replicate = [row for image_id in sampled for row in groups[str(image_id)]]
        value = statistic(replicate)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return {
        "estimate": estimate,
        "ci95": quantile_ci(values),
        "valid_replicates": len(values),
        "requested_replicates": repetitions,
        "cluster_unit": "image_id",
    }


def rank_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(y) != len(x):
        return None
    rx, ry = rank_values(x), rank_values(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def orbit_diameter(row: Mapping[str, Any], names: Sequence[str], coordinate: str) -> float | None:
    # A case enters the orbit statistic only when the entire globally frozen
    # transform set passed its audit.  Allowing a different subset per case
    # would make the diameter (and thus the gate) depend on missingness.
    if len(names) < 2 or any(
        name not in row["views"] or not row["views"][name]["clinical_guard_pass"]
        for name in names
    ):
        return None
    values = [float(row["views"][name][coordinate]) for name in names]
    return max(values) - min(values)


def deterministic_swaps(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["finding"]), int(row["reader_votes"]))].append(row)
    for (finding, votes), values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda row: hashlib.sha256(
                f"{seed}|{finding}|{votes}|{row['image_id']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < 2:
            continue
        for index, row in enumerate(ordered):
            other = ordered[(index + 1) % len(ordered)]
            pairs.append(
                {
                    "image_id": row["image_id"],
                    "other_image_id": other["image_id"],
                    "finding": finding,
                    "reader_votes": votes,
                    "absolute_baseline_polarity_difference": abs(
                        row["baseline_polarity"] - other["baseline_polarity"]
                    ),
                }
            )
    return pairs


def paired_effect_stat(rows: Sequence[Mapping[str, Any]], name: str, coordinate: str) -> float | None:
    deltas = [
        row["views"][name][coordinate] - row[f"baseline_{coordinate}"]
        for row in rows
        if name in row["views"] and row["views"][name]["clinical_guard_pass"]
    ]
    return float(np.median(deltas)) if deltas else None


def paired_effects(
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    beta: float | None,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in names:
        item: dict[str, Any] = {}
        for coordinate in ("polarity", "commitment"):
            deltas = [
                row["views"][name][coordinate] - row[f"baseline_{coordinate}"]
                for row in rows
                if name in row["views"] and row["views"][name]["clinical_guard_pass"]
            ]
            boot = bootstrap_stat(
                rows,
                lambda batch, n=name, c=coordinate: paired_effect_stat(batch, n, c),
                repetitions,
                stable_seed(seed, "paired", name, coordinate),
            )
            item[coordinate] = {
                "delta": summary(deltas),
                "median_delta_cluster_bootstrap": boot,
                "sign_consistency": (
                    max(
                        sum(value > 0 for value in deltas),
                        sum(value < 0 for value in deltas),
                    )
                    / len(deltas)
                    if deltas
                    else None
                ),
                "median_signed_reader_equivalent": (
                    boot["estimate"] / beta
                    if coordinate == "polarity" and beta is not None and beta > 0 and boot["estimate"] is not None
                    else None
                ),
            }
        output[name] = item
    return output


def deterministic_dev_half(row: Mapping[str, Any], seed: int, finding: str) -> str:
    digest = hashlib.sha256(f"{seed}|{finding}|{row['image_id']}".encode()).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def valid_paired_delta(
    row: Mapping[str, Any], name: str, coordinate: str = "polarity"
) -> float | None:
    view = row["views"].get(name)
    if view is None or not view["clinical_guard_pass"]:
        return None
    return float(view[coordinate] - row[f"baseline_{coordinate}"])


def select_and_confirm_transform(
    rows: Sequence[Mapping[str, Any]],
    finding: str,
    primary_names: Sequence[str],
    identity_name: str | None,
    beta: float | None,
    repetitions: int,
    seed: int,
    minimum_audit_rate: float,
    heldout_min_abs_re: float,
    heldout_re_ci_min_magnitude: float,
    heldout_sign_agreement_threshold: float,
) -> dict[str, Any]:
    half_a = [row for row in rows if deterministic_dev_half(row, seed, finding) == "A"]
    half_b = [row for row in rows if deterministic_dev_half(row, seed, finding) == "B"]
    candidates: dict[str, Any] = {}
    for name in primary_names:
        deltas = [
            value
            for row in half_a
            if (value := valid_paired_delta(row, name)) is not None
        ]
        candidates[name] = {
            "n_half_a": len(deltas),
            "half_a_coverage": len(deltas) / len(half_a) if half_a else 0.0,
            "half_a_median_signed_polarity_delta": (
                float(np.median(deltas)) if deltas else None
            ),
        }
    selectable = [
        name
        for name, item in candidates.items()
        if item["n_half_a"] >= 4
        and item["half_a_coverage"] >= minimum_audit_rate
        and item["half_a_median_signed_polarity_delta"] is not None
    ]
    if not selectable:
        return {
            "selected_transform": None,
            "half_assignment": "sha256(seed|finding|image_id) byte parity",
            "candidate_half_a_effects": candidates,
            "stable_heldout_effect_pass": False,
            "reader_equivalent_magnitude_pass": False,
            "heldout_sign_agreement_pass": False,
            "duplicate_noise_floor_pass": False,
            "margin_stability_pass": False,
            "pass": False,
            "reason": "no primary transform has at least four audited half-A cases",
        }
    selected = min(
        selectable,
        key=lambda name: (
            -abs(candidates[name]["half_a_median_signed_polarity_delta"]),
            name,
        ),
    )
    selection_effect = float(candidates[selected]["half_a_median_signed_polarity_delta"])
    selection_sign = 1 if selection_effect > 0 else -1 if selection_effect < 0 else 0
    heldout_rows = [row for row in half_b if valid_paired_delta(row, selected) is not None]
    heldout_coverage = len(heldout_rows) / len(half_b) if half_b else 0.0
    heldout = bootstrap_stat(
        heldout_rows,
        lambda batch: float(
            np.median([valid_paired_delta(row, selected) for row in batch])
        )
        if batch
        else None,
        repetitions,
        stable_seed(seed, finding, selected, "heldout"),
    )
    heldout_effect = heldout["estimate"]
    heldout_ci = heldout["ci95"]
    heldout_sign = (
        1 if heldout_effect is not None and heldout_effect > 0 else -1 if heldout_effect is not None and heldout_effect < 0 else 0
    )
    heldout_sign_agreement = (
        float(
            np.mean(
                [
                    valid_paired_delta(row, selected) * selection_sign > 0
                    for row in heldout_rows
                ]
            )
        )
        if heldout_rows and selection_sign
        else None
    )
    ci_excludes_zero_in_selected_direction = bool(
        heldout_ci
        and selection_sign
        and (
            (selection_sign > 0 and heldout_ci[0] > 0)
            or (selection_sign < 0 and heldout_ci[1] < 0)
        )
    )
    heldout_re = bootstrap_stat(
        rows,
        lambda batch: (
            float(
                np.median(
                    [
                        valid_paired_delta(row, selected)
                        for row in batch
                        if deterministic_dev_half(row, seed, finding) == "B"
                        and valid_paired_delta(row, selected) is not None
                    ]
                )
            )
            / local_beta
            if (
                (local_beta := theil_sen_cross_bin(batch)) is not None
                and local_beta > 0
                and any(
                    deterministic_dev_half(row, seed, finding) == "B"
                    and valid_paired_delta(row, selected) is not None
                    for row in batch
                )
            )
            else None
        ),
        repetitions,
        stable_seed(seed, finding, selected, "heldout-reader-equivalent"),
    )
    heldout_re_estimate = heldout_re["estimate"]
    heldout_re_ci = heldout_re["ci95"]
    re_magnitude_pass = bool(
        heldout_re_estimate is not None
        and abs(heldout_re_estimate) >= heldout_min_abs_re
        and heldout_re_ci
        and selection_sign
        and (
            (selection_sign > 0 and heldout_re_ci[0] > heldout_re_ci_min_magnitude)
            or (selection_sign < 0 and heldout_re_ci[1] < -heldout_re_ci_min_magnitude)
        )
    )
    sign_agreement_pass = bool(
        heldout_sign_agreement is not None
        and heldout_sign_agreement >= heldout_sign_agreement_threshold
    )
    stable = bool(
        len(heldout_rows) >= 4
        and heldout_coverage >= minimum_audit_rate
        and heldout_sign == selection_sign
        and ci_excludes_zero_in_selected_direction
        and re_magnitude_pass
        and sign_agreement_pass
    )

    # A transform must beat an exactly content-preserving export/reload path,
    # not merely have a nonzero floating-point delta.
    duplicate_rows = [
        row
        for row in heldout_rows
        if identity_name is not None and valid_paired_delta(row, identity_name) is not None
    ]

    def excess_over_duplicate(batch: list[Mapping[str, Any]]) -> float | None:
        values = []
        for row in batch:
            effect = valid_paired_delta(row, selected)
            duplicate = valid_paired_delta(row, identity_name) if identity_name else None
            if effect is not None and duplicate is not None:
                values.append(abs(effect) - abs(duplicate))
        return float(np.median(values)) if values else None

    duplicate_excess = bootstrap_stat(
        duplicate_rows,
        excess_over_duplicate,
        repetitions,
        stable_seed(seed, finding, selected, "duplicate-noise"),
    )
    selected_abs = [abs(valid_paired_delta(row, selected)) for row in duplicate_rows]
    duplicate_abs = [abs(valid_paired_delta(row, identity_name)) for row in duplicate_rows]
    selected_abs_median = float(np.median(selected_abs)) if selected_abs else None
    duplicate_abs_median = float(np.median(duplicate_abs)) if duplicate_abs else None
    noise_ratio = (
        selected_abs_median / duplicate_abs_median
        if selected_abs_median is not None and duplicate_abs_median not in (None, 0.0)
        else math.inf
        if selected_abs_median is not None and selected_abs_median > 0 and duplicate_abs_median == 0
        else None
    )
    duplicate_pass = bool(
        identity_name
        and len(duplicate_rows) >= 4
        and len(duplicate_rows) / len(heldout_rows) >= minimum_audit_rate
        and duplicate_excess["ci95"]
        and duplicate_excess["ci95"][0] > 0
        and noise_ratio is not None
        and noise_ratio > 1
    )

    # Confirm the selected signed effect away from the low-margin boundary.
    if heldout_rows:
        margin_cut = float(
            np.median([row["baseline_abs_polarity_margin"] for row in heldout_rows])
        )
    else:
        margin_cut = None
    high_margin_rows = [
        row
        for row in heldout_rows
        if margin_cut is not None and row["baseline_abs_polarity_margin"] > margin_cut
    ]
    high_margin_effect = bootstrap_stat(
        high_margin_rows,
        lambda batch: float(
            np.median([valid_paired_delta(row, selected) for row in batch])
        )
        if batch
        else None,
        repetitions,
        stable_seed(seed, finding, selected, "heldout-high-margin"),
    )
    high_ci = high_margin_effect["ci95"]
    margin_stability = bool(
        len(high_margin_rows) >= 4
        and high_ci
        and selection_sign
        and (
            (selection_sign > 0 and high_ci[0] > 0)
            or (selection_sign < 0 and high_ci[1] < 0)
        )
    )
    return {
        "selected_transform": selected,
        "selection_rule": "largest absolute half-A median paired polarity effect; lexical tie-break",
        "half_assignment": "sha256(seed|finding|image_id) byte parity",
        "n_half_a": len(half_a),
        "n_half_b": len(half_b),
        "selected_half_b_audited_coverage": heldout_coverage,
        "minimum_transform_specific_coverage": minimum_audit_rate,
        "candidate_half_a_effects": candidates,
        "selected_half_a_signed_effect": selection_effect,
        "heldout_half_b_signed_effect": heldout,
        "heldout_half_b_sign_agreement": heldout_sign_agreement,
        "heldout_sign_agreement_threshold": heldout_sign_agreement_threshold,
        "heldout_sign_agreement_pass": sign_agreement_pass,
        "heldout_signed_reader_equivalent_joint_cluster_bootstrap": heldout_re,
        "reader_equivalent_magnitude_rule": (
            f"absolute heldout RE >= {heldout_min_abs_re:g} and signed CI lower magnitude "
            f"> {heldout_re_ci_min_magnitude:g}; beta is re-estimated inside each image bootstrap"
        ),
        "reader_equivalent_magnitude_pass": re_magnitude_pass,
        "stable_heldout_effect_pass": stable,
        "identity_lossless_duplicate_control": {
            "view": identity_name,
            "n_paired_half_b": len(duplicate_rows),
            "selected_absolute_delta": summary(selected_abs),
            "duplicate_absolute_delta": summary(duplicate_abs),
            "median_absolute_effect_ratio": noise_ratio,
            "paired_median_excess_cluster_bootstrap": duplicate_excess,
            "pass": duplicate_pass,
            "rule": "paired absolute-effect excess CI lower > 0 and median ratio > 1",
        },
        "heldout_high_margin_control": {
            "median_margin_cut": margin_cut,
            "n": len(high_margin_rows),
            "signed_effect_cluster_bootstrap": high_margin_effect,
            "pass": margin_stability,
        },
        "duplicate_noise_floor_pass": duplicate_pass,
        "margin_stability_pass": margin_stability,
        "pass": bool(stable and duplicate_pass and margin_stability),
    }


def ordinary_least_squares(y: Sequence[float], columns: Sequence[Sequence[float]]) -> dict[str, Any]:
    target = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(target)), *[np.asarray(column, dtype=float) for column in columns]])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    total = float(np.sum((target - target.mean()) ** 2))
    residual = float(np.sum((target - predicted) ** 2))
    return {
        "coefficients": [float(value) for value in coefficients],
        "rank": int(rank),
        "r2": 1.0 - residual / total if total > 0 else None,
    }


def margin_controls(
    rows: Sequence[Mapping[str, Any]],
    orbit_names: Sequence[str],
    beta: float | None,
    repetitions: int,
    seed: int,
    high_margin_min_re: float,
    high_margin_ci_min_re: float,
) -> dict[str, Any]:
    usable = [row for row in rows if row.get("polarity_orbit_diameter") is not None]
    if not usable:
        return {"available": False, "pass": False, "reason": "no valid primary orbit"}
    median_margin = float(np.median([row["baseline_abs_polarity_margin"] for row in usable]))
    low = [row for row in usable if row["baseline_abs_polarity_margin"] <= median_margin]
    high = [row for row in usable if row["baseline_abs_polarity_margin"] > median_margin]

    def stratum_median_re(
        batch: list[Mapping[str, Any]], high_stratum: bool
    ) -> float | None:
        # Bootstrap the complete finding so beta remains a 0..3-reader unit;
        # then evaluate the frozen high/low-margin stratum inside that draw.
        local_beta = theil_sen_cross_bin(batch)
        if local_beta is None or local_beta <= 0:
            return None
        selected = [
            row
            for row in batch
            if (row["baseline_abs_polarity_margin"] > median_margin) == high_stratum
        ]
        values = [row["polarity_orbit_diameter"] / local_beta for row in selected]
        return float(np.median(values)) if values else None

    high_boot = bootstrap_stat(
        usable,
        lambda batch: stratum_median_re(batch, True),
        repetitions,
        stable_seed(seed, "high-margin"),
    )
    low_boot = bootstrap_stat(
        usable,
        lambda batch: stratum_median_re(batch, False),
        repetitions,
        stable_seed(seed, "low-margin"),
    )
    x_margin = [row["baseline_abs_polarity_margin"] for row in usable]
    y_orbit = [row["polarity_orbit_diameter"] for row in usable]
    correlation = bootstrap_stat(
        usable,
        lambda batch: spearman(
            [row["baseline_abs_polarity_margin"] for row in batch],
            [row["polarity_orbit_diameter"] for row in batch],
        ),
        repetitions,
        stable_seed(seed, "margin-spearman"),
    )

    # Case-render panel: quantify how much transform identity explains beyond
    # absolute baseline margin.  This is a diagnostic, not a discovery gate.
    panel: list[tuple[Mapping[str, Any], str, float]] = []
    for row in usable:
        for name in orbit_names:
            if name in row["views"] and row["views"][name]["clinical_guard_pass"]:
                panel.append(
                    (row, name, abs(row["views"][name]["polarity"] - row["baseline_polarity"]))
                )
    transform_levels = sorted({name for _, name, _ in panel})
    if panel:
        outcome = [value for _, _, value in panel]
        standardized_margin = np.asarray(
            [row["baseline_abs_polarity_margin"] for row, _, _ in panel], dtype=float
        )
        if standardized_margin.std() > 0:
            standardized_margin = (standardized_margin - standardized_margin.mean()) / standardized_margin.std()
        margin_model = ordinary_least_squares(outcome, [standardized_margin])
        dummies = [
            [1.0 if name == level else 0.0 for _, name, _ in panel]
            for level in transform_levels[1:]
        ]
        full_model = ordinary_least_squares(outcome, [standardized_margin, *dummies])
        incremental_r2 = (
            full_model["r2"] - margin_model["r2"]
            if full_model["r2"] is not None and margin_model["r2"] is not None
            else None
        )
    else:
        margin_model = full_model = {"coefficients": [], "rank": 0, "r2": None}
        incremental_r2 = None

    high_estimate = high_boot.get("estimate")
    high_ci = high_boot.get("ci95")
    passed = bool(
        len(high) >= 4
        and high_estimate is not None
        and high_estimate >= high_margin_min_re
        and high_ci is not None
        and high_ci[0] > high_margin_ci_min_re
    )
    return {
        "available": True,
        "definition": "high/low strata split at within-finding median absolute baseline polarity",
        "median_absolute_baseline_polarity": median_margin,
        "low_margin_n": len(low),
        "high_margin_n": len(high),
        "low_margin_median_reader_equivalent": low_boot,
        "high_margin_median_reader_equivalent": high_boot,
        "orbit_vs_absolute_margin_spearman": correlation,
        "absolute_delta_regression": {
            "outcome": "absolute paired polarity delta",
            "margin_only": margin_model,
            "margin_plus_transform_fixed_effects": full_model,
            "transform_levels": transform_levels,
            "incremental_r2_from_transform_identity": incremental_r2,
        },
        "pass": passed,
        "gate_rule": (
            f"high-margin median reader-equivalent >= {high_margin_min_re:g} and "
            f"image-bootstrap CI lower > {high_margin_ci_min_re:g}"
        ),
        "note": "This prevents a finding from passing solely through near-boundary cases.",
    }


def enrich_rows(
    rows: Sequence[dict[str, Any]], baseline: str, eligible_primary: Sequence[str]
) -> None:
    orbit_names = [baseline, *eligible_primary]
    for row in rows:
        base = row["views"].get(baseline)
        if base is None or not base["clinical_guard_pass"]:
            raise AnalysisInputError(
                f"case {(row['image_id'], row['finding'])} lacks a clinically valid baseline"
            )
        row["baseline_polarity"] = float(base["polarity"])
        row["baseline_commitment"] = float(base["commitment"])
        row["baseline_abs_polarity_margin"] = abs(float(base["polarity"]))
        row["polarity_orbit_diameter"] = orbit_diameter(row, orbit_names, "polarity")
        row["commitment_orbit_diameter"] = orbit_diameter(row, orbit_names, "commitment")


def finding_analysis(
    finding: str,
    rows: list[dict[str, Any]],
    baseline: str,
    primary_names: Sequence[str],
    all_nonbaseline_names: Sequence[str],
    identity_name: str | None,
    repetitions: int,
    seed: int,
    min_per_bin: int,
    median_re_threshold: float,
    median_re_ci_threshold: float,
    one_step_fraction_threshold: float,
    high_margin_min_re: float,
    high_margin_ci_min_re: float,
    minimum_audit_rate: float,
    heldout_min_abs_re: float,
    heldout_re_ci_min_magnitude: float,
    heldout_sign_agreement_threshold: float,
) -> dict[str, Any]:
    counts = Counter(int(row["reader_votes"]) for row in rows)
    beta = bootstrap_stat(
        rows,
        theil_sen_cross_bin,
        repetitions,
        stable_seed(seed, finding, "beta"),
    )
    beta_value = beta["estimate"]
    beta_ci = beta["ci95"]

    def median_re(batch: list[Mapping[str, Any]]) -> float | None:
        local_beta = theil_sen_cross_bin(batch)
        if local_beta is None or local_beta <= 0:
            return None
        values = [
            row["polarity_orbit_diameter"] / local_beta
            for row in batch
            if row.get("polarity_orbit_diameter") is not None
        ]
        return float(np.median(values)) if values else None

    def one_step_fraction(batch: list[Mapping[str, Any]]) -> float | None:
        local_beta = theil_sen_cross_bin(batch)
        if local_beta is None or local_beta <= 0:
            return None
        values = [
            row["polarity_orbit_diameter"] >= local_beta
            for row in batch
            if row.get("polarity_orbit_diameter") is not None
        ]
        return float(np.mean(values)) if values else None

    median_re = bootstrap_stat(
        rows, median_re, repetitions, stable_seed(seed, finding, "median-re")
    )
    one_step = bootstrap_stat(
        rows, one_step_fraction, repetitions, stable_seed(seed, finding, "one-step")
    )
    raw_orbits = [
        row["polarity_orbit_diameter"]
        for row in rows
        if row.get("polarity_orbit_diameter") is not None
    ]
    commitment_orbits = [
        row["commitment_orbit_diameter"]
        for row in rows
        if row.get("commitment_orbit_diameter") is not None
    ]
    paired_primary = paired_effects(rows, primary_names, beta_value, repetitions, seed)
    secondary_names = [name for name in all_nonbaseline_names if name not in primary_names]
    paired_secondary = paired_effects(rows, secondary_names, beta_value, repetitions, seed)
    margin = margin_controls(
        rows,
        [baseline, *primary_names],
        beta_value,
        repetitions,
        stable_seed(seed, finding, "margin"),
        high_margin_min_re,
        high_margin_ci_min_re,
    )
    transform_confirmation = select_and_confirm_transform(
        rows,
        finding,
        primary_names,
        identity_name,
        beta_value,
        repetitions,
        stable_seed(seed, finding, "transform-confirmation"),
        minimum_audit_rate,
        heldout_min_abs_re,
        heldout_re_ci_min_magnitude,
        heldout_sign_agreement_threshold,
    )

    beta_pass = bool(beta_value is not None and beta_value > 0 and beta_ci and beta_ci[0] > 0)
    bin_pass = all(counts[vote] >= min_per_bin for vote in range(4))
    orbit_path_a = bool(
        median_re["estimate"] is not None
        and median_re["estimate"] >= median_re_threshold
        and median_re["ci95"]
        and median_re["ci95"][0] > median_re_ci_threshold
    )
    orbit_path_b = bool(
        one_step["estimate"] is not None
        and one_step["estimate"] >= one_step_fraction_threshold
        and one_step["ci95"]
        and one_step["ci95"][0] > 0
    )
    passed = bool(
        bin_pass
        and beta_pass
        and primary_names
        and transform_confirmation["stable_heldout_effect_pass"]
        and transform_confirmation["duplicate_noise_floor_pass"]
        and transform_confirmation["margin_stability_pass"]
    )

    flip_rows = []
    for row in rows:
        baseline_prediction = row["views"][baseline]["prediction"]
        changed = any(
            row["views"][name]["prediction"] != baseline_prediction
            for name in primary_names
            if name in row["views"] and row["views"][name]["clinical_guard_pass"]
        )
        flip_rows.append(changed)

    return {
        "n_cases": len(rows),
        "reader_vote_counts": {str(vote): counts[vote] for vote in range(4)},
        "reader_step_beta": {
            **beta,
            "estimator": "median of all cross-reader-bin pairwise polarity slopes",
            "score": "baseline Yes-minus-No polarity logit",
            "pass_positive_ci": beta_pass,
        },
        "primary_continuous_orbit": {
            "eligible_transforms": list(primary_names),
            "polarity_diameter": summary(raw_orbits),
            "commitment_diameter": summary(commitment_orbits),
            "median_reader_equivalent": median_re,
            "fraction_at_least_one_reader_step": one_step,
            "paired_render_effects": paired_primary,
            "can_drive_gate": False,
            "role": "descriptive sensitivity envelope only; transform selection is independently held out",
        },
        "deterministic_transform_selection_and_confirmation": transform_confirmation,
        "secondary_controls_not_in_any_gate": {
            "transforms": secondary_names,
            "paired_render_effects": paired_secondary,
        },
        "margin_controls": margin,
        "flip_rate_diagnostic_only": {
            "n": len(flip_rows),
            "any_primary_prediction_flip_rate": float(np.mean(flip_rows)) if flip_rows else None,
            "can_drive_gate": False,
            "warning": "A token argmax flip is neither necessary nor sufficient for the mechanism claim.",
        },
        "finding_gate": {
            "passed": passed,
            "balanced_reader_bins_pass": bin_pass,
            "positive_reader_step_ci_pass": beta_pass,
            "reader_equivalent_path_a_pass": orbit_path_a,
            "one_reader_step_path_b_pass": orbit_path_b,
            "orbit_paths_can_drive_gate": False,
            "heldout_transform_stability_pass": bool(
                transform_confirmation["stable_heldout_effect_pass"]
            ),
            "duplicate_noise_floor_pass": bool(
                transform_confirmation["duplicate_noise_floor_pass"]
            ),
            "margin_control_pass": bool(transform_confirmation["margin_stability_pass"]),
            "has_primary_clinically_eligible_transform": bool(primary_names),
            "rule": (
                f">={min_per_bin} cases in every reader-vote bin; reader-step beta CI lower > 0; "
                "half-A deterministically selects one preregistered transform and its signed effect "
                "has the same sign with CI excluding zero on half-B; the effect exceeds the paired "
                "lossless-duplicate noise floor; and the half-B high-margin effect remains signed. "
                f"Heldout |RE| must be >= {heldout_min_abs_re:g}, its signed CI magnitude > "
                f"{heldout_re_ci_min_magnitude:g}, sign agreement >= "
                f"{heldout_sign_agreement_threshold:.0%}, and transform-specific coverage >= "
                f"{minimum_audit_rate:.0%}. "
                "Max-minus-min orbit and flip rates cannot drive this gate."
            ),
        },
    }


def analyze(
    run_dir: Path,
    repetitions: int,
    seed: int,
    minimum_audit_rate: float,
    min_per_bin: int,
    median_re_threshold: float,
    median_re_ci_threshold: float,
    one_step_fraction_threshold: float,
    high_margin_min_re: float,
    high_margin_ci_min_re: float,
    heldout_min_abs_re: float = 0.50,
    heldout_re_ci_min_magnitude: float = 0.25,
    heldout_sign_agreement_threshold: float = 0.65,
) -> dict[str, Any]:
    config, rows, provenance = load_rows(run_dir)
    baseline = resolve_baseline(config, rows)
    audits, primary_names = clinical_eligibility(rows, baseline, minimum_audit_rate)
    enrich_rows(rows, baseline, primary_names)
    all_nonbaseline_names = sorted(name for name in audits if name != baseline)
    identity_name = next(
        (
            name
            for candidate in IDENTITY_CONTROL_CANDIDATES
            for name in all_nonbaseline_names
            if name == candidate
            and audits[name]["availability_rate"] >= minimum_audit_rate
            and audits[name]["joint_audit_pass_rate"] >= minimum_audit_rate
        ),
        None,
    )

    by_finding: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_finding[row["finding"]].append(row)
    findings = {
        finding: finding_analysis(
            finding,
            values,
            baseline,
            primary_names,
            all_nonbaseline_names,
            identity_name,
            repetitions,
            stable_seed(seed, finding),
            min_per_bin,
            median_re_threshold,
            median_re_ci_threshold,
            one_step_fraction_threshold,
            high_margin_min_re,
            high_margin_ci_min_re,
            minimum_audit_rate,
            heldout_min_abs_re,
            heldout_re_ci_min_magnitude,
            heldout_sign_agreement_threshold,
        )
        for finding, values in sorted(by_finding.items())
    }

    swaps = deterministic_swaps(rows, seed)
    swap_by_finding: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in swaps:
        swap_by_finding[pair["finding"]].append(pair)
    swap_report: dict[str, Any] = {}
    for finding, values in sorted(swap_by_finding.items()):
        finding_rows = by_finding[finding]
        swap_values = [pair["absolute_baseline_polarity_difference"] for pair in values]
        orbit_values = [
            row["polarity_orbit_diameter"]
            for row in finding_rows
            if row.get("polarity_orbit_diameter") is not None
        ]
        swap_boot = bootstrap_stat(
            values,
            lambda batch: float(
                np.median([row["absolute_baseline_polarity_difference"] for row in batch])
            )
            if batch
            else None,
            repetitions,
            stable_seed(seed, finding, "same-support-swap"),
        )
        med_swap = float(np.median(swap_values)) if swap_values else None
        med_orbit = float(np.median(orbit_values)) if orbit_values else None
        swap_report[finding] = {
            "n_directed_pairs": len(values),
            "construction": "deterministic circular pairing within finding and reader-vote bin",
            "absolute_baseline_polarity_difference": summary(swap_values),
            "median_cluster_bootstrap": swap_boot,
            "primary_orbit_median_over_same_support_swap_median": (
                med_orbit / med_swap if med_orbit is not None and med_swap not in (None, 0.0) else None
            ),
            "role": "scale/control diagnostic; it cannot rescue or fail the formal gate",
        }

    passed_findings = [name for name, item in findings.items() if item["finding_gate"]["passed"]]
    total_findings = len(findings)
    required = max(3, math.ceil(0.75 * total_findings)) if total_findings >= 4 else 4
    overall_pass = total_findings >= 4 and len(passed_findings) >= required

    script_path = Path(__file__).resolve()
    analysis_parameters = {
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "minimum_global_clinical_audit_rate": minimum_audit_rate,
        "minimum_per_reader_vote_bin": min_per_bin,
        "median_reader_equivalent_threshold": median_re_threshold,
        "median_reader_equivalent_ci_lower_threshold": median_re_ci_threshold,
        "one_reader_step_fraction_threshold": one_step_fraction_threshold,
        "high_margin_min_reader_equivalent": high_margin_min_re,
        "high_margin_ci_lower_min_reader_equivalent": high_margin_ci_min_re,
        "heldout_min_absolute_reader_equivalent": heldout_min_abs_re,
        "heldout_reader_equivalent_ci_min_magnitude": heldout_re_ci_min_magnitude,
        "heldout_sign_agreement_threshold": heldout_sign_agreement_threshold,
    }
    fingerprint_payload = {
        "version": VERSION,
        "code_sha256": sha256_file(script_path),
        **provenance,
        "analysis_parameters": analysis_parameters,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "version": VERSION,
        "evidence_tier": "exploratory_pilot_only",
        "paper_claim_authorized": False,
        "dataset": config.get("dataset", "VinDr-CXR reader-vote pilot"),
        "model": config.get("model", config.get("model_path", "HuatuoGPT-Vision-7B")),
        "method": METHOD,
        "seed": seed,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "fingerprint": fingerprint,
        "provenance": {**provenance, "analyzer_code_sha256": sha256_file(script_path)},
        "analysis_parameters": analysis_parameters,
        "baseline_view": baseline,
        "clinical_transform_audit": {
            "global_eligibility_rule": (
                "primary continuous transform, not inversion/polarity toggle, with coverage and "
                f"clinical-guard pass jointly >= {minimum_audit_rate:.1%}"
            ),
            "eligible_primary_transforms": primary_names,
            "identity_lossless_duplicate_control": identity_name,
            "transforms": audits,
        },
        "findings": findings,
        "same_support_image_swap_control": swap_report,
        "formal_overall_gate": {
            "passed": overall_pass,
            "n_findings": total_findings,
            "required_passing_findings": required,
            "passing_findings": passed_findings,
            "rule": "at least 3 of the frozen 4 findings pass (75% if more than 4 are supplied)",
            "flip_rate_used": False,
            "secondary_transform_used": False,
            "semantic_role": "progression gate to a separately preregistered two-model or held-out replication; never paper confirmation",
        },
        "interpretation_guardrail": (
            "Passing supports reader-equivalent sensitivity to clinically audited rendering, not a "
            "scanner/source-domain center. Because this is the frozen pilot split, passing only "
            "authorizes a separately preregistered replication and no paper claim. Inversion and "
            "token flip rates are descriptive controls only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--minimum-audit-rate", type=float, default=0.95)
    parser.add_argument("--minimum-per-bin", type=int, default=5)
    parser.add_argument("--median-re-threshold", type=float, default=0.50)
    parser.add_argument("--median-re-ci-threshold", type=float, default=0.25)
    parser.add_argument("--one-step-fraction-threshold", type=float, default=0.20)
    parser.add_argument("--high-margin-min-re", type=float, default=0.25)
    parser.add_argument("--high-margin-ci-min-re", type=float, default=0.10)
    parser.add_argument("--heldout-min-abs-re", type=float, default=0.50)
    parser.add_argument("--heldout-re-ci-min-magnitude", type=float, default=0.25)
    parser.add_argument("--heldout-sign-agreement", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        parser.error("--bootstrap-repetitions must be positive")
    if not 0 < args.minimum_audit_rate <= 1:
        parser.error("--minimum-audit-rate must be in (0, 1]")
    if not 0 <= args.heldout_sign_agreement <= 1:
        parser.error("--heldout-sign-agreement must be in [0, 1]")
    output = args.output or (args.run_dir / "analysis_v1.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    payload = analyze(
        args.run_dir,
        args.bootstrap_repetitions,
        args.seed,
        args.minimum_audit_rate,
        args.minimum_per_bin,
        args.median_re_threshold,
        args.median_re_ci_threshold,
        args.one_step_fraction_threshold,
        args.high_margin_min_re,
        args.high_margin_ci_min_re,
        args.heldout_min_abs_re,
        args.heldout_re_ci_min_magnitude,
        args.heldout_sign_agreement,
    )
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
