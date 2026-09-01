#!/usr/bin/env python3
"""Frozen case-cluster analysis for full-answer Specificity Ratchet replay.

The primary signature is a late constraint-commitment gain that survives two
matched image swaps (language-side ratchet), while error cases have weaker
early own-versus-swap visual evidence than supported-specificity controls.
Layer endpoints, nuisance variables, bootstrap seed, and gates are fixed here
before physician labels or model traces are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import ContractError
from corrected_sgta.specificity_ratchet_visible_replay_v1 import RUNTIME_PROTOCOL_ID


ANALYSIS_PROTOCOL_ID = "specificity-ratchet-visible-replay-analysis-v1"
PRIMARY_ROLES = ("supported_specificity_control", "causal_escalation_error")
MIN_CASES_PER_ROLE = 12
MIN_TOTAL_CASES = 24
MIN_EDGE_TYPES = 3
RATCHET_SURVIVAL_FRACTION = 0.50
MIN_EXACT_LEXICAL_OVERLAP_BLOCKS = 10
MIN_ROLE_EFFECTIVE_CLUSTERS = 12.0
MAX_ROLE_CLUSTER_LEVERAGE = 0.20
MIN_VALID_BOOTSTRAP_FRACTION = 0.95
FROZEN_ANALYSIS_GATES = {
    "swap_language_ratchet_adjusted_ci_lower_gt": 0.0,
    "own_commitment_ratchet_adjusted_ci_lower_gt": 0.0,
    "early_visual_evidence_error_minus_control_ci_upper_lt": 0.0,
    "swap_minus_half_own_ci_lower_gt": 0.0,
    "each_individual_swap_adjusted_ci_lower_gt": 0.0,
    "image_specific_transition_adjusted_ci_upper_lt": 0.0,
    "minimum_cases_per_role_per_split": 12,
    "minimum_total_cases_per_split": 24,
    "minimum_edge_types_per_split": 3,
    "minimum_exact_lexical_overlap_blocks": 10,
    "minimum_role_effective_clusters": 12.0,
    "maximum_role_cluster_leverage": 0.20,
    "bootstrap_replicates": 5000,
    "bootstrap_seed": 7319,
    "layer_choice": "first recorded and final; no post-data selection",
    "constraint_lexical_fixed_effect": "strict sensitivity and scope gate, not primary nuisance",
    "minimum_valid_bootstrap_fraction": 0.95,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _curve(row: dict[str, Any], *keys: str) -> np.ndarray:
    value: Any = row["signals"]
    for key in keys:
        value = value[key]
    curve = np.asarray(value, dtype=float)
    if curve.ndim != 1 or curve.size < 2 or not np.isfinite(curve).all():
        raise ContractError(f"{row.get('sample_id')}: invalid layer curve at {keys}")
    return curve


def _endpoints(row: dict[str, Any]) -> dict[str, float]:
    raw = _curve(row, "raw_commitment", "constraint_minus_matched")
    own_swap = _curve(
        row,
        "primary_own_minus_matched_swaps",
        "constraint_minus_matched_difference_in_differences",
    )
    swap_constraint = _curve(row, "raw_commitment", "mean_swap_constraint_logp")
    swap_matched = _curve(
        row, "raw_commitment", "mean_swap_matched_nonconstraint_logp"
    )
    text = _curve(row, "text_only_secondary", "difference_in_differences")
    lengths = {len(raw), len(own_swap), len(swap_constraint), len(swap_matched), len(text)}
    if len(lengths) != 1:
        raise ContractError(f"{row.get('sample_id')}: layer-curve lengths differ")
    swap = swap_constraint - swap_matched
    per_swap_did = np.asarray(
        row["signals"]["primary_own_minus_matched_swaps"][
            "per_swap_difference_in_differences"
        ],
        dtype=float,
    )
    if per_swap_did.shape != (2, len(raw)) or not np.isfinite(per_swap_did).all():
        raise ContractError(f"{row.get('sample_id')}: expected exactly two per-swap curves")
    per_swap_transition = [
        float((raw[-1] - raw[0]) - (curve[-1] - curve[0]))
        for curve in per_swap_did
    ]
    return {
        "own_commitment_transition": float(raw[-1] - raw[0]),
        "swap_language_ratchet_transition": float(swap[-1] - swap[0]),
        "swap1_language_ratchet_transition": per_swap_transition[0],
        "swap2_language_ratchet_transition": per_swap_transition[1],
        "early_visual_evidence": float(own_swap[0]),
        "image_specific_transition": float(own_swap[-1] - own_swap[0]),
        "text_only_transition": float(text[-1] - text[0]),
    }


def _prepare(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    layer_ids: tuple[str, ...] | None = None
    seen: set[str] = set()
    for row in rows:
        if row.get("status") != "ok":
            raise ContractError("analysis refuses non-ok replay rows")
        if row["sample_id"] in seen:
            raise ContractError("analysis rows contain duplicate sample IDs")
        seen.add(row["sample_id"])
        current_layers = tuple(row["signals"].get("layer_ids", []))
        if len(current_layers) < 2:
            raise ContractError("analysis requires at least two frozen decoder layers")
        if layer_ids is None:
            layer_ids = current_layers
        elif current_layers != layer_ids:
            raise ContractError("analysis rows use different layer identities")
        if row["scientific_role"] not in PRIMARY_ROLES:
            continue
        counts = row["signals"]["token_counts"]
        prepared.append(
            {
                **row,
                **_endpoints(row),
                "is_error": 1.0
                if row["scientific_role"] == "causal_escalation_error"
                else 0.0,
                "log_full_answer_tokens": float(np.log1p(counts["full_visible_answer"])),
                "log_constraint_tokens": float(np.log1p(counts["constraint"])),
                "prompt_requested": float(bool(row["prompt_requested_increment"])),
            }
        )
    if not prepared:
        raise ContractError("analysis has no primary error/control rows")
    return prepared


def _category_levels(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        key: sorted({str(row[key]) for row in rows})
        for key in ("edge_type", "modality_stratum", "anatomy_stratum")
    }


def _design(
    rows: Sequence[dict[str, Any]], levels: dict[str, list[str]]
) -> np.ndarray:
    columns: list[list[float]] = [
        [1.0 for _ in rows],
        [row["is_error"] for row in rows],
        [row["log_full_answer_tokens"] for row in rows],
        [row["log_constraint_tokens"] for row in rows],
        [row["prompt_requested"] for row in rows],
    ]
    for key in ("edge_type", "modality_stratum", "anatomy_stratum"):
        for level in levels[key][1:]:
            columns.append([float(str(row[key]) == level) for row in rows])
    return np.asarray(columns, dtype=float).T


def _role_coefficient(
    rows: Sequence[dict[str, Any]], outcome: str, levels: dict[str, list[str]]
) -> float:
    design = _design(rows, levels)
    target = np.asarray([row[outcome] for row in rows], dtype=float)
    if len({row["is_error"] for row in rows}) != 2:
        raise ContractError("bootstrap sample lost one scientific role")
    nuisance = np.delete(design, 1, axis=1)
    if np.linalg.matrix_rank(design) != np.linalg.matrix_rank(nuisance) + 1:
        raise ContractError("scientific role is not identifiable after frozen nuisances")
    coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(coefficient[1])


def _design_diagnostics(
    rows: Sequence[dict[str, Any]], levels: dict[str, list[str]]
) -> dict[str, Any]:
    design = _design(rows, levels)
    nuisance = np.delete(design, 1, axis=1)
    role = design[:, 1]
    residual = role - nuisance @ np.linalg.pinv(nuisance) @ role
    if float(residual @ residual) <= 1e-10:
        return {"role_identifiable": False}
    by_case: dict[str, float] = {}
    for row, value in zip(rows, residual):
        by_case[row["case_id"]] = by_case.get(row["case_id"], 0.0) + float(value * value)
    weights = np.asarray(list(by_case.values()), dtype=float)
    weights /= weights.sum()
    singular = np.linalg.svd(nuisance, compute_uv=False)
    positive = singular[singular > 1e-10]
    return {
        "role_identifiable": True,
        "informative_cases": sum(value > 1e-10 for value in by_case.values()),
        "role_effective_clusters": float(1.0 / np.sum(weights**2)),
        "maximum_role_cluster_leverage": float(weights.max()),
        "nuisance_condition_number_nonzero_subspace": float(positive.max() / positive.min()),
    }


def _lexical_overlap(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(row["constraint_lexical_key_sha256"], []).append(row)
    blocks = {
        key: values
        for key, values in by_key.items()
        if {row["scientific_role"] for row in values} == set(PRIMARY_ROLES)
    }
    return {
        "total_keys": len(by_key),
        "cross_role_keys": len(blocks),
        "cross_role_cases_by_role": {
            role: len(
                {
                    row["case_id"]
                    for values in blocks.values()
                    for row in values
                    if row["scientific_role"] == role
                }
            )
            for role in PRIMARY_ROLES
        },
    }


def _percentile(values: Sequence[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) < 20:
        raise ContractError("too few valid case-cluster bootstrap replicates")
    return [float(value) for value in np.quantile(finite, [0.025, 0.975])]


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    statistic: Callable[[Sequence[dict[str, Any]]], float],
) -> tuple[float, list[float], int]:
    cases = sorted({row["case_id"] for row in rows})
    by_case = {case: [row for row in rows if row["case_id"] == case] for case in cases}
    point = statistic(rows)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        sample = [row for case in sampled for row in by_case[str(case)]]
        try:
            values.append(statistic(sample))
        except (ContractError, np.linalg.LinAlgError, ZeroDivisionError):
            continue
    minimum_valid = math.ceil(MIN_VALID_BOOTSTRAP_FRACTION * replicates)
    if len(values) < minimum_valid:
        raise ContractError(
            f"valid case-cluster replicates {len(values)}/{replicates} below "
            f"frozen {MIN_VALID_BOOTSTRAP_FRACTION:.2f} fraction"
        )
    return point, _percentile(values), len(values)


def analyze_rows(
    rows: Sequence[dict[str, Any]], *, bootstrap_replicates: int = 5000, seed: int = 7319
) -> dict[str, Any]:
    if bootstrap_replicates < 100:
        raise ContractError("analysis requires at least 100 bootstrap replicates")
    prepared = _prepare(rows)
    levels = _category_levels(prepared)
    cases_by_role = {
        role: len({row["case_id"] for row in prepared if row["scientific_role"] == role})
        for role in PRIMARY_ROLES
    }
    total_cases = len({row["case_id"] for row in prepared})
    edge_types = sorted({row["edge_type"] for row in prepared})
    overlap_edge_types = [
        edge_type
        for edge_type in edge_types
        if {row["scientific_role"] for row in prepared if row["edge_type"] == edge_type}
        == set(PRIMARY_ROLES)
    ]
    diagnostics = _design_diagnostics(prepared, levels)
    lexical_overlap = _lexical_overlap(prepared)
    powered = (
        min(cases_by_role.values()) >= MIN_CASES_PER_ROLE
        and total_cases >= MIN_TOTAL_CASES
        and len(overlap_edge_types) >= MIN_EDGE_TYPES
        and diagnostics.get("role_identifiable") is True
        and diagnostics.get("role_effective_clusters", 0.0) >= MIN_ROLE_EFFECTIVE_CLUSTERS
        and diagnostics.get("maximum_role_cluster_leverage", 1.0) <= MAX_ROLE_CLUSTER_LEVERAGE
    )
    if not powered:
        return {
            "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
            "status": "underpowered",
            "frozen_layer_endpoints": (
                "first recorded decoder layer versus final decoder layer"
            ),
            "cases_by_role": cases_by_role,
            "total_cases": total_cases,
            "edge_types": edge_types,
            "role_overlap_edge_types": overlap_edge_types,
            "design_diagnostics": diagnostics,
            "exact_lexical_overlap": lexical_overlap,
            "minimum_cases_per_role": MIN_CASES_PER_ROLE,
            "minimum_edge_types": MIN_EDGE_TYPES,
            "bootstrap_not_run": True,
            "estimates": {},
            "gate_checks": {},
            "interpretation_boundary": (
                "underpowered data cannot pass or fail the ratchet mechanism"
            ),
        }
    outcomes = (
        "swap_language_ratchet_transition",
        "swap1_language_ratchet_transition",
        "swap2_language_ratchet_transition",
        "own_commitment_transition",
        "early_visual_evidence",
        "image_specific_transition",
        "text_only_transition",
    )
    estimates: dict[str, Any] = {}
    for index, outcome in enumerate(outcomes):
        point, ci, valid = _cluster_bootstrap(
            prepared,
            replicates=bootstrap_replicates,
            seed=seed + index,
            statistic=lambda sample, outcome=outcome: _role_coefficient(
                sample, outcome, levels
            ),
        )
        estimates[outcome] = {
            "error_minus_supported_control_adjusted_coefficient": point,
            "case_cluster_bootstrap_95ci": ci,
            "valid_bootstrap_replicates": valid,
        }

    def survival_ratio(sample: Sequence[dict[str, Any]]) -> float:
        own = _role_coefficient(sample, "own_commitment_transition", levels)
        swap = _role_coefficient(sample, "swap_language_ratchet_transition", levels)
        if abs(own) <= 1e-8:
            raise ContractError("near-zero own commitment effect has undefined ratio")
        return swap / own

    def survival_contrast(sample: Sequence[dict[str, Any]]) -> float:
        own = _role_coefficient(sample, "own_commitment_transition", levels)
        swap = _role_coefficient(sample, "swap_language_ratchet_transition", levels)
        return swap - RATCHET_SURVIVAL_FRACTION * own

    try:
        survival_point, survival_ci, survival_valid = _cluster_bootstrap(
            prepared,
            replicates=bootstrap_replicates,
            seed=seed + len(outcomes),
            statistic=survival_ratio,
        )
    except ContractError:
        survival_point, survival_ci, survival_valid = float("nan"), [float("nan")] * 2, 0
    estimates["swap_survival_fraction_of_own_transition"] = {
        "estimate": survival_point,
        "case_cluster_bootstrap_95ci": survival_ci,
        "valid_bootstrap_replicates": survival_valid,
    }

    contrast_point, contrast_ci, contrast_valid = _cluster_bootstrap(
        prepared,
        replicates=bootstrap_replicates,
        seed=seed + len(outcomes) + 1,
        statistic=survival_contrast,
    )
    estimates["swap_minus_half_own_linear_contrast"] = {
        "estimate": contrast_point,
        "case_cluster_bootstrap_95ci": contrast_ci,
        "valid_bootstrap_replicates": contrast_valid,
    }

    checks = {
        "swap_language_ratchet_positive": estimates["swap_language_ratchet_transition"][
            "case_cluster_bootstrap_95ci"
        ][0]
        > 0,
        "own_commitment_ratchet_positive": estimates["own_commitment_transition"][
            "case_cluster_bootstrap_95ci"
        ][0]
        > 0,
        "error_has_weaker_early_visual_evidence": estimates["early_visual_evidence"][
            "case_cluster_bootstrap_95ci"
        ][1]
        < 0,
        "swap1_ratchet_positive": estimates["swap1_language_ratchet_transition"][
            "case_cluster_bootstrap_95ci"
        ][0] > 0,
        "swap2_ratchet_positive": estimates["swap2_language_ratchet_transition"][
            "case_cluster_bootstrap_95ci"
        ][0] > 0,
        "swap_exceeds_half_own_by_linear_contrast": contrast_ci[0] > 0,
        "no_positive_image_specific_late_catchup": estimates[
            "image_specific_transition"
        ]["case_cluster_bootstrap_95ci"][1] < 0,
    }
    mechanism_status = "passed" if all(checks.values()) else "failed"
    lexical_scope_qualified = (
        lexical_overlap["cross_role_keys"] >= MIN_EXACT_LEXICAL_OVERLAP_BLOCKS
    )
    status = mechanism_status if lexical_scope_qualified else (
        "failed" if mechanism_status == "failed" else "pilot_only"
    )
    return {
        "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
        "status": status,
        "frozen_layer_endpoints": "quarter-decoder recorded layer versus final decoder layer",
        "primary_estimand": (
            "error-minus-supported adjusted late-minus-early constraint-vs-matched "
            "commitment under the mean of two frozen image swaps"
        ),
        "nuisance_controls": [
            "log full-answer token count",
            "log constraint-token count",
            "prompt requested increment",
            "edge type",
            "modality",
            "anatomy",
        ],
        "secondary_sensitivities": [
            "text-only transition outcome",
            "exact normalized constraint lexical overlap scope audit",
        ],
        "cases_by_role": cases_by_role,
        "total_cases": total_cases,
        "edge_types": edge_types,
        "role_overlap_edge_types": overlap_edge_types,
        "design_diagnostics": diagnostics,
        "exact_lexical_overlap": lexical_overlap,
        "exact_lexical_scope_qualified": lexical_scope_qualified,
        "minimum_cases_per_role": MIN_CASES_PER_ROLE,
        "minimum_edge_types": MIN_EDGE_TYPES,
        "bootstrap_replicates_requested": bootstrap_replicates,
        "bootstrap_seed": seed,
        "estimates": estimates,
        "gate_checks": checks,
        "interpretation_boundary": (
            "swaps are unverified different-case positional controls, not claim-support truth; "
            "without exact lexical overlap the result is pilot-only and cannot support broad specificity"
        ),
    }


def load_runtime_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path, complete_path = run_dir / "config.json", run_dir / "COMPLETE.json"
    if not config_path.is_file() or not complete_path.is_file():
        raise ContractError("analysis requires config.json and COMPLETE.json")
    config = json.loads(config_path.read_text())
    complete = json.loads(complete_path.read_text())
    fingerprint_payload = dict(config)
    observed_fingerprint = fingerprint_payload.pop("config_fingerprint", None)
    if observed_fingerprint != hashlib.sha256(_canonical(fingerprint_payload)).hexdigest():
        raise ContractError("runtime config fingerprint does not recompute")
    if (
        config.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID
        or complete.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID
        or complete.get("status") != "complete"
        or complete.get("config_fingerprint") != config.get("config_fingerprint")
    ):
        raise ContractError("runtime completion/config contract mismatch")
    metadata_path = Path(str(config.get("metadata", "")))
    if (
        not metadata_path.is_file()
        or config.get("metadata_sha256") != _sha256(metadata_path)
    ):
        raise ContractError("runtime did not bind an intact replay metadata file")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("analysis_protocol_id") != ANALYSIS_PROTOCOL_ID
        or metadata.get("analysis_source_sha256") != _sha256(Path(__file__).resolve())
        or metadata.get("frozen_analysis_gates") != FROZEN_ANALYSIS_GATES
    ):
        raise ContractError("replay metadata does not bind this frozen analysis contract")
    manifest_path = Path(str(config.get("manifest", "")))
    if (
        not manifest_path.is_file()
        or config.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise ContractError("runtime did not bind an intact replay manifest")
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
    ]
    selected_manifest = {
        row["sample_id"]: row
        for row in manifest_rows
        if config["split"] == "all" or row["split"] == config["split"]
    }
    shards = sorted((run_dir / "shards").glob("*.json"))
    if len(shards) != complete.get("rows"):
        raise ContractError("runtime shard count differs from COMPLETE.json")
    rows = []
    for path in shards:
        shard = json.loads(path.read_text())
        if shard.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID:
            raise ContractError(f"wrong runtime shard protocol: {path}")
        payload = shard.get("payload")
        expected = hashlib.sha256(_canonical(payload)).hexdigest()
        if shard.get("payload_sha256") != expected:
            raise ContractError(f"runtime shard checksum mismatch: {path}")
        source = selected_manifest.get(payload.get("sample_id"))
        if source is None:
            raise ContractError(f"runtime shard is absent from selected manifest: {path}")
        if shard.get("row_sha256") != hashlib.sha256(_canonical(source)).hexdigest():
            raise ContractError(f"runtime shard row hash differs from manifest: {path}")
        for key in (
            "case_id",
            "edge_id",
            "split",
            "scientific_role",
            "edge_type",
            "modality_stratum",
            "anatomy_stratum",
            "prompt_requested_increment",
        ):
            if payload.get(key) != source.get(key):
                raise ContractError(f"runtime shard {key} differs from manifest: {path}")
        rows.append(payload)
    if {row["sample_id"] for row in rows} != set(selected_manifest):
        raise ContractError("runtime shards do not exactly cover selected manifest rows")
    return rows, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()
    try:
        if (
            args.bootstrap_replicates != FROZEN_ANALYSIS_GATES["bootstrap_replicates"]
            or args.seed != FROZEN_ANALYSIS_GATES["bootstrap_seed"]
        ):
            raise ContractError("scientific CLI refuses non-frozen bootstrap count or seed")
        rows, config = load_runtime_rows(args.run_dir)
        by_split = {
            split: [row for row in rows if row["split"] == split]
            for split in ("dev", "test")
            if any(row["split"] == split for row in rows)
        }
        analyses = {
            split: analyze_rows(
                split_rows,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed + index * 100,
            )
            for index, (split, split_rows) in enumerate(by_split.items())
        }
        states = {result["status"] for result in analyses.values()}
        if "failed" in states:
            overall = "failed"
        elif set(analyses) != {"dev", "test"} or "underpowered" in states:
            overall = "underpowered"
        elif "pilot_only" in states:
            overall = "pilot_only"
        else:
            overall = "passed"
        output = {
            "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
            "status": overall,
            "dataset": "VQA-RAD public image subset",
            "model": config["adapter_fingerprint"].get("model_family"),
            "method": "Specificity Ratchet full-visible-answer replay",
            "seed": args.seed,
            "command": [shlex.join(sys.argv)],
            "runtime_config_fingerprint": config["config_fingerprint"],
            "runtime_config_sha256": _sha256(args.run_dir / "config.json"),
            "analyses": analyses,
        }
        _atomic_write(
            args.output,
            (json.dumps(output, indent=2, sort_keys=True) + "\n").encode(),
        )
    except (ContractError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(output, indent=2, sort_keys=True))
    if overall != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
