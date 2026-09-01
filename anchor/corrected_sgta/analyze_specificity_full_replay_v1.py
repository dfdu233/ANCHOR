#!/usr/bin/env python3
"""Dev-frozen, case-clustered analysis of full-answer Specificity replay.

The mechanism gate is conjunctive.  Supported controls must have more early
image-specific constraint evidence; errors must acquire a larger own-image
late shift; the same error shift must occur under swapped images; and the
own-minus-swap late residual must be equivalent to zero.  Text-only remains a
reported secondary diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROTOCOL = "specificity-ratchet-full-replay-analysis-v1"
RUNTIME_PROTOCOL = "specificity-ratchet-full-visible-replay-runtime-v1"
ERROR = "causal_escalation_error"
CONTROL = "supported_specificity_control"
PRIMARY_ROLES = {ERROR, CONTROL}
MIN_CASES_PER_ROLE = 10
EQUIVALENCE_SD_MULTIPLIER = 0.2


class AnalysisError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_runtime(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path, complete_path = run_dir / "config.json", run_dir / "COMPLETE.json"
    if not config_path.is_file() or not complete_path.is_file():
        raise AnalysisError(f"incomplete replay directory: {run_dir}")
    config, complete = json.loads(config_path.read_text()), json.loads(complete_path.read_text())
    if config.get("runtime_protocol_id") != RUNTIME_PROTOCOL:
        raise AnalysisError("replay runtime protocol mismatch")
    if complete.get("runtime_protocol_id") != RUNTIME_PROTOCOL or complete.get("status") != "complete":
        raise AnalysisError("replay COMPLETE marker did not pass")
    fingerprint = config.get("config_fingerprint")
    if not fingerprint or complete.get("config_fingerprint") != fingerprint:
        raise AnalysisError("replay config/COMPLETE fingerprint mismatch")
    payloads: list[dict[str, Any]] = []
    shards = sorted((run_dir / "shards").glob("*.json"))
    for path in shards:
        shard = json.loads(path.read_text())
        payload = shard.get("payload")
        if shard.get("config_fingerprint") != fingerprint:
            raise AnalysisError(f"shard config drift: {path.name}")
        if shard.get("payload_sha256") != _sha256_bytes(_canonical(payload)):
            raise AnalysisError(f"shard checksum drift: {path.name}")
        payloads.append(payload)
    if len(payloads) != complete.get("rows") or sum(p.get("status") == "ok" for p in payloads) != complete.get("analyzable_rows"):
        raise AnalysisError("shard count differs from COMPLETE marker")
    rows = [payload for payload in payloads if payload.get("status") == "ok"]
    if not rows:
        raise AnalysisError("replay contains no analyzable rows")
    layer_contracts = {tuple(row["signals"]["layer_ids"]) for row in rows}
    if len(layer_contracts) != 1 or len(next(iter(layer_contracts))) < 2:
        raise AnalysisError("replay layer identities are inconsistent or insufficient")
    return rows, config


def _features(row: dict[str, Any]) -> dict[str, Any]:
    signals = row["signals"]
    own = np.asarray(signals["own_image"]["constraint_minus_matched"], dtype=float)
    swap = np.asarray(signals["swap_images"]["mean_constraint_minus_matched"], dtype=float)
    visual = np.asarray(signals["primary_own_minus_swap_difference_in_differences"], dtype=float)
    text = np.asarray(signals["text_only_secondary"]["constraint_minus_matched"], dtype=float)
    lengths = {len(own), len(swap), len(visual), len(text)}
    if len(lengths) != 1 or next(iter(lengths)) < 2 or not all(
        np.isfinite(values).all() for values in (own, swap, visual, text)
    ):
        raise AnalysisError(f"{row.get('sample_id')}: malformed layer signals")
    counts = signals["token_counts"]
    return {
        **row,
        "own_shift": float(own[-1] - own[0]),
        "swap_shift": float(swap[-1] - swap[0]),
        "visual_early": float(visual[0]),
        "visual_shift": float(visual[-1] - visual[0]),
        "text_shift": float(text[-1] - text[0]),
        "full_tokens": int(counts["full_visible_answer"]),
        "constraint_tokens": int(counts["constraint"]),
    }


def _nuisance_spec(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categorical = ("edge_type", "modality_stratum", "anatomy_stratum")
    levels: dict[str, list[str]] = {}
    for field in categorical:
        counts = Counter(str(row[field]) for row in rows)
        retained = sorted(value for value, count in counts.items() if count >= 5)
        levels[field] = retained if retained else ["__OTHER__"]
    return {
        "continuous_means": {
            "log_full_tokens": float(np.mean([np.log1p(row["full_tokens"]) for row in rows])),
            "log_constraint_tokens": float(np.mean([np.log1p(row["constraint_tokens"]) for row in rows])),
        },
        "categorical_levels_min_count_5": levels,
    }


def _design(rows: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    names = ["intercept", "log_full_tokens", "log_constraint_tokens", "prompt_requested_increment"]
    columns = [
        np.ones(len(rows)),
        np.asarray([np.log1p(row["full_tokens"]) for row in rows]) - spec["continuous_means"]["log_full_tokens"],
        np.asarray([np.log1p(row["constraint_tokens"]) for row in rows]) - spec["continuous_means"]["log_constraint_tokens"],
        np.asarray([float(bool(row["prompt_requested_increment"])) for row in rows]),
    ]
    for field, retained in spec["categorical_levels_min_count_5"].items():
        mapped = [str(row[field]) if str(row[field]) in retained else "__OTHER__" for row in rows]
        all_levels = sorted(set(retained) | ({"__OTHER__"} if "__OTHER__" in mapped else set()))
        for level in all_levels[1:]:
            names.append(f"{field}={level}")
            columns.append(np.asarray([float(value == level) for value in mapped]))
    return np.column_stack(columns), names


def freeze_dev_spec(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = [_features(row) for row in rows if row.get("scientific_role") in PRIMARY_ROLES]
    case_counts = {
        role: len({row["case_id"] for row in rows if row["scientific_role"] == role})
        for role in sorted(PRIMARY_ROLES)
    }
    if min(case_counts.values(), default=0) < MIN_CASES_PER_ROLE:
        raise AnalysisError(f"dev has fewer than {MIN_CASES_PER_ROLE} cases per primary role: {case_counts}")
    nuisance = _nuisance_spec(rows)
    matrix, names = _design(rows, nuisance)
    coefficients: dict[str, list[float]] = {}
    for endpoint in ("own_shift", "swap_shift", "visual_early", "visual_shift", "text_shift"):
        coefficients[endpoint] = np.linalg.lstsq(
            matrix, np.asarray([row[endpoint] for row in rows]), rcond=None
        )[0].tolist()
    adjusted_visual = np.asarray([row["visual_shift"] for row in rows]) - matrix[:, 1:] @ np.asarray(coefficients["visual_shift"])[1:]
    scale = float(np.std(adjusted_visual, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise AnalysisError("dev visual-shift scale is not positive")
    return {
        "protocol": PROTOCOL,
        "status": "dev_spec_frozen",
        "runtime_identity": {
            "manifest_sha256": config["manifest_sha256"],
            "metadata_sha256": config["metadata_sha256"],
            "identity_canary_sha256": config["identity_canary_sha256"],
            "adapter_fingerprint": config["adapter_fingerprint"],
        },
        "layers": {"early": rows[0]["signals"]["layer_ids"][0], "late": rows[0]["signals"]["layer_ids"][-1]},
        "dev_case_ids": sorted({row["case_id"] for row in rows}),
        "dev_case_counts_by_role": case_counts,
        "nuisance": nuisance,
        "design_columns": names,
        "coefficients": coefficients,
        "equivalence_margin": EQUIVALENCE_SD_MULTIPLIER * scale,
        "equivalence_rule": "two-sided 95% CI wholly within +/- 0.2 dev SD",
        "bootstrap": {"cluster": "case_id", "draws": 5000, "seed": 42},
    }


def _adjust(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    matrix, names = _design(rows, spec["nuisance"])
    if names != spec["design_columns"]:
        raise AnalysisError("test nuisance design differs from frozen dev design")
    output = []
    for index, row in enumerate(rows):
        item = dict(row)
        for endpoint, values in spec["coefficients"].items():
            item[endpoint] = float(row[endpoint] - matrix[index, 1:] @ np.asarray(values)[1:])
        output.append(item)
    return output


def _statistics(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_role = {role: [row for row in rows if row["scientific_role"] == role] for role in PRIMARY_ROLES}
    if any(not values for values in by_role.values()):
        raise AnalysisError("bootstrap draw lost a primary role")
    mean: Callable[[str, str], float] = lambda role, key: float(np.mean([row[key] for row in by_role[role]]))
    return {
        "early_control_minus_error_visual": mean(CONTROL, "visual_early") - mean(ERROR, "visual_early"),
        "error_minus_control_own_shift": mean(ERROR, "own_shift") - mean(CONTROL, "own_shift"),
        "error_swap_shift": mean(ERROR, "swap_shift"),
        "error_visual_shift": mean(ERROR, "visual_shift"),
        "error_minus_control_text_shift_secondary": mean(ERROR, "text_shift") - mean(CONTROL, "text_shift"),
    }


def analyze_test(rows: list[dict[str, Any]], config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("protocol") != PROTOCOL or spec.get("status") != "dev_spec_frozen":
        raise AnalysisError("analysis spec is not frozen")
    identity = {key: config[key] for key in ("manifest_sha256", "metadata_sha256", "identity_canary_sha256", "adapter_fingerprint")}
    if identity != spec["runtime_identity"]:
        raise AnalysisError("test runtime identity differs from frozen dev runtime")
    rows = [_features(row) for row in rows if row.get("scientific_role") in PRIMARY_ROLES]
    if set(spec["dev_case_ids"]) & {row["case_id"] for row in rows}:
        raise AnalysisError("dev/test case leakage")
    case_counts = {role: len({row["case_id"] for row in rows if row["scientific_role"] == role}) for role in sorted(PRIMARY_ROLES)}
    if min(case_counts.values(), default=0) < MIN_CASES_PER_ROLE:
        raise AnalysisError(f"test has fewer than {MIN_CASES_PER_ROLE} cases per primary role: {case_counts}")
    adjusted = _adjust(rows, spec)
    point = _statistics(adjusted)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adjusted:
        by_case[row["case_id"]].append(row)
    cases = sorted(by_case)
    rng = np.random.default_rng(int(spec["bootstrap"]["seed"]))
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(spec["bootstrap"]["draws"])):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        replicate = [row for case in sampled for row in by_case[str(case)]]
        try:
            values = _statistics(replicate)
        except AnalysisError:
            continue
        for key, value in values.items():
            draws[key].append(value)
    if min((len(values) for values in draws.values()), default=0) < 0.95 * int(spec["bootstrap"]["draws"]):
        raise AnalysisError("too many invalid clustered bootstrap draws")
    estimates = {
        key: {
            "estimate": value,
            "ci95": [float(np.quantile(draws[key], 0.025)), float(np.quantile(draws[key], 0.975))],
        }
        for key, value in point.items()
    }
    margin = float(spec["equivalence_margin"])
    gates = {
        "early_supported_visual_advantage": estimates["early_control_minus_error_visual"]["ci95"][0] > 0,
        "error_selective_own_late_shift": estimates["error_minus_control_own_shift"]["ci95"][0] > 0,
        "swapped_image_late_shift_present": estimates["error_swap_shift"]["ci95"][0] > 0,
        "no_late_visual_residual_equivalent": estimates["error_visual_shift"]["ci95"][0] > -margin and estimates["error_visual_shift"]["ci95"][1] < margin,
    }
    return {
        "protocol": PROTOCOL,
        "status": "mechanism_gate_passed" if all(gates.values()) else "mechanism_gate_failed",
        "case_counts_by_role": case_counts,
        "rows": len(adjusted),
        "estimates": estimates,
        "equivalence_margin": margin,
        "gates": gates,
        "all_primary_gates_pass": all(gates.values()),
        "text_only_is_secondary": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--dev-run", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    test = sub.add_parser("test")
    test.add_argument("--test-run", type=Path, required=True)
    test.add_argument("--spec", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            rows, config = load_runtime(args.dev_run)
            result = freeze_dev_spec(rows, config)
        else:
            rows, config = load_runtime(args.test_run)
            result = analyze_test(rows, config, json.loads(args.spec.read_text()))
        _write_once(args.output, result)
    except (AnalysisError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
