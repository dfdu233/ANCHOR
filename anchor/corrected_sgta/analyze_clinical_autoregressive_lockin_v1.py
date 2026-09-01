#!/usr/bin/env python3
"""Preregistered dev analyzer for Clinical Autoregressive Lock-in v1.

No threshold or layer is selected from the result.  Decoder quartiles, prefix
steps, bootstrap unit, directional admission, and kill rules are constants.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .clinical_autoregressive_lockin_probe_v1 import (
    RUNTIME_PROTOCOL_ID,
    ContractError,
    _canonical,
    _sha,
    _sha_file,
    _write_once_or_equal,
)


ANALYSIS_PROTOCOL_ID = "clinical-autoregressive-lockin-analysis-v1"
QUARTILES = (0.25, 0.50, 0.75)
PRIMARY_EARLY_STEP = 2  # exact common prefix: "The chest X-ray shows "
PRIMARY_LATE_STEP = 4  # pilot-frozen claim-specific modifier completed
MIN_MACRO_AUC = 0.70
MIN_FINDING_AUC = 0.60
MIN_RELATIVE_COLLAPSE = 0.50
MIN_ROWS_PER_CELL = 10
MIN_CROSS_SUPPORT_EMBEDDED_SURFACE_RATE = 0.25


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or set(y.tolist()) != {0, 1}:
        return float("nan")
    positive = s[y == 1]
    negative = s[y == 0]
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _nearest_layers(fractions: Sequence[float]) -> dict[str, int]:
    values = np.asarray(fractions, dtype=float)
    output = {}
    for target in QUARTILES:
        index = int(np.argmin(np.abs(values - target)))
        if abs(float(values[index]) - target) > 0.08:
            raise ContractError(f"no declared layer within 0.08 of decoder fraction {target}")
        output[f"q{int(target * 100)}"] = index
    if len(set(output.values())) != len(output):
        raise ContractError("decoder quartiles map to duplicate layers")
    output["final"] = len(values) - 1
    return output


def _load_payloads(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "config.json"
    controls_path = run_dir / "controls.json"
    if not all(path.is_file() for path in (complete_path, config_path, controls_path)):
        raise ContractError("runtime COMPLETE/config/controls artifacts are inseparable")
    complete = json.loads(complete_path.read_text())
    config = json.loads(config_path.read_text())
    controls = json.loads(controls_path.read_text())
    if complete.get("runtime_protocol_id") != RUNTIME_PROTOCOL_ID:
        raise ContractError("wrong runtime protocol")
    if not complete.get("analysis_input_complete"):
        raise ContractError("runtime exclusions failed the preregistered cell-size gate")
    if complete.get("scientific_gate_authorized") is not False:
        raise ContractError("runtime improperly pre-authorized a scientific mechanism gate")
    payloads = []
    for path in sorted((run_dir / "shards").glob("*.json")):
        shard = json.loads(path.read_text())
        if shard.get("config_fingerprint") != config.get("config_fingerprint"):
            raise ContractError("shard/config fingerprint mismatch")
        payload = shard.get("payload")
        if shard.get("payload_sha256") != _sha(_canonical(payload)):
            raise ContractError("shard payload checksum mismatch")
        if payload.get("status") == "ok":
            payloads.append(payload)
    if len(payloads) != complete.get("analyzable_rows"):
        raise ContractError("analyzable shard count differs from COMPLETE")
    reference = payloads[0]
    identity = (reference["layer_ids"], reference["layer_fractions"], reference["template_id"])
    if any(
        (row["layer_ids"], row["layer_fractions"], row["template_id"]) != identity
        for row in payloads[1:]
    ):
        raise ContractError("payload layer/template identity differs")
    return payloads, config, controls


def _step(row: dict[str, Any], step: int) -> dict[str, Any]:
    values = [value for value in row["prefix_ladder"] if value["step"] == step]
    if len(values) != 1:
        raise ContractError("payload prefix ladder is incomplete or duplicated")
    return values[0]


def _mean(rows: Iterable[dict[str, Any]], getter) -> float:
    values = [float(getter(row)) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if bool((norms <= 1e-12).any()):
        raise ContractError("zero-norm prompt-end hidden state")
    return values / norms


def _centroid_direction(
    fit_rows: Sequence[dict[str, Any]], layer_index: int
) -> tuple[np.ndarray, float]:
    features = []
    labels = []
    for row in fit_rows:
        label = int(row["positive_votes"] == 3)
        for variant in ("original", "same_support_swap"):
            features.append(row["prompt_end_readout"]["layer_hidden"][variant][layer_index])
            labels.append(label)
    matrix = _unit_rows(np.asarray(features, dtype=float))
    target = np.asarray(labels, dtype=int)
    if set(target.tolist()) != {0, 1}:
        raise ContractError("prompt-end fit split lost one reader-polarity class")
    direction = matrix[target == 1].mean(axis=0) - matrix[target == 0].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ContractError("prompt-end centroid direction has zero norm")
    return direction / norm, norm


def _prompt_end_probe(
    rows: Sequence[dict[str, Any]], layer_index: int, *, shuffled_controls: int = 256
) -> dict[str, Any]:
    finding_aucs = {}
    causal_excess = {}
    shuffled_macro: list[float] = []
    directions = {}
    direction_norms = {}
    text_only_vectors: dict[str, list[list[float]]] = defaultdict(list)
    text_only_provenance_ok = True
    for row in rows:
        control = row["prompt_end_readout"].get("text_only_control", {})
        text_only_provenance_ok = text_only_provenance_ok and (
            control.get("same_prompt_no_image") is True
            and control.get("used_to_fit_or_select_prompt_end_probe") is False
            and control.get("token_identity_required") is False
            and control.get("layer_and_hidden_dimension_alignment_required") is True
        )
        text_only_vectors[row["finding"]].append(
            row["prompt_end_readout"]["layer_hidden"]["text_only"][layer_index]
        )
    text_only_deviation_by_finding = {}
    for finding, values in text_only_vectors.items():
        text_matrix = np.asarray(values, dtype=float)
        text_only_deviation_by_finding[finding] = float(
            np.max(np.linalg.norm(text_matrix - text_matrix[0], axis=1))
        )
    text_only_max_deviation = max(text_only_deviation_by_finding.values())
    text_only_invariant = text_only_provenance_ok and all(
        value <= 1e-8 for value in text_only_deviation_by_finding.values()
    )
    for finding in sorted({row["finding"] for row in rows}):
        fit = [
            row
            for row in rows
            if row["finding"] == finding and row["prompt_end_probe_role"] == "probe_fit"
        ]
        evaluate = [
            row
            for row in rows
            if row["finding"] == finding and row["prompt_end_probe_role"] == "probe_eval"
        ]
        if not fit or not evaluate:
            raise ContractError("prompt-end fit/eval block split is empty")
        direction, direction_norm = _centroid_direction(fit, layer_index)
        directions[finding] = direction
        direction_norms[finding] = direction_norm
        labels = []
        scores = []
        row_excess = []
        for row in evaluate:
            sign = 1.0 if row["positive_votes"] == 3 else -1.0
            variant_scores = {}
            for variant in ("original", "same_support_swap", "opposite_support_swap"):
                hidden = _unit_rows(
                    np.asarray(
                        [row["prompt_end_readout"]["layer_hidden"][variant][layer_index]],
                        dtype=float,
                    )
                )[0]
                variant_scores[variant] = float(hidden @ direction)
            # Both original and same-support swap are unique eval DICOMs.
            labels.extend([int(row["positive_votes"] == 3)] * 2)
            scores.extend([variant_scores["original"], variant_scores["same_support_swap"]])
            opposite = sign * (
                variant_scores["original"] - variant_scores["opposite_support_swap"]
            )
            same = abs(variant_scores["original"] - variant_scores["same_support_swap"])
            row_excess.append(opposite - same)
        finding_aucs[finding] = _auc(labels, scores)
        causal_excess[finding] = float(np.mean(row_excess))

    # Frozen block-level label flips retain class balance within every fit block.
    # They test claim-specific decoding rather than high-dimensional memorization.
    for permutation in range(shuffled_controls):
        permuted_statistics = []
        for finding in sorted(directions):
            fit = [
                row
                for row in rows
                if row["finding"] == finding and row["prompt_end_probe_role"] == "probe_fit"
            ]
            evaluate = [
                row
                for row in rows
                if row["finding"] == finding and row["prompt_end_probe_role"] == "probe_eval"
            ]
            # A single label per block would collapse its two paired rows to the
            # same class.  Instead use a row-aware proxy key for exact exchange.
            row_labels = {
                f"{row['block_id']}|{row['sample_id']}": (
                    int(row["positive_votes"] == 3)
                    ^ (int(_sha(f"prompt-end-shuffle|{permutation}|{finding}|{row['block_id']}".encode())[:8], 16) % 2)
                )
                for row in fit
            }
            features = []
            shuffled_labels = []
            for row in fit:
                for variant in ("original", "same_support_swap"):
                    features.append(row["prompt_end_readout"]["layer_hidden"][variant][layer_index])
                    shuffled_labels.append(row_labels[f"{row['block_id']}|{row['sample_id']}"])
            matrix = _unit_rows(np.asarray(features, dtype=float))
            target = np.asarray(shuffled_labels, dtype=int)
            if set(target.tolist()) != {0, 1}:
                continue
            direction = matrix[target == 1].mean(axis=0) - matrix[target == 0].mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction /= norm
            labels = []
            scores = []
            for row in evaluate:
                for variant in ("original", "same_support_swap"):
                    hidden = _unit_rows(
                        np.asarray(
                            [row["prompt_end_readout"]["layer_hidden"][variant][layer_index]],
                            dtype=float,
                        )
                    )[0]
                    labels.append(int(row["positive_votes"] == 3))
                    scores.append(float(hidden @ direction))
            auc = _auc(labels, scores)
            permuted_statistics.append((2.0 * auc - 1.0) * norm)
        if permuted_statistics:
            shuffled_macro.append(float(np.mean(permuted_statistics)))
    true_statistic = float(
        np.mean(
            [
                (2.0 * finding_aucs[finding] - 1.0) * direction_norms[finding]
                for finding in sorted(finding_aucs)
            ]
        )
    )
    return {
        "finding_auc": finding_aucs,
        "macro_auc": float(np.mean(list(finding_aucs.values()))),
        "causal_excess_by_finding": causal_excess,
        "polarity_causal_excess": float(np.mean(list(causal_excess.values()))),
        "fit_direction_norm_by_finding": direction_norms,
        "heldout_auc_times_fit_magnitude_statistic": true_statistic,
        "shuffled_block_label_statistic_95pct": (
            float(np.quantile(shuffled_macro, 0.95)) if shuffled_macro else float("nan")
        ),
        "fit_uses_only_prompt_end_hidden": True,
        "teacher_forced_likelihood_used_for_gate1": False,
        "same_support_swap_used_in_eval": True,
        "block_label_shuffle_used": True,
        "text_only_prompt_end_control": {
            "provenance_ok": text_only_provenance_ok,
            "max_hidden_deviation_same_prompt": text_only_max_deviation,
            "max_hidden_deviation_by_claim_specific_prompt": text_only_deviation_by_finding,
            "invariant": text_only_invariant,
            "used_to_fit_probe": False,
        },
    }


def _changepoint(profile: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(profile, dtype=float)
    if len(values) < 4 or not np.isfinite(values).all():
        return {"step": None, "sse": None, "drop": None}
    candidates = []
    for split in range(1, len(values)):
        left, right = values[:split], values[split:]
        sse = float(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())
        candidates.append((sse, split, float(left.mean() - right.mean())))
    sse, split, drop = min(candidates, key=lambda item: (item[0], item[1]))
    return {"step": split, "sse": sse, "drop": drop}


def _metric_bundle(
    rows: Sequence[dict[str, Any]], layer_index: int, *, shuffled_controls: int = 256
) -> dict[str, Any]:
    findings = sorted({row["finding"] for row in rows})
    prompt_end = _prompt_end_probe(rows, layer_index, shuffled_controls=shuffled_controls)
    nonattractor_template_excess = _mean(
        rows,
        lambda row: row["non_attractor_preclaim_template_control"][
            "causal_excess_over_same_support"
        ][layer_index],
    )
    early_by_finding = {}
    late_by_finding = {}
    decline_by_finding = {}
    text_attraction_by_finding = {}
    changepoints = {}
    length_residuals = {}
    for finding in findings:
        group = [row for row in rows if row["finding"] == finding]
        profile = [
            _mean(
                group,
                lambda row, step=step: _step(row, step)["effects"][
                    "causal_excess_over_same_support"
                ][layer_index],
            )
            for step in range(5)
        ]
        early, late = profile[PRIMARY_EARLY_STEP], profile[PRIMARY_LATE_STEP]
        early_by_finding[finding] = early
        late_by_finding[finding] = late
        decline_by_finding[finding] = early - late
        text_attraction_by_finding[finding] = _mean(
            group,
            lambda row: _step(row, PRIMARY_LATE_STEP)["layer_mean_logp"]["text_only"][layer_index]
            - _step(row, PRIMARY_EARLY_STEP)["layer_mean_logp"]["text_only"][layer_index],
        )
        changepoints[finding] = _changepoint(profile)
        prefix_lengths = np.asarray(
            [len(_step(group[0], step)["prefix_token_ids"]) for step in range(3)],
            dtype=float,
        )
        coefficients = np.polyfit(prefix_lengths, np.asarray(profile[:3]), deg=1)
        late_length = len(_step(group[0], PRIMARY_LATE_STEP)["prefix_token_ids"])
        predicted = float(np.polyval(coefficients, late_length))
        length_residuals[finding] = late - predicted
    macro_early = float(np.mean(list(early_by_finding.values())))
    macro_late = float(np.mean(list(late_by_finding.values())))
    relative = (macro_early - macro_late) / max(abs(macro_early), 1e-8)
    return {
        "prompt_end_probe": prompt_end,
        "finding_auc": prompt_end["finding_auc"],
        "macro_auc": prompt_end["macro_auc"],
        "polarity_causal_excess": prompt_end["polarity_causal_excess"],
        "nonattractor_template_causal_excess": nonattractor_template_excess,
        "early_causal_excess_by_finding": early_by_finding,
        "late_causal_excess_by_finding": late_by_finding,
        "decline_by_finding": decline_by_finding,
        "macro_early_causal_excess": macro_early,
        "macro_late_causal_excess": macro_late,
        "relative_collapse": relative,
        "text_only_attraction_by_finding": text_attraction_by_finding,
        "prefix_changepoint_by_finding": changepoints,
        "late_residual_after_common_prefix_length_trend": length_residuals,
    }


def _bootstrap_rows(rows: Sequence[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    output = []
    for finding in sorted({row["finding"] for row in rows}):
        blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["finding"] == finding:
                blocks[row["block_id"]].append(row)
        block_ids = sorted(blocks)
        sampled = rng.choice(block_ids, size=len(block_ids), replace=True)
        for replicate, block_id in enumerate(sampled):
            for row in blocks[str(block_id)]:
                clone = dict(row)
                clone["block_id"] = f"bootstrap-{finding}-{replicate}"
                output.append(clone)
    return output


def _interval(values: Sequence[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def analyze_payloads(
    rows: Sequence[dict[str, Any]], *, bootstrap_replicates: int = 2000, seed: int = 20260802
) -> dict[str, Any]:
    if bootstrap_replicates < 100:
        raise ContractError("at least 100 block-bootstrap replicates are required")
    cells = CounterKey((row["finding"], row["positive_votes"]) for row in rows)
    if min(cells.values()) < MIN_ROWS_PER_CELL:
        raise ContractError("analyzable cell fell below 10 anchors")
    layer_map = _nearest_layers(rows[0]["layer_fractions"])
    generation_endpoint = {}
    for finding in sorted({row["finding"] for row in rows}):
        blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["finding"] == finding:
                blocks[row["block_id"]].append(row)
        embedded = []
        full_text = []
        for block_rows in blocks.values():
            embedded_values = {
                bool(row["generation_endpoint"]["original_opposite_same_embedded_claim_surface"])
                for row in block_rows
            }
            full_values = {
                bool(row["generation_endpoint"]["original_opposite_exact_full_text_collision"])
                for row in block_rows
            }
            if len(embedded_values) != 1 or len(full_values) != 1:
                raise ContractError("paired block generation endpoint is not symmetric")
            embedded.append(embedded_values.pop())
            full_text.append(full_values.pop())
        generation_endpoint[finding] = {
            "independent_blocks": len(blocks),
            "cross_support_same_embedded_claim_surface_rate": float(np.mean(embedded)),
            "cross_support_exact_full_text_collision_rate": float(np.mean(full_text)),
            "surface_rate_gate_pass": float(np.mean(embedded))
            >= MIN_CROSS_SUPPORT_EMBEDDED_SURFACE_RATE,
            "clinical_correctness_metric": False,
        }
    generation_endpoint_pass = all(
        value["surface_rate_gate_pass"] for value in generation_endpoint.values()
    )
    point = {name: _metric_bundle(rows, index) for name, index in layer_map.items()}
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in layer_map
    }
    for _ in range(bootstrap_replicates):
        sampled = _bootstrap_rows(rows, rng)
        for name, index in layer_map.items():
            metrics = _metric_bundle(sampled, index, shuffled_controls=0)
            bootstrap[name]["macro_auc"].append(metrics["macro_auc"])
            bootstrap[name]["polarity_causal_excess"].append(metrics["polarity_causal_excess"])
            bootstrap[name]["macro_early_causal_excess"].append(
                metrics["macro_early_causal_excess"]
            )
            bootstrap[name]["macro_decline"].append(
                metrics["macro_early_causal_excess"] - metrics["macro_late_causal_excess"]
            )
            bootstrap[name]["text_attraction"].append(
                float(np.mean(list(metrics["text_only_attraction_by_finding"].values())))
            )
            bootstrap[name]["nonattractor_template_causal_excess"].append(
                metrics["nonattractor_template_causal_excess"]
            )
    layer_results = {}
    admitted_layers = []
    lockin_layers = []
    for name, metrics in point.items():
        intervals = {key: _interval(values) for key, values in bootstrap[name].items()}
        admission = (
            metrics["macro_auc"] >= MIN_MACRO_AUC
            and min(metrics["finding_auc"].values()) >= MIN_FINDING_AUC
            and metrics["prompt_end_probe"]["heldout_auc_times_fit_magnitude_statistic"]
            > metrics["prompt_end_probe"]["shuffled_block_label_statistic_95pct"]
            and metrics["prompt_end_probe"]["text_only_prompt_end_control"]["invariant"]
            and intervals["macro_auc"][0] > 0.5
            and intervals["polarity_causal_excess"][0] > 0.0
        )
        modifier_changepoints = all(
            value["step"] in {3, 4} and value["drop"] is not None and value["drop"] > 0
            for value in metrics["prefix_changepoint_by_finding"].values()
        )
        length_control = all(
            value < 0
            for value in metrics["late_residual_after_common_prefix_length_trend"].values()
        )
        lockin = (
            admission
            and min(metrics["early_causal_excess_by_finding"].values()) > 0
            and metrics["relative_collapse"] >= MIN_RELATIVE_COLLAPSE
            and intervals["macro_decline"][0] > 0
            and intervals["text_attraction"][0] > 0
            and intervals["nonattractor_template_causal_excess"][0] > 0
            and modifier_changepoints
            and length_control
        )
        if name != "final" and admission:
            admitted_layers.append(name)
        if name != "final" and lockin:
            lockin_layers.append(name)
        layer_results[name] = {
            "layer_index": layer_map[name],
            "layer_id": rows[0]["layer_ids"][layer_map[name]],
            "layer_fraction": rows[0]["layer_fractions"][layer_map[name]],
            "point": metrics,
            "bootstrap_95ci": intervals,
            "directional_admission_pass": admission,
            "lockin_layer_pass": lockin,
            "modifier_changepoint_pass": modifier_changepoints,
            "smooth_prefix_length_null_rejected_directionally": length_control,
        }
    directional_gate = len(admitted_layers) >= 2
    lockin_gate = directional_gate and len(lockin_layers) >= 2 and generation_endpoint_pass
    legacy_counterfactual_decision = (
        "would_have_passed_legacy_numeric_gates"
        if lockin_gate
        else "would_have_failed_legacy_numeric_gates"
    )
    return {
        "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
        "bootstrap": {
            "unit": "independent finding-specific four-DICOM block",
            "stratified_by_finding": True,
            "replicates": bootstrap_replicates,
            "seed": seed,
        },
        "frozen_primary_steps": {
            "early": PRIMARY_EARLY_STEP,
            "late": PRIMARY_LATE_STEP,
            "allowed_changepoint_steps": [3, 4],
        },
        "frozen_thresholds": {
            "minimum_macro_polarity_auc": MIN_MACRO_AUC,
            "minimum_per_finding_polarity_auc": MIN_FINDING_AUC,
            "minimum_relative_collapse": MIN_RELATIVE_COLLAPSE,
            "minimum_passing_nonfinal_quartile_layers": 2,
            "minimum_cross_support_same_embedded_claim_surface_rate_per_finding": (
                MIN_CROSS_SUPPORT_EMBEDDED_SURFACE_RATE
            ),
        },
        "generation_endpoint": generation_endpoint,
        "generation_endpoint_pass": generation_endpoint_pass,
        "layer_results": layer_results,
        "directional_admission_pass": directional_gate,
        "legacy_counterfactual_lockin_gate": lockin_gate,
        "legacy_counterfactual_decision": legacy_counterfactual_decision,
        "lockin_mechanism_pass": False,
        "decision": "rejected_f6_construct_invalid",
        "confirmation_or_patching_authorized": False,
        "forensic_only_f6_rejected": True,
        "claim_scope": "Huatuo VinDr pleural-effusion and lung-opacity embedded claims only",
    }


def _random_pair_control(
    rows: Sequence[dict[str, Any]],
    mapping: dict[str, str],
    layer_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    if set(mapping) != set(by_id) or any(source not in by_id for source in mapping.values()):
        raise ContractError("random-pair mapping does not exactly cover analyzable rows")
    output = {}
    for name, layer_result in layer_results.items():
        layer_index = int(layer_result["layer_index"])
        per_step = []
        for step_index in range(5):
            values = []
            for target_id, source_id in mapping.items():
                target = by_id[target_id]
                source = by_id[source_id]
                if target["block_id"] == source["block_id"]:
                    raise ContractError("random-pair control reuses the target independent block")
                sign = 1.0 if target["positive_votes"] == 3 else -1.0
                target_step = _step(target, step_index)
                source_step = _step(source, step_index)
                target_score = target_step["layer_mean_logp"]["original"][layer_index]
                source_score = source_step["layer_mean_logp"]["original"][layer_index]
                same_drift = target_step["effects"]["absolute_original_minus_same"][layer_index]
                values.append(sign * (target_score - source_score) - same_drift)
            per_step.append(float(np.mean(values)))
        early = per_step[PRIMARY_EARLY_STEP]
        late = per_step[PRIMARY_LATE_STEP]
        primary = layer_result["point"]
        pass_control = (
            primary["macro_early_causal_excess"] > early
            and (
                primary["macro_early_causal_excess"]
                - primary["macro_late_causal_excess"]
            )
            > (early - late)
        )
        output[name] = {
            "mean_causal_excess_by_step": per_step,
            "early": early,
            "late": late,
            "decline": early - late,
            "primary_exceeds_random_at_early_and_decline": pass_control,
        }
    return output


class CounterKey(dict):
    def __init__(self, values: Iterable[tuple[Any, Any]]):
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


def analyze_run(
    run_dir: Path,
    output: Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20260802,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    raise ContractError(
        "clinical-lockin-f6-unnatural-prefix-continuation-rejection-v1: "
        "formal v4 analysis is forbidden; no early/late result is scientifically admissible"
    )
    # Unreachable legacy analyzer retained for forensic reproducibility only.
    rows, config, controls = _load_payloads(run_dir)
    result = analyze_payloads(rows, bootstrap_replicates=bootstrap_replicates, seed=seed)
    random_mapping = controls["random_pair_control"]["mapping_target_to_source"]
    random_control = _random_pair_control(rows, random_mapping, result["layer_results"])
    result["random_pair_control"] = random_control
    for name, layer in result["layer_results"].items():
        random_pass = random_control[name]["primary_exceeds_random_at_early_and_decline"]
        layer["random_pair_control_pass"] = random_pass
        layer["lockin_layer_pass"] = bool(layer["lockin_layer_pass"] and random_pass)
    passing = sum(
        name != "final" and layer["lockin_layer_pass"]
        for name, layer in result["layer_results"].items()
    )
    result["lockin_mechanism_pass"] = bool(
        result["directional_admission_pass"]
        and result["generation_endpoint_pass"]
        and passing >= 2
    )
    result["decision"] = (
        "go_to_fresh_confirmation_and_selective_patching"
        if result["lockin_mechanism_pass"]
        else "kill_lockin_mechanism"
    )
    result["confirmation_or_patching_authorized"] = result["lockin_mechanism_pass"]
    result["cross_claim_exact_prefix_length_diagnostic"] = controls[
        "length_control"
    ]["cross_claim_step_matches"]
    result["cross_claim_length_matching_used_for_gate"] = False
    result["runtime_config_fingerprint"] = config["config_fingerprint"]
    result["controls_sha256"] = _sha_file(run_dir / "controls.json")
    result["analyzer_source_sha256"] = _sha_file(Path(__file__))
    result["exact_command"] = list(command or [])
    result["threshold_contract_sha256"] = _sha(
        _canonical(
            {
                "analysis_protocol_id": result["analysis_protocol_id"],
                "frozen_primary_steps": result["frozen_primary_steps"],
                "frozen_thresholds": result["frozen_thresholds"],
                "bootstrap": result["bootstrap"],
            }
        )
    )
    result["analysis_fingerprint"] = _sha(_canonical(result))
    _write_once_or_equal(
        output,
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    result = analyze_run(
        args.run_dir,
        args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        command=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
