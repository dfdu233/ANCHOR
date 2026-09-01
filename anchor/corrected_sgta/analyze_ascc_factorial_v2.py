#!/usr/bin/env python3
"""Pre-outcome, fail-closed analysis for the ASCC-v2 symmetric 2x2 assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit_diagnostic_completion_substrate_v1 import sha256_file
from .prepare_ascc_factorial_v2 import (
    MARKERS as FROZEN_MARKERS,
    PROMPTS as FROZEN_PROMPTS,
    VERSION as SUBSTRATE_VERSION,
)
from .prepare_pragmatic_commitment_confirmatory_v1 import canonical_hash
from .run_huatuo_ascc_factorial_v2 import (
    PRIMARY_EDGE,
    VERSION as SCORE_VERSION,
    load_jsonl,
    record_key,
    tree_fingerprint,
)


VERSION = "ascc-symmetric-factorial-analysis-v2.1-blind-locked"
EQUIVALENCE_BOUND = 0.2
MINIMUM_MEANINGFUL_DID = float(np.log(1.5))
MINIMUM_MARKER_TOP1_RATE = 0.90
MINIMUM_AFFINE_R2 = 0.50
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.99
AFFINE_SLOPE_RANGE = (0.5, 2.0)
MAX_CLEAR_COMMITMENT_ABS_BIAS = 0.10
MAX_CLEAR_COMMITMENT_RMSE = 0.20
FROZEN_SEED = 99173
FROZEN_BOOTSTRAP_ITERATIONS = 5000
LOCAL_BOUNDARIES = {
    "negative_boundary": (1, 0),
    "positive_boundary": (2, 3),
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def interval(values: Sequence[float], level: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    return {
        "level": level,
        "low": float(np.quantile(array, alpha)),
        "high": float(np.quantile(array, 1.0 - alpha)),
    }


def validate_inputs(
    substrate_dir: Path, score_dir: Path, image_root: Path, edge_id: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    substrate_path = substrate_dir / "substrate_config.json"
    manifest_path = substrate_dir / "selected_manifest.jsonl"
    score_path = score_dir / "score_config.json"
    marker_contract_path = score_dir / "marker_token_contract.json"
    for path in (substrate_path, manifest_path, score_path, marker_contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    substrate = json.loads(substrate_path.read_text())
    if substrate.get("version") != SUBSTRATE_VERSION:
        raise ValueError("wrong ASCC-v2 substrate version")
    if substrate.get("status") != "untouched_confirmatory_census_gpu_not_run":
        raise ValueError("ASCC-v2 substrate status is not the frozen untouched state")
    if substrate.get("selection_uses_model_output") is not False:
        raise ValueError("ASCC-v2 selection used model output")
    if substrate.get("discovery_exclusion_uses_generation_output") is not False:
        raise ValueError("ASCC-v2 discovery exclusion used generation outcomes")
    substrate_payload = {key: value for key, value in substrate.items() if key != "fingerprint"}
    if canonical_hash(substrate_payload) != substrate.get("fingerprint"):
        raise ValueError("ASCC-v2 substrate fingerprint mismatch")
    if sha256_file(manifest_path) != substrate.get("manifest_sha256"):
        raise ValueError("ASCC-v2 manifest hash mismatch")
    score = json.loads(score_path.read_text())
    if score.get("version") != SCORE_VERSION:
        raise ValueError("wrong ASCC-v2 score version")
    score_payload = {
        key: value
        for key, value in score.items()
        if key not in {"fingerprint", "created_at", "command"}
    }
    if canonical_hash(score_payload) != score.get("fingerprint"):
        raise ValueError("ASCC-v2 score config fingerprint mismatch")
    if score.get("substrate_fingerprint") != substrate.get("fingerprint"):
        raise ValueError("ASCC-v2 score/substrate fingerprint mismatch")
    if score.get("substrate_manifest_sha256") != substrate.get("manifest_sha256"):
        raise ValueError("ASCC-v2 score/substrate manifest mismatch")
    if score.get("prompts") != substrate.get("prompts") or score.get("markers") != substrate.get("markers"):
        raise ValueError("ASCC-v2 prompt/marker provenance mismatch")
    if substrate.get("prompts") != list(FROZEN_PROMPTS):
        raise ValueError("ASCC-v2 prompts differ from the frozen source contract")
    if substrate.get("markers") != list(FROZEN_MARKERS):
        raise ValueError("ASCC-v2 markers differ from the frozen source contract")
    runner_path = Path(__file__).with_name("run_huatuo_ascc_factorial_v2.py")
    if sha256_file(runner_path) != score.get("runner_sha256"):
        raise ValueError("ASCC-v2 runner source changed after scoring")
    current_model_tree = tree_fingerprint(
        Path(score["model_dir"]), {".bin", ".safetensors", ".json", ".model"}
    )
    if current_model_tree.get("fingerprint") != score.get("model_tree", {}).get("fingerprint"):
        raise ValueError("ASCC-v2 model tree changed after scoring")
    current_source_tree = tree_fingerprint(Path(score["huatuo_root"]), {".py"})
    if current_source_tree.get("fingerprint") != score.get("huatuo_source_tree", {}).get("fingerprint"):
        raise ValueError("ASCC-v2 Huatuo source tree changed after scoring")
    all_rows = load_jsonl(manifest_path)
    if len(all_rows) != int(substrate["registered_rows"]):
        raise ValueError("ASCC-v2 substrate registered row mismatch")
    if len(all_rows) != int(score["registered_rows"]):
        raise ValueError("ASCC-v2 score registered row mismatch")
    if len({str(row["item_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate ASCC-v2 item identity")
    prompts = list(substrate["prompts"])
    if int(substrate["registered_jobs_per_model"]) != len(all_rows) * len(prompts):
        raise ValueError("ASCC-v2 substrate registered job mismatch")
    if int(score["registered_jobs"]) != len(all_rows) * len(prompts):
        raise ValueError("ASCC-v2 score registered job mismatch")
    expected_primary_jobs = sum(
        str(row["edge_id"]) == PRIMARY_EDGE for row in all_rows
    ) * len(prompts)
    if int(score["primary_jobs"]) != expected_primary_jobs:
        raise ValueError("ASCC-v2 score primary job mismatch")
    rows = [row for row in all_rows if row["edge_id"] == edge_id]
    if not rows:
        raise ValueError(f"edge absent from ASCC-v2 substrate: {edge_id}")
    vote_counts = {
        vote: sum(int(row["child_votes"]) == vote for row in rows)
        for vote in range(4)
    }
    if min(vote_counts.values()) < 10:
        raise ValueError(f"ASCC-v2 edge has an inadequate vote bin: {vote_counts}")
    for votes in LOCAL_BOUNDARIES.values():
        fixed_overlap_weights(rows, *votes)
    fingerprint = str(score["fingerprint"])
    marker_contract = json.loads(marker_contract_path.read_text())
    if marker_contract.get("fingerprint") != fingerprint:
        raise ValueError("ASCC-v2 marker-token contract fingerprint mismatch")
    if marker_contract.get("assistant_suffix") != repr(" \n"):
        raise ValueError("ASCC-v2 assistant suffix mismatch")
    prefixes = {str(row["fixed_prefix"]) for row in all_rows}
    marker_ids_by_prefix = marker_contract.get("marker_ids_by_prefix")
    if not isinstance(marker_ids_by_prefix, dict) or set(marker_ids_by_prefix) != prefixes:
        raise ValueError("ASCC-v2 marker contract prefix coverage mismatch")
    expected_markers = tuple(str(value) for value in substrate["markers"])
    for prefix, mapping in marker_ids_by_prefix.items():
        if not isinstance(mapping, dict) or set(mapping) != set(expected_markers):
            raise ValueError(f"ASCC-v2 marker contract incomplete: {prefix}")
        values = list(mapping.values())
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValueError(f"ASCC-v2 marker IDs are not integers: {prefix}")
        if len(set(values)) != len(expected_markers):
            raise ValueError(f"ASCC-v2 marker IDs are not distinct: {prefix}")
    expected_layers = list(score.get("adapter_fingerprint", {}).get("layer_ids", []))
    if not expected_layers:
        raise ValueError("ASCC-v2 adapter layer identity missing")
    image_hashes: dict[str, str] = {}
    shards: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    layer_keys: list[str] | None = None
    for row in rows:
        image_id = str(row["image_id"])
        image_path = image_root / f"{image_id}.dicom"
        image_hashes.setdefault(image_id, sha256_file(image_path))
        for prompt in prompts:
            path = score_dir / "shards" / f"{record_key(row['item_id'], prompt['name'])}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            shard = json.loads(path.read_text())
            if shard.get("version") != SCORE_VERSION:
                raise ValueError(f"ASCC-v2 shard version mismatch: {path}")
            expected = {
                "fingerprint": fingerprint,
                "substrate_fingerprint": substrate["fingerprint"],
                "item_id": row["item_id"],
                "image_id": row["image_id"],
                "edge_id": row["edge_id"],
                "parent_votes": row["parent_votes"],
                "child_votes": row["child_votes"],
                "prompt_name": prompt["name"],
                "speech_act": prompt["speech_act"],
                "clinical_noun": prompt["clinical_noun"],
                "prompt": prompt["prompt"],
                "fixed_prefix": row["fixed_prefix"],
                "image_sha256": image_hashes[image_id],
            }
            for key, value in expected.items():
                if shard.get(key) != value:
                    raise ValueError(f"ASCC-v2 shard {key} mismatch: {path}")
            expected_marker_ids = marker_ids_by_prefix[row["fixed_prefix"]]
            if shard.get("marker_token_ids") != expected_marker_ids:
                raise ValueError(f"ASCC-v2 marker IDs mismatch: {path}")
            current_layers = list(shard.get("layer_scores", {}))
            if current_layers != expected_layers:
                raise ValueError(f"ASCC-v2 shard layer order mismatch: {path}")
            if layer_keys is None:
                layer_keys = current_layers
            elif current_layers != layer_keys:
                raise ValueError(f"ASCC-v2 layer identity mismatch: {path}")
            for layer_index, layer_id in enumerate(expected_layers):
                layer_score = shard["layer_scores"][layer_id]
                expected_source = (
                    "ordinary_causallm_forward"
                    if layer_index == len(expected_layers) - 1
                    else "quartile_hidden_native_norm_full_lm_head"
                )
                if layer_score.get("source") != expected_source:
                    raise ValueError(f"ASCC-v2 layer source mismatch: {path}:{layer_id}")
                logits = layer_score.get("logits")
                if not isinstance(logits, dict) or set(logits) != set(expected_markers):
                    raise ValueError(f"ASCC-v2 marker logit schema mismatch: {path}:{layer_id}")
                numeric = [
                    *logits.values(),
                    layer_score.get("restricted_log_probability_mass"),
                    layer_score.get("full_logit_mean"),
                    layer_score.get("full_logit_std"),
                ]
                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in numeric
                ):
                    raise ValueError(f"ASCC-v2 non-finite score: {path}:{layer_id}")
                if float(layer_score["restricted_log_probability_mass"]) > 1e-6:
                    raise ValueError(f"ASCC-v2 invalid restricted probability mass: {path}:{layer_id}")
                if float(layer_score["full_logit_std"]) <= 0:
                    raise ValueError(f"ASCC-v2 invalid full-logit scale: {path}:{layer_id}")
                if not isinstance(layer_score.get("restricted_top1"), bool):
                    raise ValueError(f"ASCC-v2 restricted top1 is not boolean: {path}:{layer_id}")
                negative = float(logits[" absent"])
                uncertain = float(logits[" uncertain"])
                positive = float(logits[" present"])
                recomputed = {
                    "commitment": float(np.logaddexp(positive, negative) - uncertain),
                    "uncertainty_preference": float(uncertain - np.logaddexp(positive, negative)),
                    "polarity": positive - negative,
                    "positive_overcommitment": positive - uncertain,
                    "negative_overcommitment": negative - uncertain,
                }
                coordinates = layer_score.get("coordinates")
                if not isinstance(coordinates, dict) or set(coordinates) != set(recomputed):
                    raise ValueError(f"ASCC-v2 coordinate schema mismatch: {path}:{layer_id}")
                if any(
                    not isinstance(coordinates[key], (int, float))
                    or isinstance(coordinates[key], bool)
                    or not math.isfinite(float(coordinates[key]))
                    or not np.isclose(float(coordinates[key]), value, atol=1e-6, rtol=1e-6)
                    for key, value in recomputed.items()
                ):
                    raise ValueError(f"ASCC-v2 coordinate recomputation mismatch: {path}:{layer_id}")
            shards[(str(row["item_id"]), str(prompt["name"]))] = shard
    if missing:
        raise FileNotFoundError(
            f"ASCC-v2 analysis requires complete registered edge; missing={len(missing)}, first={missing[0]}"
        )
    return substrate, score, rows, shards


def fixed_overlap_weights(rows: Sequence[Mapping[str, Any]], left_vote: int, right_vote: int) -> dict[tuple[int, str], float]:
    counts: dict[tuple[int, str, int], int] = defaultdict(int)
    for row in rows:
        counts[(int(row["parent_votes"]), str(row["aspect_bucket"]), int(row["child_votes"]))] += 1
    raw = {}
    for parent_votes in (2, 3):
        for aspect in ("portrait", "wide", "square"):
            left = counts[(parent_votes, aspect, left_vote)]
            right = counts[(parent_votes, aspect, right_vote)]
            if left and right:
                raw[(parent_votes, aspect)] = left * right / (left + right)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(f"no overlap for reader boundary {left_vote} vs {right_vote}")
    return {key: value / total for key, value in raw.items()}


def stratified_effect(
    items: Sequence[Mapping[str, Any]], metric: str, left_vote: int, right_vote: int, weights: Mapping[tuple[int, str], float]
) -> float:
    cells: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for item in items:
        row = item["row"]
        cells[(int(row["parent_votes"]), str(row["aspect_bucket"]), int(row["child_votes"]))].append(float(item[metric]))
    return float(
        sum(
            weight
            * (
                np.mean(cells[(parent_votes, aspect, left_vote)])
                - np.mean(cells[(parent_votes, aspect, right_vote)])
            )
            for (parent_votes, aspect), weight in weights.items()
        )
    )


def bootstrap_items(
    items: Sequence[Mapping[str, Any]], seed: int, iterations: int
) -> Iterable[list[Mapping[str, Any]]]:
    groups: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        row = item["row"]
        groups[(int(row["parent_votes"]), str(row["aspect_bucket"]), int(row["child_votes"]))].append(item)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sample = []
        for group in groups.values():
            indices = rng.integers(0, len(group), size=len(group))
            sample.extend(group[int(index)] for index in indices)
        yield sample


def assign_stratified_folds(items: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        row = item["row"]
        groups[
            (
                int(row["child_votes"]),
                int(row["parent_votes"]),
                str(row["aspect_bucket"]),
            )
        ].append(item)
    for key, group in groups.items():
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"ascc-v2-fold:{key}:{item['row']['image_id']}".encode()
            ).hexdigest(),
        )
        for index, item in enumerate(ordered):
            item["fold"] = index % 5


def gauge_contrasts(raw_logits: Sequence[float]) -> np.ndarray:
    negative, uncertain, positive = np.asarray(raw_logits, dtype=np.float64)
    return np.asarray([negative - uncertain, positive - uncertain])


def apply_crossfold_affine(
    source_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a gauge-invariant, endpoint-bias-aware affine nuisance out of fold."""

    items = []
    for source_item in source_items:
        item = dict(source_item)
        item.pop("affine_residual_noun_delta", None)
        item.pop("affine_residual_noun_delta_describe", None)
        item.pop("affine_residual_noun_delta_list", None)
        item["affine_residual_by_speech"] = {}
        items.append(item)
    if any("fold" not in item for item in items):
        assign_stratified_folds(items)
    diagnostics: dict[str, Any] = {
        "fits": {},
        "valid": True,
        "quality_valid": True,
    }
    heldout_actual: dict[str, list[float]] = defaultdict(list)
    heldout_predicted: dict[str, list[float]] = defaultdict(list)
    heldout_commitment_residual: dict[tuple[str, int], list[float]] = defaultdict(list)
    for speech_act in ("describe", "list"):
        for fold in range(5):
            train = [
                item
                for item in items
                if int(item["fold"]) != fold
                and int(item["row"]["child_votes"]) in {0, 3}
            ]
            design_rows, targets = [], []
            for item in train:
                x = gauge_contrasts(item["raw_logits"][(speech_act, "findings")])
                y = gauge_contrasts(item["raw_logits"][(speech_act, "abnormalities")])
                design_rows.extend(((x[0], 1.0, 0.0), (x[1], 0.0, 1.0)))
                targets.extend((y[0], y[1]))
            design = np.asarray(design_rows, dtype=np.float64)
            target = np.asarray(targets, dtype=np.float64)
            if design.shape[0] < 30 or np.linalg.matrix_rank(design) != 3:
                diagnostics["valid"] = False
                coefficients = np.asarray([np.nan, np.nan, np.nan])
            else:
                coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            slope, negative_bias, positive_bias = coefficients.tolist()
            stable = bool(
                np.isfinite(coefficients).all()
                and AFFINE_SLOPE_RANGE[0] <= slope <= AFFINE_SLOPE_RANGE[1]
            )
            diagnostics["valid"] = bool(diagnostics["valid"] and stable)
            diagnostics["fits"][f"{speech_act}:fold{fold}"] = {
                "train_clear_items": len(train),
                "slope": slope,
                "negative_endpoint_bias": negative_bias,
                "positive_endpoint_bias": positive_bias,
                "stable": stable,
            }
            for item in items:
                if int(item["fold"]) != fold:
                    continue
                x = gauge_contrasts(item["raw_logits"][(speech_act, "findings")])
                y = gauge_contrasts(item["raw_logits"][(speech_act, "abnormalities")])
                predicted = slope * x + np.asarray([negative_bias, positive_bias])
                if int(item["row"]["child_votes"]) in {0, 3}:
                    heldout_actual[speech_act].extend(y.tolist())
                    heldout_predicted[speech_act].extend(predicted.tolist())
                if stable:
                    adjusted = (y - np.asarray([negative_bias, positive_bias])) / slope
                    adjusted_commitment = float(np.logaddexp(adjusted[0], adjusted[1]))
                    residual = adjusted_commitment - float(
                        item["commitment"][(speech_act, "findings")]
                    )
                else:
                    residual = float("nan")
                if int(item["row"]["child_votes"]) in {0, 3}:
                    heldout_commitment_residual[
                        (speech_act, int(item["row"]["child_votes"]))
                    ].append(residual)
                item["affine_residual_by_speech"][speech_act] = residual
    diagnostics["heldout_clear_r2"] = {}
    for speech_act in ("describe", "list"):
        actual = np.asarray(heldout_actual[speech_act], dtype=np.float64)
        predicted = np.asarray(heldout_predicted[speech_act], dtype=np.float64)
        denominator = float(((actual - actual.mean()) ** 2).sum())
        r2 = (
            float(1.0 - ((actual - predicted) ** 2).sum() / denominator)
            if denominator > 0 and np.isfinite(predicted).all()
            else float("nan")
        )
        diagnostics["heldout_clear_r2"][speech_act] = r2
        diagnostics["quality_valid"] = bool(
            diagnostics["quality_valid"]
            and np.isfinite(r2)
            and r2 >= MINIMUM_AFFINE_R2
        )
        for clear_vote in (0, 3):
            residuals = np.asarray(
                heldout_commitment_residual[(speech_act, clear_vote)],
                dtype=np.float64,
            )
            bias = float(np.mean(residuals)) if residuals.size else float("nan")
            rmse = (
                float(np.sqrt(np.mean(np.square(residuals))))
                if residuals.size
                else float("nan")
            )
            cell = f"{speech_act}:{clear_vote}of3"
            diagnostics.setdefault("heldout_clear_commitment_bias", {})[
                cell
            ] = bias
            diagnostics.setdefault("heldout_clear_commitment_rmse", {})[
                cell
            ] = rmse
            diagnostics["quality_valid"] = bool(
                diagnostics["quality_valid"]
                and np.isfinite(bias)
                and abs(bias) <= MAX_CLEAR_COMMITMENT_ABS_BIAS
                and np.isfinite(rmse)
                and rmse <= MAX_CLEAR_COMMITMENT_RMSE
            )
    diagnostics["quality_valid"] = bool(
        diagnostics["quality_valid"] and diagnostics["valid"]
    )
    for item in items:
        item["affine_residual_noun_delta"] = float(
            np.mean(list(item["affine_residual_by_speech"].values()))
        )
        item["affine_residual_noun_delta_describe"] = float(
            item["affine_residual_by_speech"]["describe"]
        )
        item["affine_residual_noun_delta_list"] = float(
            item["affine_residual_by_speech"]["list"]
        )
    return items, diagnostics


def stratified_level(
    items: Sequence[Mapping[str, Any]],
    metric: str,
    vote: int,
    weights: Mapping[tuple[int, str], float],
) -> float:
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for item in items:
        row = item["row"]
        if int(row["child_votes"]) == vote:
            cells[(int(row["parent_votes"]), str(row["aspect_bucket"]))].append(
                float(item[metric])
            )
    return float(
        sum(weight * np.mean(cells[key]) for key, weight in weights.items())
    )


def metric_summary(
    estimate: float, draws: Sequence[float]
) -> dict[str, Any]:
    finite = [float(value) for value in draws if math.isfinite(float(value))]
    if not math.isfinite(float(estimate)) or not finite:
        return {"estimate": None, "ci95": None, "ci90": None, "valid_draws": len(finite)}
    return {
        "estimate": float(estimate),
        "ci95": interval(finite, 0.95),
        "ci90": interval(finite, 0.90),
        "valid_draws": len(finite),
    }


def conditional_marker_probabilities(raw_logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(raw_logits, dtype=np.float64)
    return np.exp(values - np.logaddexp.reduce(values))


def panel_proxy_losses(probabilities: Sequence[float], child_votes: int) -> tuple[float, float]:
    # This is an explicitly constructed panel-state proxy, not patient truth:
    # unanimous negative -> absent, split panel -> undetermined, unanimous positive -> present.
    target_index = 0 if child_votes == 0 else 2 if child_votes == 3 else 1
    target = np.zeros(3, dtype=np.float64)
    target[target_index] = 1.0
    probabilities = np.asarray(probabilities, dtype=np.float64)
    brier = float(np.sum(np.square(probabilities - target)))
    nll = float(-np.log(max(float(probabilities[target_index]), np.finfo(float).tiny)))
    return brier, nll


def analyze_layer(
    rows: Sequence[Mapping[str, Any]],
    shards: Mapping[tuple[str, str], Mapping[str, Any]],
    prompts: Sequence[Mapping[str, str]],
    layer_id: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    prompt_name = {
        (prompt["speech_act"], prompt["clinical_noun"]): prompt["name"]
        for prompt in prompts
    }
    items: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row["item_id"])
        commitment, polarity, raw_logits, probabilities = {}, {}, {}, {}
        masses, top1 = {}, {}
        for cell, name in prompt_name.items():
            score = shards[(item_id, name)]["layer_scores"][layer_id]
            coordinates = score["coordinates"]
            commitment[cell] = float(coordinates["commitment"])
            polarity[cell] = float(coordinates["polarity"])
            logits = score["logits"]
            raw_logits[cell] = [
                float(logits[" absent"]),
                float(logits[" uncertain"]),
                float(logits[" present"]),
            ]
            probabilities[cell] = conditional_marker_probabilities(raw_logits[cell])
            masses[cell] = float(score["restricted_log_probability_mass"])
            top1[cell] = bool(score["restricted_top1"])
        noun_delta_by_speech = {
            speech_act: commitment[(speech_act, "abnormalities")]
            - commitment[(speech_act, "findings")]
            for speech_act in ("describe", "list")
        }
        polarity_delta_by_speech = {
            speech_act: polarity[(speech_act, "abnormalities")]
            - polarity[(speech_act, "findings")]
            for speech_act in ("describe", "list")
        }
        speech_act_delta_by_noun = {
            noun: commitment[("list", noun)] - commitment[("describe", noun)]
            for noun in ("findings", "abnormalities")
        }
        uncertain_mass_loss_by_speech = {}
        proxy_brier_worsening_by_speech = {}
        proxy_nll_worsening_by_speech = {}
        definite_transition_by_speech = {}
        vote = int(row["child_votes"])
        for speech_act in ("describe", "list"):
            findings_q = probabilities[(speech_act, "findings")]
            abnormalities_q = probabilities[(speech_act, "abnormalities")]
            findings_brier, findings_nll = panel_proxy_losses(findings_q, vote)
            abnormalities_brier, abnormalities_nll = panel_proxy_losses(
                abnormalities_q, vote
            )
            uncertain_mass_loss_by_speech[speech_act] = float(
                findings_q[1] - abnormalities_q[1]
            )
            proxy_brier_worsening_by_speech[speech_act] = float(
                abnormalities_brier - findings_brier
            )
            proxy_nll_worsening_by_speech[speech_act] = float(
                abnormalities_nll - findings_nll
            )
            definite_transition_by_speech[speech_act] = float(
                int(np.argmax(findings_q)) == 1
                and int(np.argmax(abnormalities_q)) in {0, 2}
            )
        items.append(
            {
                "row": row,
                "commitment": commitment,
                "polarity": polarity,
                "raw_logits": raw_logits,
                "restricted_probabilities": probabilities,
                "noun_delta": float(np.mean(list(noun_delta_by_speech.values()))),
                "noun_delta_by_speech": noun_delta_by_speech,
                "noun_delta_describe": float(noun_delta_by_speech["describe"]),
                "noun_delta_list": float(noun_delta_by_speech["list"]),
                "polarity_noun_delta": float(np.mean(list(polarity_delta_by_speech.values()))),
                "polarity_noun_delta_describe": float(
                    polarity_delta_by_speech["describe"]
                ),
                "polarity_noun_delta_list": float(
                    polarity_delta_by_speech["list"]
                ),
                "speech_act_delta": float(
                    np.mean(list(speech_act_delta_by_noun.values()))
                ),
                "noun_by_speech_interaction": float(
                    noun_delta_by_speech["list"] - noun_delta_by_speech["describe"]
                ),
                "uncertain_mass_loss": float(
                    np.mean(list(uncertain_mass_loss_by_speech.values()))
                ),
                "uncertain_mass_loss_describe": float(
                    uncertain_mass_loss_by_speech["describe"]
                ),
                "uncertain_mass_loss_list": float(
                    uncertain_mass_loss_by_speech["list"]
                ),
                "panel_proxy_brier_worsening": float(
                    np.mean(list(proxy_brier_worsening_by_speech.values()))
                ),
                "panel_proxy_nll_worsening": float(
                    np.mean(list(proxy_nll_worsening_by_speech.values()))
                ),
                "definite_transition": float(
                    np.mean(list(definite_transition_by_speech.values()))
                ),
                "findings_uncertainty": float(
                    -np.mean([commitment[(speech_act, "findings")] for speech_act in ("describe", "list")])
                ),
                "findings_polarity": float(
                    np.mean([polarity[(speech_act, "findings")] for speech_act in ("describe", "list")])
                ),
                "restricted_log_mass_by_cell": masses,
                "restricted_top1_by_cell": top1,
            }
        )
    assign_stratified_folds(items)
    items, affine_diagnostics = apply_crossfold_affine(items)

    weights_by_family = {
        family: fixed_overlap_weights(rows, *votes)
        for family, votes in LOCAL_BOUNDARIES.items()
    }

    def local(
        metric: str,
        family: str,
        sample: Sequence[Mapping[str, Any]],
        *,
        polarity_admission: bool = False,
    ) -> float:
        left, right = LOCAL_BOUNDARIES[family]
        if polarity_admission and family == "positive_boundary":
            left, right = 3, 2
        return stratified_effect(
            sample, metric, left, right, weights_by_family[family]
        )

    restricted_cells = {}
    for speech_act in ("describe", "list"):
        for noun in ("findings", "abnormalities"):
            for vote in range(4):
                group = [
                    item
                    for item in items
                    if int(item["row"]["child_votes"]) == vote
                ]
                key = f"{speech_act}:{noun}:{vote}of3"
                restricted_cells[key] = {
                    "n": len(group),
                    "top1_rate": float(
                        np.mean(
                            [
                                item["restricted_top1_by_cell"][(speech_act, noun)]
                                for item in group
                            ]
                        )
                    ),
                    "mean_probability_mass": float(
                        np.mean(
                            [
                                np.exp(
                                    item["restricted_log_mass_by_cell"][(speech_act, noun)]
                                )
                                for item in group
                            ]
                        )
                    ),
                    "conditional_marker_probabilities": {
                        marker: float(
                            np.mean(
                                [
                                    item["restricted_probabilities"][(speech_act, noun)][marker_index]
                                    for item in group
                                ]
                            )
                        )
                        for marker_index, marker in enumerate(
                            ("absent", "panel_undetermined_proxy", "present")
                        )
                    },
                    "conditional_top1_distribution": {
                        marker: float(
                            np.mean(
                                [
                                    int(
                                        np.argmax(
                                            item["restricted_probabilities"][(speech_act, noun)]
                                        )
                                    )
                                    == marker_index
                                    for item in group
                                ]
                            )
                        )
                        for marker_index, marker in enumerate(
                            ("absent", "panel_undetermined_proxy", "present")
                        )
                    },
                }
    summary: dict[str, Any] = {
        "layer_id": layer_id,
        "n_items": len(items),
        "restricted_marker_cells": restricted_cells,
        "affine_fit": affine_diagnostics,
        "local": {},
        "absolute_ambiguous": {},
    }
    raw_local_metrics = {
        "noun_commitment_interaction": "noun_delta",
        "noun_commitment_interaction_describe": "noun_delta_describe",
        "noun_commitment_interaction_list": "noun_delta_list",
        "polarity_interaction": "polarity_noun_delta",
        "polarity_interaction_describe": "polarity_noun_delta_describe",
        "polarity_interaction_list": "polarity_noun_delta_list",
        "neutral_uncertainty_admission": "findings_uncertainty",
        "neutral_polarity_admission": "findings_polarity",
        "speech_act_reader_interaction": "speech_act_delta",
        "noun_by_speech_reader_interaction": "noun_by_speech_interaction",
    }
    affine_local_metrics = {
        "affine_residual_interaction": "affine_residual_noun_delta",
        "affine_residual_interaction_describe": "affine_residual_noun_delta_describe",
        "affine_residual_interaction_list": "affine_residual_noun_delta_list",
    }
    absolute_metrics = {
        "raw_noun_shift": "noun_delta",
        "raw_noun_shift_describe": "noun_delta_describe",
        "raw_noun_shift_list": "noun_delta_list",
        "absolute_polarity_shift": "polarity_noun_delta",
        "absolute_polarity_shift_describe": "polarity_noun_delta_describe",
        "absolute_polarity_shift_list": "polarity_noun_delta_list",
        "uncertain_mass_loss": "uncertain_mass_loss",
        "panel_proxy_brier_worsening": "panel_proxy_brier_worsening",
        "panel_proxy_nll_worsening": "panel_proxy_nll_worsening",
        "definite_transition_rate": "definite_transition",
    }
    affine_absolute_metrics = {
        "affine_residual_shift": "affine_residual_noun_delta",
        "affine_residual_shift_describe": "affine_residual_noun_delta_describe",
        "affine_residual_shift_list": "affine_residual_noun_delta_list",
    }
    local_points: dict[tuple[str, str], float] = {}
    local_draws: dict[tuple[str, str], list[float]] = defaultdict(list)
    absolute_points: dict[tuple[int, str], float] = {}
    absolute_draws: dict[tuple[int, str], list[float]] = defaultdict(list)
    for family in LOCAL_BOUNDARIES:
        for label, metric in {**raw_local_metrics, **affine_local_metrics}.items():
            polarity_admission = label == "neutral_polarity_admission"
            local_points[(family, label)] = local(
                metric, family, items, polarity_admission=polarity_admission
            )
    vote_family = {1: "negative_boundary", 2: "positive_boundary"}
    for vote, family in vote_family.items():
        for label, metric in {**absolute_metrics, **affine_absolute_metrics}.items():
            absolute_points[(vote, label)] = stratified_level(
                items, metric, vote, weights_by_family[family]
            )

    raw_did_point = float(
        np.mean(
            [
                local_points[(family, "noun_commitment_interaction")]
                for family in LOCAL_BOUNDARIES
            ]
        )
    )
    affine_did_point = float(
        np.mean(
            [
                local_points[(family, "affine_residual_interaction")]
                for family in LOCAL_BOUNDARIES
            ]
        )
    )
    raw_did_draws: list[float] = []
    affine_did_draws: list[float] = []
    affine_clear_bias_draws: dict[str, list[float]] = defaultdict(list)
    valid_affine_bootstraps = 0
    for sample in bootstrap_items(items, seed, iterations):
        for family in LOCAL_BOUNDARIES:
            for label, metric in raw_local_metrics.items():
                local_draws[(family, label)].append(
                    local(
                        metric,
                        family,
                        sample,
                        polarity_admission=label == "neutral_polarity_admission",
                    )
                )
        for vote, family in vote_family.items():
            for label, metric in absolute_metrics.items():
                absolute_draws[(vote, label)].append(
                    stratified_level(sample, metric, vote, weights_by_family[family])
                )
        raw_did_draws.append(
            float(
                np.mean(
                    [
                        local("noun_delta", family, sample)
                        for family in LOCAL_BOUNDARIES
                    ]
                )
            )
        )
        fitted_sample, fitted_diagnostics = apply_crossfold_affine(sample)
        if not fitted_diagnostics["valid"]:
            continue
        valid_affine_bootstraps += 1
        for cell, value in fitted_diagnostics[
            "heldout_clear_commitment_bias"
        ].items():
            affine_clear_bias_draws[cell].append(float(value))
        for family in LOCAL_BOUNDARIES:
            for label, metric in affine_local_metrics.items():
                local_draws[(family, label)].append(
                    local(metric, family, fitted_sample)
                )
        for vote, family in vote_family.items():
            for label, metric in affine_absolute_metrics.items():
                absolute_draws[(vote, label)].append(
                    stratified_level(
                        fitted_sample, metric, vote, weights_by_family[family]
                    )
                )
        affine_did_draws.append(
            float(
                np.mean(
                    [
                        local(
                            "affine_residual_noun_delta", family, fitted_sample
                        )
                        for family in LOCAL_BOUNDARIES
                    ]
                )
            )
        )

    for family in LOCAL_BOUNDARIES:
        summary["local"][family] = {
            label: metric_summary(
                local_points[(family, label)], local_draws[(family, label)]
            )
            for label in {**raw_local_metrics, **affine_local_metrics}
        }
    for vote, family in vote_family.items():
        overlap_keys = set(weights_by_family[family])
        n_overlap = sum(
            int(item["row"]["child_votes"]) == vote
            and (
                int(item["row"]["parent_votes"]),
                str(item["row"]["aspect_bucket"]),
            )
            in overlap_keys
            for item in items
        )
        summary["absolute_ambiguous"][str(vote)] = {
            "n_overlap": n_overlap,
            "overlap_weights": {
                f"parent{key[0]}:{key[1]}": value
                for key, value in weights_by_family[family].items()
            },
            **{
                label: metric_summary(
                    absolute_points[(vote, label)], absolute_draws[(vote, label)]
                )
                for label in {**absolute_metrics, **affine_absolute_metrics}
            },
        }
    summary["noun_did"] = metric_summary(raw_did_point, raw_did_draws)
    summary["affine_residual_did"] = metric_summary(
        affine_did_point, affine_did_draws
    )
    summary["affine_bootstrap"] = {
        "valid": valid_affine_bootstraps,
        "requested": iterations,
        "valid_fraction": valid_affine_bootstraps / iterations,
        "nuisance_refit_within_each_draw": True,
        "heldout_clear_commitment_bias": {
            cell: metric_summary(value, affine_clear_bias_draws[cell])
            for cell, value in affine_diagnostics[
                "heldout_clear_commitment_bias"
            ].items()
        },
    }

    summary["parent_vote_sensitivity"] = {}
    for parent_vote in (2, 3):
        parent_rows = [
            row for row in rows if int(row["parent_votes"]) == parent_vote
        ]
        parent_items = [
            item
            for item in items
            if int(item["row"]["parent_votes"]) == parent_vote
        ]
        family_effects = {}
        for family, votes in LOCAL_BOUNDARIES.items():
            try:
                weights = fixed_overlap_weights(parent_rows, *votes)
                family_effects[family] = stratified_effect(
                    parent_items, "noun_delta", *votes, weights
                )
            except ValueError:
                family_effects[family] = None
        summary["parent_vote_sensitivity"][str(parent_vote)] = family_effects
    return summary


def ci_positive(summary: Mapping[str, Any], level: str = "ci95") -> bool:
    ci = summary.get(level)
    return bool(isinstance(ci, Mapping) and float(ci["low"]) > 0.0)


def ci_equivalent(
    summary: Mapping[str, Any], bound: float = EQUIVALENCE_BOUND
) -> bool:
    ci = summary.get("ci90")
    return bool(
        isinstance(ci, Mapping)
        and float(ci["low"]) >= -bound
        and float(ci["high"]) <= bound
    )


def meaningful_positive(summary: Mapping[str, Any]) -> bool:
    estimate = summary.get("estimate")
    return bool(
        estimate is not None
        and float(estimate) >= MINIMUM_MEANINGFUL_DID
        and ci_positive(summary)
    )


def build_computational_screen(final: Mapping[str, Any]) -> dict[str, bool]:
    local = final["local"]
    ambiguous = final["absolute_ambiguous"]
    screen = {
        "neutral_third_state_admission_each_local_ci_positive": all(
            ci_positive(local[family]["neutral_uncertainty_admission"])
            for family in LOCAL_BOUNDARIES
        ),
        "neutral_polarity_admission_each_local_ci_positive": all(
            ci_positive(local[family]["neutral_polarity_admission"])
            for family in LOCAL_BOUNDARIES
        ),
        "noun_interaction_each_local_ci_positive": all(
            ci_positive(local[family]["noun_commitment_interaction"])
            for family in LOCAL_BOUNDARIES
        ),
        "noun_interaction_each_speech_act_and_local_ci_positive": all(
            ci_positive(
                local[family][f"noun_commitment_interaction_{speech_act}"]
            )
            for family in LOCAL_BOUNDARIES
            for speech_act in ("describe", "list")
        ),
        "ambiguous_absolute_noun_shift_each_speech_ci_positive": all(
            ci_positive(ambiguous[str(vote)][f"raw_noun_shift_{speech_act}"])
            for vote in (1, 2)
            for speech_act in ("describe", "list")
        ),
        "polarity_each_local_and_speech_equivalent": all(
            ci_equivalent(local[family][f"polarity_interaction_{speech_act}"])
            for family in LOCAL_BOUNDARIES
            for speech_act in ("describe", "list")
        ),
        "absolute_ambiguous_polarity_each_speech_equivalent": all(
            ci_equivalent(
                ambiguous[str(vote)][f"absolute_polarity_shift_{speech_act}"]
            )
            for vote in (1, 2)
            for speech_act in ("describe", "list")
        ),
        "noun_by_speech_reader_interaction_equivalent": all(
            ci_equivalent(local[family]["noun_by_speech_reader_interaction"])
            for family in LOCAL_BOUNDARIES
        ),
        "uncertain_marker_mass_lost_in_each_ambiguous_bin": all(
            ci_positive(ambiguous[str(vote)]["uncertain_mass_loss"])
            for vote in (1, 2)
        ),
        "panel_state_proxy_brier_worsens_in_each_ambiguous_bin": all(
            ci_positive(ambiguous[str(vote)]["panel_proxy_brier_worsening"])
            for vote in (1, 2)
        ),
        "affine_fit_valid_on_original_sample": bool(final["affine_fit"]["valid"]),
        "affine_fit_quality_valid_on_original_sample": bool(
            final["affine_fit"]["quality_valid"]
        ),
        "affine_clear_bin_bias_bootstrap_equivalent": all(
            ci_equivalent(cell)
            for cell in final["affine_bootstrap"][
                "heldout_clear_commitment_bias"
            ].values()
        ),
        "affine_nested_bootstrap_valid_fraction": float(
            final["affine_bootstrap"]["valid_fraction"]
        )
        >= MINIMUM_VALID_BOOTSTRAP_FRACTION,
        "affine_residual_each_local_and_speech_ci_positive": all(
            ci_positive(
                local[family][f"affine_residual_interaction_{speech_act}"]
            )
            for family in LOCAL_BOUNDARIES
            for speech_act in ("describe", "list")
        ),
        "affine_residual_absolute_shift_each_speech_ci_positive": all(
            ci_positive(
                ambiguous[str(vote)][f"affine_residual_shift_{speech_act}"]
            )
            for vote in (1, 2)
            for speech_act in ("describe", "list")
        ),
        "minimum_meaningful_raw_noun_did": meaningful_positive(final["noun_did"]),
        "minimum_meaningful_affine_residual_did": meaningful_positive(
            final["affine_residual_did"]
        ),
        "restricted_marker_top1_each_prompt_vote_cell": all(
            float(cell["top1_rate"]) >= MINIMUM_MARKER_TOP1_RATE
            for cell in final["restricted_marker_cells"].values()
        ),
        "parent_vote_direction_consistent": all(
            estimate is not None and float(estimate) > 0
            for family_effects in final["parent_vote_sensitivity"].values()
            for estimate in family_effects.values()
        ),
    }
    screen["computational_screen_passed"] = all(screen.values())
    return screen


def write_once_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    data = (
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def acquire_formal_identity_lock(
    score_dir: Path,
    output: Path,
    edge_id: str,
    analysis_role: str,
    score_fingerprint: str,
) -> tuple[Path, dict[str, Any]]:
    canonical_name = (
        "primary_analysis_v2_1_blind_locked.json"
        if analysis_role == "primary"
        else f"replication_analysis_v2_1_{edge_id}_blind_locked.json"
    )
    canonical_output = score_dir.resolve() / canonical_name
    if output.resolve() != canonical_output:
        raise ValueError(f"formal ASCC-v2.1 output must be {canonical_output}")
    analyzer_sha = sha256_file(Path(__file__).resolve())
    payload = {
        "version": VERSION,
        "status": "formal_identity_locked_before_outcome_computation",
        "score_fingerprint": score_fingerprint,
        "edge_id": edge_id,
        "analysis_role": analysis_role,
        "seed": FROZEN_SEED,
        "bootstrap_iterations": FROZEN_BOOTSTRAP_ITERATIONS,
        "canonical_output": canonical_name,
        "analyzer_sha256": analyzer_sha,
    }
    payload["fingerprint"] = canonical_hash(payload)
    lock_path = score_dir.resolve() / (
        f"FORMAL_ANALYSIS_V2_1_{analysis_role}_{edge_id}_LOCK.json"
    )
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        if existing != payload:
            raise ValueError("formal ASCC-v2.1 identity is already locked differently")
    else:
        write_once_json(lock_path, payload)
    return lock_path, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--edge", default=PRIMARY_EDGE)
    parser.add_argument(
        "--analysis-role", choices=("primary", "replication"), default="primary"
    )
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=FROZEN_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed != FROZEN_SEED:
        raise ValueError("formal ASCC-v2.1 seed is frozen")
    if args.bootstrap_iterations != FROZEN_BOOTSTRAP_ITERATIONS:
        raise ValueError("formal ASCC-v2.1 bootstrap count is frozen")
    if args.analysis_role == "primary" and args.edge != PRIMARY_EDGE:
        raise ValueError("primary ASCC-v2 analysis is restricted to the frozen primary edge")
    if args.analysis_role == "replication" and args.edge == PRIMARY_EDGE:
        raise ValueError("replication role requires a non-primary edge")
    preliminary_score_path = args.score_dir / "score_config.json"
    if not preliminary_score_path.is_file():
        raise FileNotFoundError(preliminary_score_path)
    preliminary_score = json.loads(preliminary_score_path.read_text())
    lock_path, lock = acquire_formal_identity_lock(
        args.score_dir,
        args.output,
        args.edge,
        args.analysis_role,
        str(preliminary_score.get("fingerprint")),
    )
    substrate, score, rows, shards = validate_inputs(
        args.substrate_dir, args.score_dir, args.image_root, args.edge
    )
    prompts = list(substrate["prompts"])
    layer_ids = list(score["adapter_fingerprint"]["layer_ids"])
    layers = {
        layer_id: analyze_layer(
            rows, shards, prompts, layer_id, args.seed + index * 10000, args.bootstrap_iterations
        )
        for index, layer_id in enumerate(layer_ids)
    }
    final = layers[layer_ids[-1]]
    screen = build_computational_screen(final)
    analyzer_path = Path(__file__).resolve()
    analysis_contract = {
        "version": VERSION,
        "edge_id": args.edge,
        "analysis_role": args.analysis_role,
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "local_boundaries": LOCAL_BOUNDARIES,
        "equivalence_bound": EQUIVALENCE_BOUND,
        "minimum_meaningful_did": MINIMUM_MEANINGFUL_DID,
        "minimum_marker_top1_rate": MINIMUM_MARKER_TOP1_RATE,
        "minimum_affine_r2": MINIMUM_AFFINE_R2,
        "minimum_valid_bootstrap_fraction": MINIMUM_VALID_BOOTSTRAP_FRACTION,
        "affine_slope_range": AFFINE_SLOPE_RANGE,
        "maximum_clear_commitment_abs_bias": MAX_CLEAR_COMMITMENT_ABS_BIAS,
        "maximum_clear_commitment_rmse": MAX_CLEAR_COMMITMENT_RMSE,
        "analyzer_sha256": sha256_file(analyzer_path),
        "formal_identity_lock": lock["fingerprint"],
    }
    result = {
        "version": VERSION,
        "status": f"{args.analysis_role}_restricted_choice_computational_screen_only",
        "terminology": "panel-undetermined proxy; not patient truth or latent uncertainty",
        "analysis_contract": analysis_contract,
        "analysis_contract_fingerprint": canonical_hash(analysis_contract),
        "formal_identity_lock_path": str(lock_path),
        "formal_identity_lock_fingerprint": lock["fingerprint"],
        "substrate_fingerprint": substrate["fingerprint"],
        "score_fingerprint": score["fingerprint"],
        "edge_id": args.edge,
        "registered_rows": len(rows),
        "registered_jobs": len(rows) * len(prompts),
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "equivalence_bound": EQUIVALENCE_BOUND,
        "minimum_meaningful_did": MINIMUM_MEANINGFUL_DID,
        "layers": layers,
        "final_layer_id": layer_ids[-1],
        "computational_screen": screen,
        "promotion_authorized": False,
        "promotion_prohibited_without": [
            "three-or-more radiologist marker-semantic admission",
            "independent radiologist panel-state reference admission",
            "no-parent-prefix sensitivity",
            "second model",
            "replication edge",
            "text-only and image-swap controls",
            "generic VUF separation",
            "natural OE intent-to-treat external validity",
            "physician construct review",
        ],
    }
    write_once_json(args.output, result)
    print(json.dumps(screen, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
