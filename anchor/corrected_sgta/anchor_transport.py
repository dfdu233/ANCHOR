"""Core mathematics and audit utilities for ANCHOR evidence transport."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


VERSION = "anchor-source-native-evidence-transport-v1"
FEATURE_NAMES = (
    "image_nll",
    "visual_log_likelihood_gain",
    "normalized_position",
    "is_eos",
)
DEFAULT_PROJECTIONS = 32
DEFAULT_QUANTILES = 32
DEFAULT_PROJECTION_SEED = 20260726
DEFAULT_NEIGHBORS = 32
MODEL_ARTIFACT_SUFFIXES = {
    ".bin", ".json", ".model", ".pt", ".safetensors", ".txt",
}


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        # Multi-shard checkpoints can exceed the container memory limit when
        # repeated integrity scans leave their pages in the cgroup file cache.
        # The advice changes neither file contents nor the computed digest.
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(
                    handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                )
            except OSError:
                # Cache eviction is an efficiency hint, never an integrity
                # requirement (and is unsupported on some filesystems).
                pass
    return digest.hexdigest()


def model_artifact_fingerprint(model_path: Path) -> dict[str, Any]:
    """Hash local model/tokenizer artifacts, following symlinks to their bytes."""

    root = model_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES
            or path.name in {"tokenizer_config.json", "special_tokens_map.json"}
        )
    )
    if not candidates:
        raise RuntimeError(f"no model artifacts found under {root}")
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in candidates
    ]
    return {
        "root": str(root),
        "artifacts": artifacts,
        "fingerprint": stable_json_sha256(artifacts),
    }


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"empty manifest: {path}")
    payload = None
    if text.lstrip().startswith(("[", "{")):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        records = payload.get("records")
        rows = records if isinstance(records, list) else [payload]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest must contain non-empty object rows: {path}")
    return list(rows)


def _first_present(row: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def normalize_manifest_record(
    row: dict[str, Any],
    *,
    require_answer: bool,
    default_domain: str | None = None,
) -> dict[str, Any]:
    """Normalize aliases without retaining fields that selection must not use."""

    identifier = _first_present(row, ("id", "question_id", "qid"))
    image = _first_present(row, ("image", "image_path"))
    if isinstance(image, list):
        if not image:
            image = None
        else:
            image = image[0]
    prompt = _first_present(row, ("prompt", "question", "text"))
    answer = _first_present(
        row, ("answer", "reference", "report", "ground_truth", "gt_answer")
    )
    conversations = row.get("conversations")
    if isinstance(conversations, list) and len(conversations) >= 2:
        if prompt is None and isinstance(conversations[0], dict):
            prompt = conversations[0].get("value", conversations[0].get("content"))
        if answer is None and isinstance(conversations[1], dict):
            answer = conversations[1].get("value", conversations[1].get("content"))
    domain = _first_present(row, ("domain", "dataset", "source_domain"))
    if domain is None:
        domain = default_domain
    missing = [
        name
        for name, value in (
            ("id", identifier), ("image", image), ("prompt", prompt),
            ("domain", domain),
        )
        if value is None or not str(value).strip()
    ]
    if require_answer and (answer is None or not str(answer).strip()):
        missing.append("answer")
    if missing:
        raise ValueError(f"manifest row missing required fields: {missing}")
    output = {
        "id": str(identifier),
        "image": str(image),
        "prompt": str(prompt).replace("<image>", "").strip(),
        "domain": str(domain),
        "patient_id": str(
            _first_present(row, ("patient_id", "subject_id", "patient"))
            or identifier
        ),
    }
    if require_answer:
        output["answer"] = str(answer).strip()
    return output


def resolve_image_path(value: str, image_root: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute():
        if image_root is None:
            raise ValueError(f"relative image path requires --image-root: {value}")
        path = image_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_trajectory(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"trajectory must have shape [tokens,{len(FEATURE_NAMES)}], "
            f"got {array.shape}"
        )
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError("trajectory must be non-empty and finite")
    if not np.all((array[:, 2] > 0.0) & (array[:, 2] <= 1.0)):
        raise ValueError("normalized positions must lie in (0,1]")
    if not np.all(np.isin(array[:, 3], (0.0, 1.0))):
        raise ValueError("EOS indicator must be binary")
    if int(array[:, 3].sum()) > 1:
        raise ValueError("a trajectory may contain at most one EOS token")
    return array


def robust_feature_statistics(
    trajectories: Iterable[np.ndarray],
) -> dict[str, list[float]]:
    arrays = [validate_trajectory(value) for value in trajectories]
    if not arrays:
        raise ValueError("at least one source trajectory is required")
    pooled = np.concatenate(arrays, axis=0)
    location = np.median(pooled, axis=0)
    mad = np.median(np.abs(pooled - location), axis=0)
    scale = np.where(1.4826 * mad < 1e-6, 1.0, 1.4826 * mad)
    location[2:] = 0.0
    scale[2:] = 1.0
    return {
        "feature_names": list(FEATURE_NAMES),
        "location": location.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "estimator": "median_and_1.4826_mad;position_and_eos_unscaled",
    }


def normalize_trajectory(
    trajectory: np.ndarray, statistics: dict[str, Any]
) -> np.ndarray:
    value = validate_trajectory(trajectory)
    if statistics.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("source feature-statistics schema mismatch")
    location = np.asarray(statistics["location"], dtype=np.float64)
    scale = np.asarray(statistics["scale"], dtype=np.float64)
    if location.shape != (4,) or scale.shape != (4,) or np.any(scale <= 0):
        raise ValueError("invalid source feature statistics")
    normalized = (value - location) / scale
    if not np.isfinite(normalized).all():
        raise FloatingPointError("non-finite normalized trajectory")
    return normalized


def l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in {1, 2} or not np.isfinite(array).all():
        raise ValueError("embedding must be a finite vector or matrix")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero-norm embedding is not retrievable")
    return array / norm


def deterministic_directions(
    dimension: int = 4,
    count: int = DEFAULT_PROJECTIONS,
    seed: int = DEFAULT_PROJECTION_SEED,
) -> np.ndarray:
    if dimension <= 0 or count <= 0:
        raise ValueError("projection dimension and count must be positive")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, dimension))
    return l2_normalize(directions)


def _quantile_resample(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("quantile count must be positive")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError("projected measure must be a non-empty vector")
    source = (np.arange(ordered.size, dtype=np.float64) + 0.5) / ordered.size
    target = (np.arange(count, dtype=np.float64) + 0.5) / count
    return np.interp(target, source, ordered)


def sliced_wasserstein_squared(
    left: np.ndarray,
    right: np.ndarray,
    *,
    directions: np.ndarray | None = None,
    quantiles: int = DEFAULT_QUANTILES,
) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.ndim != 2 or right_array.ndim != 2
        or left_array.shape[1] != right_array.shape[1]
        or left_array.shape[0] == 0 or right_array.shape[0] == 0
    ):
        raise ValueError("measures must be non-empty [points,shared-dimension] arrays")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("measures must be finite")
    projection = (
        deterministic_directions(left_array.shape[1])
        if directions is None else np.asarray(directions, dtype=np.float64)
    )
    if projection.ndim != 2 or projection.shape[1] != left_array.shape[1]:
        raise ValueError("projection directions have the wrong shape")
    projection = l2_normalize(projection)
    left_projected = left_array @ projection.T
    right_projected = right_array @ projection.T
    distances = []
    for index in range(projection.shape[0]):
        left_quantiles = _quantile_resample(left_projected[:, index], quantiles)
        right_quantiles = _quantile_resample(right_projected[:, index], quantiles)
        distances.append(np.square(left_quantiles - right_quantiles).mean())
    value = float(np.mean(distances))
    if value < -1e-12 or not math.isfinite(value):
        raise FloatingPointError("invalid sliced Wasserstein distance")
    return max(0.0, value)


def nearest_source_indices(
    query_embedding: np.ndarray,
    source_embeddings: np.ndarray,
    neighbors: int = DEFAULT_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float64))
    source = l2_normalize(np.asarray(source_embeddings, dtype=np.float64))
    if query.ndim != 1 or source.ndim != 2 or source.shape[1] != query.shape[0]:
        raise ValueError("query/source embedding dimensions differ")
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    count = min(neighbors, source.shape[0])
    similarities = source @ query
    order = np.lexsort((np.arange(source.shape[0]), -similarities))[:count]
    return order.astype(np.int64), similarities[order]


def source_frechet_energy(
    trajectory: np.ndarray,
    source_trajectories: Sequence[np.ndarray],
    *,
    directions: np.ndarray | None = None,
    quantiles: int = DEFAULT_QUANTILES,
) -> tuple[float, list[float]]:
    if not source_trajectories:
        raise ValueError("at least one retrieved source trajectory is required")
    distances = [
        sliced_wasserstein_squared(
            trajectory, source, directions=directions, quantiles=quantiles
        )
        for source in source_trajectories
    ]
    return float(np.mean(distances)), distances


def anchor_sequence_score(
    mean_image_log_probability: float,
    source_distance: float,
    lambda_value: float,
) -> float:
    if lambda_value < 0:
        raise ValueError("lambda must be non-negative")
    values = (mean_image_log_probability, source_distance, lambda_value)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("score components must be finite")
    return float(mean_image_log_probability - lambda_value * source_distance)


def select_candidate(
    candidates: Sequence[dict[str, Any]], lambda_value: float
) -> tuple[int, list[float]]:
    if not candidates:
        raise ValueError("candidate list is empty")
    scores = [
        anchor_sequence_score(
            float(candidate["mean_image_log_probability"]),
            float(candidate["source_distance"]), lambda_value,
        )
        for candidate in candidates
    ]
    return int(np.argmax(np.asarray(scores))), scores


def conformal_quantile(values: Sequence[float], coverage: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("calibration scores must be a non-empty finite vector")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between zero and one")
    rank = min(array.size, int(math.ceil((array.size + 1) * coverage)))
    return float(np.partition(array, rank - 1)[rank - 1])


def worst_source_conformal_threshold(
    scores_by_domain: dict[str, Sequence[float]], coverage: float
) -> tuple[float, dict[str, float]]:
    if not scores_by_domain:
        raise ValueError("at least one source domain is required")
    per_domain = {
        domain: conformal_quantile(scores, coverage)
        for domain, scores in sorted(scores_by_domain.items())
    }
    return max(per_domain.values()), per_domain


def conformal_candidate_indices(
    candidate_scores: Sequence[float], threshold: float
) -> list[int]:
    values = np.asarray(candidate_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("candidate scores must be a non-empty finite vector")
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("conformal threshold must be finite and non-negative")
    nonconformity = float(values.max()) - values
    selected = np.flatnonzero(nonconformity <= threshold + 1e-12).tolist()
    if not selected:
        raise RuntimeError("conformal set unexpectedly empty")
    return [int(index) for index in selected]
