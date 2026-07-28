"""Normal-bundle proximal geometry for ANCHOR-NBP.

The public API in this file is deliberately small: build a weighted source
bank, choose a local affine patch by cosine kNN, and return one additive delta
for all visual tokens.  The method never uses target labels or answer logits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

VERSION = "anchor-nbp-geometry-v1"
METHODS = (
    "greedy",
    "nbp",
    "local_isotropic",
    "nn_interpolation",
    "tangent_matched",
    "random_matched",
    "global_pca",
)


def stable_json_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), eps)


def normalize_matrix(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), eps, None)


@dataclass(frozen=True)
class SourceBankRecord:
    record_id: str
    domain: str
    task: str
    modality: str
    view: str
    question_family: str
    image_path: str
    z: list[float]
    q: list[float]
    success_score: float
    reliability_weight: float
    answer: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class NBPConfig:
    k: int
    rank: int
    alpha: float
    conditioning: str = "task_modality_view_family"
    normalization: str = "l2_knn_raw_pca"
    seed: int = 20260727
    random_seed: int = 0


@dataclass(frozen=True)
class LocalPatch:
    neighbor_indices: list[int]
    neighbor_ids: list[str]
    weights: list[float]
    mu: list[float]
    basis: list[list[float]]
    e_perp: float
    density: float
    condition_level: str


@dataclass(frozen=True)
class GeometryDelta:
    method: str
    delta: list[float]
    delta_norm: float
    e_perp: float
    tangent_energy: float
    patch: LocalPatch | None


def _condition_key(row: dict[str, Any], level: int) -> tuple[str, ...]:
    task = str(row.get("task", "unknown"))
    modality = str(row.get("modality", "xray"))
    view = str(row.get("view", "unknown"))
    family = str(row.get("question_family", "unknown"))
    if task == "ce":
        if level == 0:
            return (task, modality, view, family)
        if level == 1:
            return (task, modality, family)
    if level == 0:
        return (task, modality, view)
    if level == 1:
        return (task, modality)
    return (task, modality)


def condition_levels(row: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    if str(row.get("task", "unknown")) == "ce":
        return [
            ("task+modality+view+question_family", _condition_key(row, 0)),
            ("task+modality+question_family", _condition_key(row, 1)),
            ("task+modality", _condition_key(row, 2)),
        ]
    return [
        ("task+modality+view", _condition_key(row, 0)),
        ("task+modality", _condition_key(row, 1)),
    ]


class SourceBank:
    def __init__(self, records: Iterable[SourceBankRecord]):
        self.records = list(records)
        if not self.records:
            raise ValueError("source bank is empty")
        self.z = np.asarray([record.z for record in self.records], dtype=np.float64)
        self.q = normalize_matrix(np.asarray([record.q for record in self.records], dtype=np.float64))
        self.weights = np.asarray([record.reliability_weight for record in self.records], dtype=np.float64)
        self.ids = [record.record_id for record in self.records]
        self.meta_rows = [asdict(record) for record in self.records]
        self.dimension = int(self.z.shape[1])
        self._global_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def to_json(self, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "version": VERSION,
            "dimension": self.dimension,
            "n": len(self.records),
            "config": config or {},
            "records": self.meta_rows,
        }
        payload["fingerprint"] = stable_json_sha256(
            {
                "version": VERSION,
                "dimension": self.dimension,
                "records": [
                    {
                        "record_id": r.record_id,
                        "domain": r.domain,
                        "task": r.task,
                        "success_score": r.success_score,
                        "reliability_weight": r.reliability_weight,
                        "z_sha": hashlib.sha256(np.asarray(r.z, dtype=np.float32).tobytes()).hexdigest(),
                    }
                    for r in self.records
                ],
                "config": config or {},
            }
        )
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SourceBank":
        if payload.get("version") != VERSION:
            raise RuntimeError("NBP source bank version mismatch")
        return cls(SourceBankRecord(**row) for row in payload["records"])


def weighted_local_patch(
    bank: SourceBank,
    z0: np.ndarray,
    target_meta: dict[str, Any],
    *,
    k: int,
    rank: int,
    weights_mode: Literal["reliability", "unweighted", "shuffled", "low_reliability"] = "reliability",
    seed: int = 0,
) -> tuple[LocalPatch, np.ndarray, np.ndarray]:
    if k <= 1:
        raise ValueError("k must be > 1")
    if rank <= 0 or rank >= k:
        raise ValueError("rank must satisfy 0 < rank < k")
    q0 = l2_normalize(z0)
    selected = None
    level_name = None
    for name, key in condition_levels(target_meta):
        idx = [
            i
            for i, row in enumerate(bank.meta_rows)
            if _condition_key(row, 0 if "view" in name and "question" in name else 1 if "question" in name else 2)
            == key
        ]
        # The helper above is awkward for CE fallbacks, so use direct row keys.
        idx = []
        for i, row in enumerate(bank.meta_rows):
            for row_name, row_key in condition_levels(row):
                if row_name == name and row_key == key:
                    idx.append(i)
                    break
        if len(idx) >= k:
            selected = np.asarray(idx, dtype=np.int64)
            level_name = name
            break
    if selected is None:
        # Last-resort deterministic fallback within task+modality.
        key = (str(target_meta.get("task", "unknown")), str(target_meta.get("modality", "xray")))
        selected = np.asarray(
            [
                i
                for i, row in enumerate(bank.meta_rows)
                if (str(row.get("task", "unknown")), str(row.get("modality", "xray"))) == key
            ],
            dtype=np.int64,
        )
        level_name = "task+modality-forced"
    if selected.size < k:
        raise ValueError(f"not enough source records for condition: have {selected.size}, need {k}")

    sims = bank.q[selected] @ q0
    order = np.argsort(-sims, kind="mergesort")[:k]
    nn = selected[order]
    z_nn = bank.z[nn]
    if weights_mode == "reliability":
        w = bank.weights[nn].copy()
    elif weights_mode == "unweighted":
        w = np.ones(k, dtype=np.float64)
    elif weights_mode == "low_reliability":
        w = 1.1 - bank.weights[nn]
    elif weights_mode == "shuffled":
        rng = np.random.default_rng(seed)
        w = bank.weights[nn].copy()
        rng.shuffle(w)
    else:
        raise ValueError(f"unknown weights mode: {weights_mode}")
    w = np.clip(w, 1e-6, None)
    w = w / w.sum()
    mu = (w[:, None] * z_nn).sum(axis=0)
    centered = z_nn - mu
    # Weighted thin SVD of X^T W X without materializing a dxd covariance.
    xw = centered * np.sqrt(w[:, None])
    _, _, vt = np.linalg.svd(xw, full_matrices=False)
    use_rank = min(rank, vt.shape[0], k - 1)
    basis = vt[:use_rank]
    residual = z0 - mu
    tangent = basis.T @ (basis @ residual)
    normal = residual - tangent
    density = float(np.mean(sims[order]))
    patch = LocalPatch(
        neighbor_indices=[int(i) for i in nn],
        neighbor_ids=[bank.ids[int(i)] for i in nn],
        weights=[float(x) for x in w],
        mu=mu.astype(float).tolist(),
        basis=basis.astype(float).tolist(),
        e_perp=float(normal @ normal),
        density=density,
        condition_level=str(level_name),
    )
    return patch, basis, mu


def compute_delta(
    bank: SourceBank,
    z0: np.ndarray,
    target_meta: dict[str, Any],
    *,
    method: str,
    config: NBPConfig,
    weights_mode: str = "reliability",
) -> GeometryDelta:
    if method == "greedy":
        zero = np.zeros_like(z0, dtype=np.float64)
        return GeometryDelta(method, zero.astype(float).tolist(), 0.0, 0.0, 0.0, None)
    patch, basis, mu = weighted_local_patch(
        bank,
        z0,
        target_meta,
        k=config.k,
        rank=config.rank,
        weights_mode=weights_mode,  # type: ignore[arg-type]
        seed=config.seed + config.random_seed,
    )
    residual = z0.astype(np.float64) - mu
    tangent = basis.T @ (basis @ residual)
    normal = residual - tangent
    rng = np.random.default_rng(config.seed + config.random_seed)
    alpha = float(config.alpha)
    if method == "nbp":
        delta = -alpha * normal
    elif method == "local_isotropic":
        delta = -alpha * residual
    elif method == "nn_interpolation":
        first = np.asarray(bank.records[patch.neighbor_indices[0]].z, dtype=np.float64)
        delta = alpha * (first - z0)
    elif method == "tangent_matched":
        norm = float(np.linalg.norm(normal))
        base = tangent
        base_norm = float(np.linalg.norm(base))
        delta = np.zeros_like(z0) if base_norm < 1e-12 else -alpha * norm * base / base_norm
    elif method == "random_matched":
        norm = float(np.linalg.norm(normal))
        random = rng.normal(size=z0.shape)
        # Remove tangent projection to make this a fair random normal-like direction.
        random = random - basis.T @ (basis @ random)
        rnorm = float(np.linalg.norm(random))
        delta = np.zeros_like(z0) if rnorm < 1e-12 else -alpha * norm * random / rnorm
    elif method == "global_pca":
        cache_rank = int(config.rank)
        if cache_rank not in bank._global_cache:
            global_mu = bank.z.mean(axis=0)
            _, _, vt = np.linalg.svd(bank.z - global_mu, full_matrices=False)
            bank._global_cache[cache_rank] = (global_mu, vt[: min(cache_rank, vt.shape[0])])
        global_mu, global_basis = bank._global_cache[cache_rank]
        gres = z0 - global_mu
        gnormal = gres - global_basis.T @ (global_basis @ gres)
        delta = -alpha * gnormal
    else:
        raise ValueError(f"unknown NBP method: {method}")
    return GeometryDelta(
        method=method,
        delta=delta.astype(float).tolist(),
        delta_norm=float(np.linalg.norm(delta)),
        e_perp=float(normal @ normal),
        tangent_energy=float(tangent @ tangent),
        patch=patch,
    )
