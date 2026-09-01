"""Single-image projection toward a local source-frequency prototype."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from corrected_sgta.mosec import model_visible_image, structure_metrics
from corrected_sgta.qls_tr import (
    PCAIndex,
    kde_neighbors,
    reconstruct_mean_shift_view,
    spectral_descriptor,
)


@dataclass
class LocalSourceIndex:
    geometry: PCAIndex
    records: list[dict[str, Any]]
    metadata: dict[str, Any]
    rgb_mean: np.ndarray | None = None
    rgb_std: np.ndarray | None = None


def load_local_source_index(path: Path) -> LocalSourceIndex:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    arrays = np.load(path, allow_pickle=False)
    records_path = Path(metadata["records"])
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    geometry = PCAIndex(
        mean=arrays["mean"],
        components=arrays["components"],
        coordinates=arrays["coordinates"],
        bandwidth=float(metadata["bandwidth"]),
        median_nn_distance=float(metadata["median_nn_distance"]),
    )
    if len(records) != len(geometry.coordinates):
        raise ValueError("local source records/coordinates length mismatch")
    return LocalSourceIndex(
        geometry=geometry,
        records=records,
        metadata=metadata,
        rgb_mean=arrays["rgb_mean"] if "rgb_mean" in arrays.files else None,
        rgb_std=arrays["rgb_std"] if "rgb_std" in arrays.files else None,
    )


@lru_cache(maxsize=256)
def _load_archived_source_image(archive_path: str, image_path: str) -> Image.Image:
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(image_path)
    with Image.open(io.BytesIO(payload)) as image:
        return model_visible_image(image.convert("RGB"))


def load_archived_source_image(record: dict[str, Any]) -> Image.Image:
    return _load_archived_source_image(
        str(record["archive"]), str(record["image"])
    ).copy()


def local_source_projection(
    image: Image.Image,
    source: LocalSourceIndex,
    *,
    low_frequency_ratio: float,
    neighbors: int = 8,
    radius_fraction: float = 0.25,
) -> tuple[Image.Image, dict[str, Any]]:
    """Project once toward the KDE-weighted local source amplitude.

    The step size is bounded only by held-out source geometry. No target label,
    question, model logits, or generated text participates in the transform.
    """

    feature = spectral_descriptor(image)
    ids, weights, mean_shift = kde_neighbors(
        feature, source.geometry, k=neighbors
    )
    shift_norm = float(np.linalg.norm(mean_shift))
    radius = radius_fraction * source.geometry.median_nn_distance
    blend = min(1.0, radius / max(shift_norm, 1e-12))
    source_images = [
        load_archived_source_image(source.records[int(index)]) for index in ids
    ]
    projected = reconstruct_mean_shift_view(
        image,
        source_images,
        weights,
        blend=blend,
        low_frequency_ratio=low_frequency_ratio,
    )
    metadata = {
        "identity": False,
        "changed_band_count": None,
        "low_frequency_ratio": low_frequency_ratio,
        "neighbors": ids.tolist(),
        "neighbor_weights": weights.tolist(),
        "mean_shift_norm": shift_norm,
        "source_radius": float(radius),
        "blend": float(blend),
        "structure": structure_metrics(image, projected),
    }
    return projected, metadata


def source_mean_std_projection(
    image: Image.Image,
    source: LocalSourceIndex,
    *,
    strength: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Match RGB first/second moments toward source-only aggregate moments."""

    if source.rgb_mean is None or source.rgb_std is None:
        raise ValueError("local source index has no RGB statistics")
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    current_mean = array.reshape(-1, 3).mean(axis=0)
    current_std = array.reshape(-1, 3).std(axis=0) + 1e-6
    matched = (
        (array - current_mean.reshape(1, 1, 3))
        / current_std.reshape(1, 1, 3)
        * source.rgb_std.reshape(1, 1, 3)
        + source.rgb_mean.reshape(1, 1, 3)
    )
    output = (1.0 - strength) * array + strength * matched
    projected = Image.fromarray(
        np.clip(output, 0, 255).astype(np.uint8), mode="RGB"
    )
    return projected, {
        "identity": False,
        "changed_band_count": None,
        "strength": strength,
        "source_rgb_mean": source.rgb_mean.tolist(),
        "source_rgb_std": source.rgb_std.tolist(),
        "input_rgb_mean": current_mean.tolist(),
        "input_rgb_std": current_std.tolist(),
        "structure": structure_metrics(image, projected),
    }
