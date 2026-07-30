"""Build shared-content PubMedVision-CXR style prototypes for lineage probing."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageFilter, ImageOps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from anchor.corrected_sgta.analyze_pubmed_style_prior import (
    SEED,
    clinical_labels,
    load_features,
    question_answer,
    select_rows,
    sha256,
)


VERSION = "pubmed-shared-content-style-prototypes-v1"


def pad_square(image_bytes: bytes, size: int = 392) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.contain(
            image.convert("L"), (size, size), Image.Resampling.BILINEAR
        )
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(
        image,
        ((size - image.width) // 2, (size - image.height) // 2),
    )
    return np.asarray(canvas, dtype=np.float32) / 255.0


def smooth_low_frequency_mask(size: int, radius: float) -> np.ndarray:
    yy = np.fft.fftfreq(size)[:, None]
    xx = np.fft.fftfreq(size)[None, :]
    distance = np.sqrt(xx * xx + yy * yy)
    mask = np.zeros_like(distance, dtype=np.float32)
    active = distance < radius
    mask[active] = 0.5 * (1.0 + np.cos(np.pi * distance[active] / radius))
    return mask


def transfer_amplitude(
    base: np.ndarray,
    target_log_amplitude: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    spectrum = np.fft.fft2(base)
    log_amplitude = np.log(np.abs(spectrum) + 1e-6)
    mixed = log_amplitude + mask * (target_log_amplitude - log_amplitude)
    result = np.fft.ifft2(np.exp(mixed) * np.exp(1j * np.angle(spectrum))).real
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2048)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)

    rows = select_rows(args.manifest, args.max_images)
    features, _, labels, _, _ = load_features(rows)
    parquet_paths = sorted({str(row["source_parquet"]) for row in rows})
    columns = {
        path: pq.read_table(path, columns=["image_bytes"])["image_bytes"]
        for path in parquet_paths
    }

    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features)
    kmeans = KMeans(
        n_clusters=args.clusters, random_state=SEED, n_init=30
    ).fit(normalized)
    cluster_ids = kmeans.labels_

    normal_indices = np.flatnonzero(labels[:, -1] == 1)
    if len(normal_indices) < 16:
        raise RuntimeError("insufficient normal-reference images")
    normal_indices = normal_indices[:64]
    base_stack = []
    for index in normal_indices:
        row = rows[int(index)]
        image_bytes = columns[str(row["source_parquet"])][
            int(row["parquet_row_index"])
        ].as_py()
        base_stack.append(pad_square(image_bytes))
    base = np.median(np.stack(base_stack), axis=0)
    base_image = Image.fromarray(np.uint8(np.clip(base, 0, 1) * 255))
    base = np.asarray(
        base_image.filter(ImageFilter.GaussianBlur(radius=6)),
        dtype=np.float32,
    ) / 255.0
    Image.fromarray(np.uint8(base * 255)).save(image_dir / "shared_base.png")

    mask = smooth_low_frequency_mask(base.shape[0], args.radius)
    records: list[dict] = []
    for cluster in range(args.clusters):
        indices = np.flatnonzero(cluster_ids == cluster)
        distance = np.square(
            normalized[indices] - kmeans.cluster_centers_[cluster]
        ).sum(axis=1)
        selected = indices[np.argsort(distance)[:96]]
        for replicate in range(args.replicates):
            subset = selected[replicate:: args.replicates]
            log_amplitudes = []
            source_ids = []
            for index in subset:
                row = rows[int(index)]
                image_bytes = columns[str(row["source_parquet"])][
                    int(row["parquet_row_index"])
                ].as_py()
                array = pad_square(image_bytes)
                log_amplitudes.append(
                    np.log(np.abs(np.fft.fft2(array)) + 1e-6)
                )
                source_ids.append(str(row["id"]))
            center = np.median(np.stack(log_amplitudes), axis=0)
            prototype = transfer_amplitude(base, center, mask)
            path = image_dir / f"cluster_{cluster}_rep_{replicate}.png"
            Image.fromarray(np.uint8(prototype * 255)).save(path)
            records.append(
                {
                    "id": f"cluster-{cluster}-rep-{replicate}",
                    "cluster": cluster,
                    "replicate": replicate,
                    "image": str(path.resolve()),
                    "source_count": len(subset),
                    "source_ids_sha256": hashlib.sha256(
                        "\n".join(source_ids).encode()
                    ).hexdigest(),
                    "base_correlation": float(
                        np.corrcoef(base.ravel(), prototype.ravel())[0, 1]
                    ),
                    "mean_absolute_change": float(
                        np.abs(base - prototype).mean()
                    ),
                }
            )

    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n_source_images": len(rows),
        "clusters": args.clusters,
        "replicates": args.replicates,
        "normal_base_count": len(normal_indices),
        "radius": args.radius,
        "shared_phase_and_high_frequency": True,
        "target_data_accessed": False,
        "mean_base_correlation": float(
            np.mean([record["base_correlation"] for record in records])
        ),
        "mean_absolute_change": float(
            np.mean([record["mean_absolute_change"] for record in records])
        ),
    }
    with (args.output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
