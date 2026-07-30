"""Audit style-conditioned clinical priors in PubMedVision CXR training data.

This is a source-data mechanism probe, not a target-domain performance
experiment. It asks whether acquisition/presentation-style proxies add
out-of-article information about clinical concepts beyond the question text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


VERSION = "pubmed-style-prior-audit-v1"
SEED = 2027
LABEL_PATTERNS = {
    "pneumothorax": r"\bpneumothorax\b",
    "effusion": r"\b(?:pleural )?effusion\b",
    "opacity": r"\b(?:opacity|opacities|infiltrate|infiltration|consolidation|pneumonia)\b",
    "cardiomegaly": r"\b(?:cardiomegaly|enlarged heart|cardiac enlargement)\b",
    "edema": r"\b(?:pulmonary )?edema\b",
    "fracture": r"\bfractur(?:e|es|ed)\b",
    "device": r"\b(?:catheter|tube|pacemaker|defibrillator|implant|line|port)\b",
    "normal": r"\b(?:no acute|normal chest|unremarkable|no abnormalit|clear lungs)\b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(record: dict) -> str:
    return str(record.get("image_sha256") or record["id"])


def select_rows(manifest: Path, max_images: int) -> list[dict]:
    by_image: dict[str, dict] = {}
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("is_strict_cxr"):
                by_image.setdefault(stable_key(row), row)
    rows = sorted(
        by_image.values(),
        key=lambda row: hashlib.sha256(stable_key(row).encode()).hexdigest(),
    )
    return rows[:max_images]


def question_answer(record: dict) -> tuple[str, str]:
    question = " ".join(
        turn["content"]
        for turn in record["conversations"]
        if turn["role"] == "user"
    )
    answer = " ".join(
        turn["content"]
        for turn in record["conversations"]
        if turn["role"] == "assistant"
    )
    return question, answer


def question_family(question: str) -> str:
    text = question.lower()
    if re.search(r"\b(?:device|tube|catheter|line|implant|pacemaker)\b", text):
        return "device"
    if re.search(r"\b(?:where|location|side|position)\b", text):
        return "location"
    if re.search(r"\b(?:diagnos|condition|disease)\b", text):
        return "diagnosis"
    if re.search(r"\b(?:abnormal|finding|observed|visible|show)\b", text):
        return "finding"
    if re.match(r"^(?:is|are|does|do|has|have|can|could)\b", text):
        return "polar"
    return "interpretation"


def clinical_labels(answer: str) -> np.ndarray:
    text = answer.lower()
    return np.asarray(
        [bool(re.search(pattern, text)) for pattern in LABEL_PATTERNS.values()],
        dtype=np.int64,
    )


def radial_profile(power: np.ndarray, bins: int = 16) -> np.ndarray:
    height, width = power.shape
    yy = np.fft.fftfreq(height)[:, None]
    xx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(yy * yy + xx * xx)
    edges = np.linspace(0.0, float(radius.max()) + 1e-8, bins + 1)
    values = np.empty(bins, dtype=np.float32)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        values[index] = float(power[mask].mean()) if mask.any() else 0.0
    values -= values.mean()
    return values


def style_features(image_bytes: bytes) -> np.ndarray:
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        gray = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    border = np.concatenate(
        [
            array[:16].ravel(),
            array[-16:].ravel(),
            array[16:-16, :16].ravel(),
            array[16:-16, -16:].ravel(),
        ]
    )
    gy, gx = np.gradient(array)
    gradient = np.sqrt(gx * gx + gy * gy)
    log_power = np.log1p(np.abs(np.fft.fft2(array)) ** 2)
    quantiles = np.linspace(0.05, 0.95, 10)
    features = np.concatenate(
        [
            np.quantile(array, quantiles),
            np.quantile(border, quantiles),
            np.quantile(gradient, quantiles),
            radial_profile(log_power),
            np.asarray(
                [
                    math.log(max(width, 1) / max(height, 1)),
                    array.mean(),
                    array.std(),
                    border.mean(),
                    border.std(),
                    np.mean(array < 0.02),
                    np.mean(array > 0.98),
                    np.mean(gradient > np.quantile(gradient, 0.9)),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return features.astype(np.float32)


def load_features(rows: list[dict]) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[str]]:
    parquet_paths = sorted({str(row["source_parquet"]) for row in rows})
    columns = {
        path: pq.read_table(path, columns=["image_bytes"])["image_bytes"]
        for path in parquet_paths
    }
    features: list[np.ndarray] = []
    questions: list[str] = []
    labels: list[np.ndarray] = []
    groups: list[str] = []
    families: list[str] = []
    for index, row in enumerate(rows):
        image_bytes = columns[str(row["source_parquet"])][
            int(row["parquet_row_index"])
        ].as_py()
        question, answer = question_answer(row)
        features.append(style_features(image_bytes))
        questions.append(question)
        labels.append(clinical_labels(answer))
        groups.append(str(row["group_id"]))
        families.append(question_family(question))
        if (index + 1) % 250 == 0:
            print(json.dumps({"processed": index + 1}), flush=True)
    return np.stack(features), questions, np.stack(labels), groups, families


def grouped_bootstrap_aucs(
    y: np.ndarray,
    question_score: np.ndarray,
    style_score: np.ndarray,
    combined_score: np.ndarray,
    groups: np.ndarray,
    draws: int = 500,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    deltas = []
    style_aucs = []
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if np.unique(y[indices]).size < 2:
            continue
        deltas.append(
            roc_auc_score(y[indices], combined_score[indices])
            - roc_auc_score(y[indices], question_score[indices])
        )
        style_aucs.append(roc_auc_score(y[indices], style_score[indices]))
    if not deltas:
        nan_ci = (float("nan"), float("nan"))
        return nan_ci, nan_ci
    return (
        tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975])),
        tuple(float(value) for value in np.quantile(style_aucs, [0.025, 0.975])),
    )


def fit_scores(
    x_train: sparse.spmatrix,
    x_test: sparse.spmatrix,
    y_train: np.ndarray,
) -> np.ndarray:
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def plot_results(
    output: Path,
    metrics: list[dict],
    prevalence: np.ndarray,
    cluster_sizes: np.ndarray,
) -> None:
    labels = [item["label"] for item in metrics]
    delta = [item["combined_minus_question_auc"] for item in metrics]
    low = [item["delta_auc_ci95"][0] for item in metrics]
    high = [item["delta_auc_ci95"][1] for item in metrics]
    errors = np.asarray(
        [[value - lo for value, lo in zip(delta, low)], [hi - value for value, hi in zip(delta, high)]]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    colors = ["#16697A" if value > 0 else "#D1495B" for value in delta]
    axes[0].barh(labels[::-1], np.asarray(delta)[::-1], color=colors[::-1])
    axes[0].errorbar(
        np.asarray(delta)[::-1],
        np.arange(len(labels)),
        xerr=errors[:, ::-1],
        fmt="none",
        ecolor="#263238",
        capsize=2,
        linewidth=1,
    )
    axes[0].axvline(0, color="#263238", linewidth=0.8)
    axes[0].set_xlabel("AUROC gain: question + style over question")
    axes[0].set_title("Out-of-article incremental signal")
    image = axes[1].imshow(prevalence.T, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    axes[1].set_xticks(range(len(cluster_sizes)), [f"C{i}\n(n={n})" for i, n in enumerate(cluster_sizes)])
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_title("Cluster prevalence minus global prevalence")
    axes[1].set_xlabel("Fourier/presentation style cluster")
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="prevalence difference")
    fig.suptitle("PubMedVision-CXR style–clinical prior audit", fontweight="bold")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = select_rows(args.manifest, args.max_images)
    features, questions, labels, groups, families = load_features(rows)
    groups_array = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train, test = next(splitter.split(features, groups=groups_array))

    scaler = StandardScaler().fit(features[train])
    style_train = scaler.transform(features[train])
    style_test = scaler.transform(features[test])
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=5000,
        sublinear_tf=True,
    )
    question_train = vectorizer.fit_transform(np.asarray(questions)[train])
    question_test = vectorizer.transform(np.asarray(questions)[test])
    combined_train = sparse.hstack([question_train, sparse.csr_matrix(style_train)], format="csr")
    combined_test = sparse.hstack([question_test, sparse.csr_matrix(style_test)], format="csr")

    kmeans = KMeans(n_clusters=6, random_state=SEED, n_init=20)
    kmeans.fit(style_train)
    clusters = kmeans.predict(scaler.transform(features))
    cluster_sizes = np.bincount(clusters[test], minlength=6)
    global_prevalence = labels[test].mean(axis=0)
    prevalence = np.zeros((6, labels.shape[1]), dtype=np.float32)
    for cluster in range(6):
        subset = test[clusters[test] == cluster]
        prevalence[cluster] = (
            labels[subset].mean(axis=0) - global_prevalence if len(subset) else np.nan
        )

    metrics: list[dict] = []
    for label_index, label in enumerate(LABEL_PATTERNS):
        y_train = labels[train, label_index]
        y_test = labels[test, label_index]
        if y_train.sum() < 10 or y_test.sum() < 5 or np.unique(y_test).size < 2:
            continue
        question_score = fit_scores(question_train, question_test, y_train)
        style_score = fit_scores(
            sparse.csr_matrix(style_train), sparse.csr_matrix(style_test), y_train
        )
        combined_score = fit_scores(combined_train, combined_test, y_train)
        question_auc = float(roc_auc_score(y_test, question_score))
        style_auc = float(roc_auc_score(y_test, style_score))
        combined_auc = float(roc_auc_score(y_test, combined_score))
        delta_ci, style_ci = grouped_bootstrap_aucs(
            y_test, question_score, style_score, combined_score, groups_array[test]
        )
        metrics.append(
            {
                "label": label,
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
                "question_auc": question_auc,
                "style_only_auc": style_auc,
                "style_only_auc_ci95": list(style_ci),
                "combined_auc": combined_auc,
                "combined_minus_question_auc": combined_auc - question_auc,
                "delta_auc_ci95": list(delta_ci),
            }
        )

    supported = [
        item
        for item in metrics
        if item["combined_minus_question_auc"] >= 0.03
        and item["delta_auc_ci95"][0] > 0
    ]
    unconditional = [
        item
        for item in metrics
        if item["style_only_auc"] >= 0.55
        and item["style_only_auc_ci95"][0] > 0.5
    ]
    result = {
        "version": VERSION,
        "status": "source_only_diagnostic",
        "claim_ceiling": (
            "training distribution contains an out-of-article association between "
            "presentation-style proxies and selected answer concepts; this does not "
            "establish that the VLM uses the association"
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n": len(rows),
        "train_n": len(train),
        "test_n": len(test),
        "unique_groups": len(set(groups)),
        "split": "70/30 GroupShuffleSplit by PMC group_id",
        "features": (
            "phase-free radial log-power, intensity/gradient/border quantiles, "
            "aspect ratio and saturation statistics"
        ),
        "question_control": "word/bi-gram TF-IDF logistic regression",
        "labels": metrics,
        "style_clusters": {
            "test_sizes": cluster_sizes.tolist(),
            "prevalence_minus_global": prevalence.tolist(),
            "label_order": list(LABEL_PATTERNS),
        },
        "decision": {
            "criterion": "at least three concepts have AUROC gain >=0.03 with grouped-bootstrap CI lower bound >0",
            "supported_concepts": [item["label"] for item in supported],
            "gate_passed": len(supported) >= 3,
            "unconditional_style_signal": [
                item["label"] for item in unconditional
            ],
        },
        "limitations": [
            "answer concepts are regex-derived rather than radiologist labels",
            "Fourier/presentation features can retain some coarse clinical content",
            "PubMed figures do not expose hospital or scanner identity",
            "association in training data is necessary but not sufficient evidence of model shortcut use",
        ],
        "target_data_accessed": False,
        "seed": SEED,
    }
    with (args.output / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    plot_results(
        args.output / "style_prior_audit.png",
        metrics,
        prevalence,
        cluster_sizes,
    )
    with (args.output / "label_metrics.jsonl").open("w") as handle:
        for item in metrics:
            handle.write(json.dumps(item) + "\n")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
