"""Test whether PubMedVision CXR style acts through a question-style interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from anchor.corrected_sgta.analyze_pubmed_style_prior import (
    LABEL_PATTERNS,
    SEED,
    load_features,
    select_rows,
    sha256,
)


VERSION = "pubmed-style-question-interaction-v1"


def interaction_features(question: np.ndarray, style: np.ndarray) -> np.ndarray:
    """Flatten the per-sample Kronecker product q x s."""
    return np.einsum("bi,bj->bij", question, style).reshape(
        len(question), question.shape[1] * style.shape[1]
    )


def shuffled_within_family(
    style: np.ndarray,
    families: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = style.copy()
    for family in np.unique(families):
        indices = np.flatnonzero(families == family)
        result[indices] = style[rng.permutation(indices)]
    return result


def fit_score(x_train: np.ndarray, x_test: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = LogisticRegression(
        C=0.1,
        solver="liblinear",
        max_iter=1500,
        random_state=SEED,
    )
    model.fit(x_train, y)
    return model.predict_proba(x_test)[:, 1]


def grouped_bootstrap_deltas(
    y: np.ndarray,
    baseline: np.ndarray,
    interaction: np.ndarray,
    shuffled: np.ndarray,
    groups: np.ndarray,
    draws: int = 1000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    deltas_additive: list[float] = []
    deltas_shuffled: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if np.unique(y[indices]).size < 2:
            continue
        auc_interaction = roc_auc_score(y[indices], interaction[indices])
        deltas_additive.append(
            auc_interaction - roc_auc_score(y[indices], baseline[indices])
        )
        deltas_shuffled.append(
            auc_interaction - roc_auc_score(y[indices], shuffled[indices])
        )
    return {
        "interaction_minus_additive_ci95": np.quantile(
            deltas_additive, [0.025, 0.975]
        ).tolist(),
        "interaction_minus_shuffled_ci95": np.quantile(
            deltas_shuffled, [0.025, 0.975]
        ).tolist(),
    }


def plot_metrics(path: Path, metrics: list[dict]) -> None:
    labels = [item["label"] for item in metrics]
    real = np.asarray([item["interaction_minus_additive_auc"] for item in metrics])
    shuffled = np.asarray(
        [item["shuffled_minus_additive_auc"] for item in metrics]
    )
    lower = np.asarray(
        [item["interaction_minus_additive_ci95"][0] for item in metrics]
    )
    upper = np.asarray(
        [item["interaction_minus_additive_ci95"][1] for item in metrics]
    )
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.5), constrained_layout=True)
    width = 0.36
    ax.barh(
        positions + width / 2,
        real,
        width,
        color="#16697A",
        label=r"real $q\otimes s$",
    )
    ax.barh(
        positions - width / 2,
        shuffled,
        width,
        color="#B8C4CE",
        label=r"shuffled $q\otimes s$",
    )
    ax.errorbar(
        real,
        positions + width / 2,
        xerr=np.vstack([real - lower, upper - real]),
        fmt="none",
        ecolor="#263238",
        capsize=2,
        linewidth=1,
    )
    ax.axvline(0, color="#263238", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("AUROC gain over question + additive style")
    ax.set_title("Does style switch a question-conditioned clinical prior?")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = select_rows(args.manifest, args.max_images)
    style, questions, labels, groups, families = load_features(rows)
    questions = np.asarray(questions)
    groups = np.asarray(groups)
    families = np.asarray(families)
    folds = GroupKFold(n_splits=5)

    prediction_names = [
        "question",
        "additive",
        "interaction",
        "shuffled_interaction",
    ]
    predictions = {
        name: np.full(labels.shape, np.nan, dtype=np.float64)
        for name in prediction_names
    }

    for fold, (train, test) in enumerate(folds.split(style, groups=groups)):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=5000,
            sublinear_tf=True,
        )
        q_train_sparse = vectorizer.fit_transform(questions[train])
        q_test_sparse = vectorizer.transform(questions[test])
        q_svd = TruncatedSVD(n_components=32, random_state=SEED)
        q_train = q_svd.fit_transform(q_train_sparse)
        q_test = q_svd.transform(q_test_sparse)
        q_scaler = StandardScaler().fit(q_train)
        q_train = q_scaler.transform(q_train)
        q_test = q_scaler.transform(q_test)

        style_scaler = StandardScaler().fit(style[train])
        s_train_full = style_scaler.transform(style[train])
        s_test_full = style_scaler.transform(style[test])
        style_pca = PCA(n_components=12, whiten=True, random_state=SEED)
        s_train = style_pca.fit_transform(s_train_full)
        s_test = style_pca.transform(s_test_full)

        additive_train = np.hstack([q_train, s_train])
        additive_test = np.hstack([q_test, s_test])
        interaction_train = np.hstack(
            [additive_train, interaction_features(q_train, s_train)]
        )
        interaction_test = np.hstack(
            [additive_test, interaction_features(q_test, s_test)]
        )
        rng = np.random.default_rng(SEED + fold)
        shuffled_train = shuffled_within_family(
            s_train, families[train], rng
        )
        shuffled_test = shuffled_within_family(s_test, families[test], rng)
        shuffled_interaction_train = np.hstack(
            [
                q_train,
                shuffled_train,
                interaction_features(q_train, shuffled_train),
            ]
        )
        shuffled_interaction_test = np.hstack(
            [
                q_test,
                shuffled_test,
                interaction_features(q_test, shuffled_test),
            ]
        )

        for label_index in range(labels.shape[1]):
            y_train = labels[train, label_index]
            if np.unique(y_train).size < 2:
                continue
            predictions["question"][test, label_index] = fit_score(
                q_train, q_test, y_train
            )
            predictions["additive"][test, label_index] = fit_score(
                additive_train, additive_test, y_train
            )
            predictions["interaction"][test, label_index] = fit_score(
                interaction_train, interaction_test, y_train
            )
            predictions["shuffled_interaction"][test, label_index] = fit_score(
                shuffled_interaction_train, shuffled_interaction_test, y_train
            )
        print(json.dumps({"completed_fold": fold + 1}), flush=True)

    metrics: list[dict] = []
    for label_index, label in enumerate(LABEL_PATTERNS):
        y = labels[:, label_index]
        valid = np.logical_and.reduce(
            [np.isfinite(predictions[name][:, label_index]) for name in prediction_names]
        )
        if np.unique(y[valid]).size < 2:
            continue
        auc = {
            name: float(
                roc_auc_score(y[valid], predictions[name][valid, label_index])
            )
            for name in prediction_names
        }
        ci = grouped_bootstrap_deltas(
            y[valid],
            predictions["additive"][valid, label_index],
            predictions["interaction"][valid, label_index],
            predictions["shuffled_interaction"][valid, label_index],
            groups[valid],
        )
        metrics.append(
            {
                "label": label,
                "positive": int(y[valid].sum()),
                **{f"{name}_auc": value for name, value in auc.items()},
                "interaction_minus_additive_auc": (
                    auc["interaction"] - auc["additive"]
                ),
                "shuffled_minus_additive_auc": (
                    auc["shuffled_interaction"] - auc["additive"]
                ),
                "interaction_minus_shuffled_auc": (
                    auc["interaction"] - auc["shuffled_interaction"]
                ),
                **ci,
            }
        )

    supported = [
        item
        for item in metrics
        if item["interaction_minus_additive_auc"] >= 0.02
        and item["interaction_minus_additive_ci95"][0] > 0
        and item["interaction_minus_shuffled_ci95"][0] > 0
    ]
    result = {
        "version": VERSION,
        "status": "source_only_conditional_interaction_diagnostic",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n": len(rows),
        "unique_groups": int(len(np.unique(groups))),
        "split": "five-fold GroupKFold by PMC group_id",
        "question_representation": "TF-IDF -> fixed 32-dimensional SVD",
        "style_representation": "54 handcrafted style proxies -> fixed 12-dimensional PCA",
        "interaction": "full 32x12 Kronecker product with L2-regularized logistic regression",
        "negative_control": "style vectors independently shuffled within question family in each train/test fold",
        "labels": metrics,
        "decision": {
            "criterion": (
                "at least three concepts: real interaction gain >=0.02, "
                "CI lower >0 over additive and shuffled controls"
            ),
            "supported_concepts": [item["label"] for item in supported],
            "gate_passed": len(supported) >= 3,
        },
        "claim_ceiling": (
            "A positive result would show source-distribution q-style interaction; "
            "it would still not prove VLM causal use."
        ),
        "target_data_accessed": False,
        "seed": SEED,
    }
    with (args.output / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    with (args.output / "label_metrics.jsonl").open("w") as handle:
        for item in metrics:
            handle.write(json.dumps(item) + "\n")
    plot_metrics(args.output / "style_question_interaction.png", metrics)
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
