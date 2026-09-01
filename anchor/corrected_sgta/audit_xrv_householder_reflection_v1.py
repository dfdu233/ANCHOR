#!/usr/bin/env python3
"""CPU audit of a local Householder counterfactual in XRV logit space.

The audit has a deliberately narrow purpose.  A recent cache screen found that
the full 18-dimensional XRV diagnostic signature contains information beyond a
claim-specific XRV scalar.  This script asks whether calling the nearest
positive/negative geometry a *Householder reflection* creates a new statistic
or a usable counterfactual interface to a frozen VLM.

It does not run a model and does not modify any baseline artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from anchor.corrected_sgta.screen_external_visual_increment_v1 import FINDINGS, load_claims
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import FINDING_TARGETS, XRV_LABELS


VERSION = "xrv-local-householder-audit-v1"


def load_logits(path: Path) -> dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=False)
    if [str(x) for x in pack["labels"]] != list(XRV_LABELS):
        raise ValueError("XRV label order drift")
    return {
        str(image_id): np.asarray(vector, dtype=np.float64)
        for image_id, vector in zip(pack["image_ids"], pack["logits"])
    }


def standardize(dev: list[dict[str, Any]], logits: dict[str, np.ndarray]):
    image_ids = sorted({row["image_id"] for row in dev})
    matrix = np.stack([logits[image_id] for image_id in image_ids])
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale < 1e-8] = 1.0
    return {key: (value - mean) / scale for key, value in logits.items()}


def attach_scalar(rows: list[dict[str, Any]], logits: dict[str, np.ndarray]) -> None:
    index = {name: i for i, name in enumerate(XRV_LABELS)}
    for row in rows:
        row["xrv_scalar"] = float(
            max(logits[row["image_id"]][index[name]] for name in FINDING_TARGETS[row["finding"]])
        )


def class_banks(dev, z, finding):
    selected = [row for row in dev if row["finding"] == finding]
    return {
        label: (
            np.stack([z[row["image_id"]] for row in selected if row["label"] == label]),
            np.asarray([row["image_id"] for row in selected if row["label"] == label]),
        )
        for label in (0, 1)
    }


def nearest(query, matrix, ids, query_id, k):
    eligible = ids != query_id
    current = matrix[eligible]
    current_ids = ids[eligible]
    distance = np.linalg.norm(current - query[None, :], axis=1)
    order = np.argsort(distance)[: min(k, len(distance))]
    return current[order], current_ids[order], distance[order]


def attach_geometry(dev, target, logits, k):
    z = standardize(dev, logits)
    diagnostics = []
    for finding in FINDINGS:
        banks = class_banks(dev, z, finding)
        for row in target:
            if row["finding"] != finding:
                continue
            query = z[row["image_id"]]
            a0s, _, d0s = nearest(query, *banks[0], row["image_id"], k)
            a1s, _, d1s = nearest(query, *banks[1], row["image_id"], k)
            a0, a1 = a0s[0], a1s[0]
            delta = a1 - a0
            length = float(np.linalg.norm(delta))
            if length < 1e-12:
                raise ValueError("Coincident positive and negative anchors")
            normal = delta / length
            midpoint = (a0 + a1) / 2.0
            signed = float((query - midpoint) @ normal)
            reflected = query - 2.0 * signed * normal
            reflected_twice = reflected - 2.0 * float((reflected - midpoint) @ normal) * normal

            # Exact identity: difference of squared anchor distances is a
            # scaled signed coordinate along the bisector normal.
            squared_difference = float(np.sum((query - a0) ** 2) - np.sum((query - a1) ** 2))
            identity_error = abs(squared_difference - 2.0 * length * signed)
            swap_error = max(
                abs(np.linalg.norm(reflected - a1) - np.linalg.norm(query - a0)),
                abs(np.linalg.norm(reflected - a0) - np.linalg.norm(query - a1)),
            )
            involution_error = float(np.linalg.norm(reflected_twice - query))

            row["knn_boundary"] = float(d0s.mean() - d1s.mean())
            row["householder_coordinate"] = signed
            row["nearest_squared_distance_difference"] = squared_difference
            diagnostics.append((identity_error, swap_error, involution_error))
    return diagnostics


def normalizers(rows, fields):
    result = {}
    for finding in FINDINGS:
        result[finding] = {}
        selected = [row for row in rows if row["finding"] == finding]
        for field in fields:
            values = np.asarray([row[field] for row in selected])
            result[finding][field] = (float(values.mean()), float(max(values.std(), 1e-8)))
    return result


def design(rows, norm, fields):
    output = []
    for row in rows:
        vector = [float(row["finding"] == finding) for finding in FINDINGS[:-1]]
        for field in fields:
            mean, scale = norm[row["finding"]][field]
            vector.append((row[field] - mean) / scale)
        output.append(vector)
    return np.asarray(output)


def macro_auc(rows, score):
    values = []
    for finding in FINDINGS:
        index = [i for i, row in enumerate(rows) if row["finding"] == finding]
        values.append(roc_auc_score([rows[i]["label"] for i in index], score[index]))
    return float(np.mean(values))


def fit_predict(dev, test, fields, seed):
    norm = normalizers(dev, fields)
    model = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    model.fit(design(dev, norm, fields), [row["label"] for row in dev])
    return model.predict_proba(design(test, norm, fields))[:, 1]


def bootstrap(rows, a, b, draws, seed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["image_id"]].append(i)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(draws):
        sampled = rng.choice(image_ids, len(image_ids), replace=True)
        index = np.asarray([i for image_id in sampled for i in groups[image_id]])
        current = [rows[i] for i in index]
        delta.append(macro_auc(current, a[index]) - macro_auc(current, b[index]))
    return {
        "mean": float(np.mean(delta)),
        "ci95": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
    }


def analyze(dev_path, test_path, model_name, logits, k, draws, seed):
    dev = load_claims(dev_path, "development", model_name)
    test = load_claims(test_path, "confirmation", model_name)
    attach_scalar(dev + test, logits)
    diagnostics = attach_geometry(dev, dev + test, logits, k)
    specs = {
        "vlm_plus_scalar": ("margin", "xrv_scalar"),
        "plus_knn_boundary": ("margin", "xrv_scalar", "knn_boundary"),
        "plus_householder": ("margin", "xrv_scalar", "householder_coordinate"),
        "plus_both": (
            "margin",
            "xrv_scalar",
            "knn_boundary",
            "householder_coordinate",
        ),
    }
    prediction = {name: fit_predict(dev, test, fields, seed) for name, fields in specs.items()}
    point = {name: macro_auc(test, score) for name, score in prediction.items()}
    return {
        "n": len(test),
        "images": len({row["image_id"] for row in test}),
        "macro_auroc": point,
        "householder_over_knn": bootstrap(
            test, prediction["plus_both"], prediction["plus_knn_boundary"], draws, seed
        ),
        "knn_over_scalar": bootstrap(
            test, prediction["plus_knn_boundary"], prediction["vlm_plus_scalar"], draws, seed + 1
        ),
        "max_exact_identity_error": float(max(x[0] for x in diagnostics)),
        "max_anchor_distance_swap_error": float(max(x[1] for x in diagnostics)),
        "max_involution_error": float(max(x[2] for x in diagnostics)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-test", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-test", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    logits = load_logits(args.xrv_logits)
    result = {
        "version": VERSION,
        "scope": "CPU/cache-only formula and feature audit; not a mitigation result",
        "exact_result": (
            "For nearest opposite-class anchors a0,a1, the Householder signed coordinate "
            "equals (||z-a0||^2-||z-a1||^2)/(2||a1-a0||). The reflection is an "
            "involution and swaps distances to these anchors; it therefore renames a "
            "nearest-counterexample decision coordinate rather than adding evidence."
        ),
        "analyses": {
            "huatuo": analyze(
                args.huatuo_dev, args.huatuo_test, "huatuo", logits, args.k, args.draws, args.seed
            ),
            "hulu": analyze(
                args.hulu_dev, args.hulu_test, "hulu", logits, args.k, args.draws, args.seed + 10
            ),
        },
        "intervention_boundary": (
            "Reflecting XRV logits alone cannot change a frozen VLM because XRV is not on "
            "the VLM causal path. A nontrivial bridge must edit the image, VLM state, token "
            "distribution, or output; these are respectively visual counterfactuals, "
            "activation steering, expert-guided decoding, or verification."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
