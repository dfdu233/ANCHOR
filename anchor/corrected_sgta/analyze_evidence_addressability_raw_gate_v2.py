#!/usr/bin/env python3
"""Leakage-resistant Stage-2 Addressability selection and confirmation.

``select`` uses development labels and writes an immutable model-selection
receipt. ``confirm`` accepts only that receipt, refuses overwrite, opens the
fresh fixed-panel holdout once, and compares the nested models F+M+N versus
F+M+N+V.  The output establishes at most incremental decodability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from corrected_sgta.analyze_evidence_addressability_gate_v1 import (
    brier,
    fit_logistic,
    metric_summary,
    reader_nll,
)
from corrected_sgta.analyze_evidence_addressability_raw_gate_v1 import (
    PCS,
    REGULARIZATION,
    REPRESENTATIONS,
    CentroidTransform,
    NuisanceTransform,
    PCATransform,
    fit_base_transform,
    fit_centroid,
    fit_nuisance,
    fit_pca,
    join_raw,
    load_raw as load_raw_unchecked,
)


PROTOCOL = "evidence-addressability-raw-increment-gate-v2"
PANEL = {"R8", "R9", "R10"}
FOLDS = 5
FRESH_PER_CELL = 19
DEV_PER_CELL = 60
BOOTSTRAP_DRAWS = 5000
PERMUTATIONS = 1000
FROZEN_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
ANALYSIS_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("analyze_evidence_addressability_gate_v1.py"),
    Path(__file__).with_name("analyze_evidence_addressability_raw_gate_v1.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def analysis_contract() -> dict[str, Any]:
    contract = {
        "sources": {
            str(path.resolve()): sha256(path)
            for path in ANALYSIS_SOURCES
        },
        "python": sys.version,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutations": PERMUTATIONS,
        "bootstrap_unit": "19-row resampling independently within each frozen finding_x_vote cell",
    }
    return {**contract, "sha256": canonical_sha256(contract)}


def require_findings(values: Iterable[str]) -> tuple[str, ...]:
    findings = tuple(str(value) for value in values)
    if findings != FROZEN_FINDINGS or len(findings) != len(set(findings)):
        raise ValueError(
            f"findings must exactly equal the frozen ordered tuple: {FROZEN_FINDINGS}"
        )
    return findings


def load_holdout_contract(lock_path: Path, exposure_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    expected_cells = {
        f"{finding}:{vote}": FRESH_PER_CELL
        for finding in FROZEN_FINDINGS
        for vote in range(4)
    }
    if (
        lock.get("version") != "vindr-addressability-fresh-holdout-v2"
        or tuple(lock.get("findings", [])) != FROZEN_FINDINGS
        or set(lock.get("reader_panel", [])) != PANEL
        or lock.get("per_cell") != FRESH_PER_CELL
        or lock.get("claims") != len(FROZEN_FINDINGS) * 4 * FRESH_PER_CELL
        or lock.get("unique_images") != lock.get("claims")
        or lock.get("cell_counts") != expected_cells
        or lock.get("one_claim_per_image") is not True
    ):
        raise ValueError("holdout lock receipt violates the frozen 7x4x19 contract")
    manifest = Path(lock["manifest"])
    exclusion = Path(lock["exclusion_manifest"])
    if (
        sha256(manifest) != lock.get("manifest_sha256")
        or sha256(exclusion) != lock.get("exclusion_manifest_sha256")
    ):
        raise ValueError("holdout manifest/exclusion hash differs from lock receipt")
    if (
        exposure.get("protocol") != "vindr-holdout-prior-exposure-audit-v1"
        or exposure.get("status") != "complete"
        or exposure.get("holdout_manifest_sha256") != lock.get("manifest_sha256")
        or exposure.get("direct_ce_exclusion_manifest_sha256")
        != lock.get("exclusion_manifest_sha256")
        or exposure.get("direct_ce_exclusion_verified") is not True
        or exposure.get("direct_ce_overlap_count") != 0
        or exposure.get("endpoint_prospective") is not True
        or exposure.get("image_unseen_across_repository") is not False
        or exposure.get("audit_code_sha256")
        != sha256(Path(__file__).with_name("audit_vindr_holdout_prior_exposure_v1.py"))
        or set(exposure.get("audited_prior_roots", {}))
        != {
            str(Path("/home/dbw/ANCHOR/corrected_runs/ascc").resolve()),
            str(Path("/home/dbw/ANCHOR/corrected_runs/specificity_ratchet").resolve()),
            str(Path("/home/dbw/ANCHOR/corrected_runs/ppi_v31").resolve()),
        }
        or exposure.get("claim_boundary")
        != "prospective endpoint-held-out relative to the prior direct-CE manifest; not image-unseen"
    ):
        raise ValueError("holdout exposure audit is absent, stale, or overclaims image novelty")
    return {
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": sha256(lock_path),
        "manifest_sha256": lock["manifest_sha256"],
        "exclusion_manifest_sha256": lock["exclusion_manifest_sha256"],
        "record_keys_sha256": lock["record_keys_sha256"],
        "exposure_path": str(exposure_path.resolve()),
        "exposure_sha256": sha256(exposure_path),
        "image_unseen_across_repository": False,
        "endpoint_prospective": True,
    }


def artifact_rows(
    directory: Path,
    model: str,
    findings: tuple[str, ...],
    expected_per_cell: int,
) -> dict[str, Any]:
    config_path = directory / "config.json"
    summary_path = directory / "summary.json"
    metadata_path = directory / "metadata.jsonl"
    hidden_path = directory / "hidden_states.npz"
    for path in (config_path, summary_path, metadata_path, hidden_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if config.get("model_id") != model or summary.get("status") != "complete":
        raise ValueError(f"hidden artifact model/status mismatch: {directory}")
    all_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [row for row in all_rows if str(row["finding"]) in set(findings)]
    with np.load(hidden_path, allow_pickle=False) as archive:
        layers = np.asarray(archive["layers"], dtype=np.int64)
    final_layer = str(int(layers[-1]))
    keys, margins = [], []
    cells: Counter[tuple[str, int]] = Counter()
    for row in rows:
        finding, vote = str(row["finding"]), int(row["positive_votes"])
        if finding not in findings or vote not in range(4):
            raise ValueError("hidden row lies outside frozen finding/vote universe")
        if set(str(value) for value in row.get("reader_ids", [])) != PANEL:
            reader_votes = row.get("reader_votes", [])
            if set(str(item.get("rad_id")) for item in reader_votes) != PANEL:
                raise ValueError("hidden row is not the frozen R8/R9/R10 panel")
        key = str(row.get("record_key") or f"{finding}:{row['image_id']}")
        keys.append(key)
        cells[(finding, vote)] += 1
        logits = row["diagnostic_plain_logit_lens"][final_layer]
        margins.append(float(logits["supported"]) - float(logits["refuted"]))
    if len(keys) != len(set(keys)):
        raise ValueError("hidden artifact has duplicate record keys")
    expected = {(finding, vote): expected_per_cell for finding in findings for vote in range(4)}
    if dict(cells) != expected:
        raise ValueError(f"cell-count contract mismatch: observed={dict(cells)} expected={expected}")
    output = {
        "directory": str(directory.resolve()),
        "config": config,
        "config_sha256": sha256(config_path),
        "summary_sha256": sha256(summary_path),
        "metadata_sha256": sha256(metadata_path),
        "hidden_sha256": sha256(hidden_path),
        "finding": np.asarray([str(row["finding"]) for row in rows]),
        "votes": np.asarray([int(row["positive_votes"]) for row in rows], dtype=np.int64),
        "image": np.asarray([str(row["image_id"]) for row in rows]),
        "margin": np.asarray(margins, dtype=np.float64),
        "record_key": np.asarray(keys),
        "rows": rows,
        "record_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "labels_sha256": canonical_sha256(
            [(key, int(vote)) for key, vote in zip(keys, [row["positive_votes"] for row in rows])]
        ),
    }
    return output


def strict_raw(directory: Path, model: str) -> dict[str, Any]:
    config_path = directory / "config.json"
    summary_path = directory / "summary.json"
    canary_path = directory / "prompt_invariance_canary.json"
    for path in (config_path, summary_path, canary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    static = {
        key: value
        for key, value in config.items()
        if key not in {"created_at", "command", "fingerprint"}
    }
    if config.get("fingerprint") != canonical_sha256(static):
        raise ValueError("raw config fingerprint is internally inconsistent")
    if config.get("model") != model or summary.get("model") != model:
        raise ValueError("raw feature/model mismatch")
    if summary.get("status") != "complete" or canary.get("status") != "pass":
        raise ValueError("raw artifact or prompt-invariance canary is incomplete")
    metadata_path, features_path = directory / "metadata.jsonl", directory / "features.npz"
    if (
        summary.get("metadata_sha256") != sha256(metadata_path)
        or summary.get("features_sha256") != sha256(features_path)
        or summary.get("prompt_invariance_canary_sha256") != sha256(canary_path)
        or summary.get("config_fingerprint") != config.get("fingerprint")
    ):
        raise ValueError("raw summary/file hash mismatch")
    raw = load_raw_unchecked(directory)
    raw.update(
        {
            "config": config,
            "config_sha256": sha256(config_path),
            "summary_sha256": sha256(summary_path),
            "canary_sha256": sha256(canary_path),
        }
    )
    return raw


def bind_hidden_raw(hidden: dict[str, Any], raw: dict[str, Any], role: str) -> None:
    contract = raw["config"][f"{role}_hidden_contract"]
    if (
        contract.get("directory") != hidden["directory"]
        or contract.get("config_sha256") != hidden["config_sha256"]
        or contract.get("metadata_sha256") != hidden["metadata_sha256"]
        or contract.get("summary_sha256") != hidden["summary_sha256"]
    ):
        raise ValueError(f"raw artifact is not bound to {role} hidden artifact")


def grouped_folds(data: dict[str, Any]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    strata = np.asarray(
        [f"{finding}:{vote}" for finding, vote in zip(data["finding"], data["votes"])]
    )
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    return splitter.split(np.zeros(len(strata)), strata, groups=data["image"])


def visual_block(
    transform: CentroidTransform | PCATransform,
    family: str,
    finding: np.ndarray,
    margin: np.ndarray,
    visual: np.ndarray,
    size: int,
) -> np.ndarray:
    base_width = 2 * len(transform.base.findings)
    if family == "centroid_scalar":
        design = transform.design(finding, margin, visual)
    else:
        design = transform.design(finding, margin, visual, size)
    return design[:, base_width:]


def select_nested(data: dict[str, Any], findings: tuple[str, ...]) -> dict[str, Any]:
    base_scores = {
        value: np.full(len(data["votes"]), np.nan, dtype=np.float64)
        for value in REGULARIZATION
    }
    configurations = [
        (representation, family, size, c_value)
        for representation in REPRESENTATIONS
        for family, sizes in (("centroid_scalar", (1,)), ("pca_interaction", PCS))
        for size in sizes
        for c_value in REGULARIZATION
    ]
    enhanced_scores = {
        config: np.full(len(data["votes"]), np.nan, dtype=np.float64)
        for config in configurations
    }
    for train, valid in grouped_folds(data):
        nuisance = fit_nuisance(data, train, findings)
        n_train = nuisance.design(
            data["finding"][train], data["margin"][train],
            data["nuisance_numeric"][train], data["nuisance_categorical"][train]
        )
        n_valid = nuisance.design(
            data["finding"][valid], data["margin"][valid],
            data["nuisance_numeric"][valid], data["nuisance_categorical"][valid]
        )
        for c_value in REGULARIZATION:
            base_scores[c_value][valid] = fit_logistic(
                n_train, data["votes"][train], c_value
            ).predict_proba(n_valid)[:, 1]
        for representation in REPRESENTATIONS:
            visual = data["representations"][representation]
            transforms: dict[str, CentroidTransform | PCATransform] = {
                "centroid_scalar": fit_centroid(
                    data["finding"][train], data["votes"][train], data["margin"][train],
                    visual[train], findings
                ),
                "pca_interaction": fit_pca(
                    data["finding"][train], data["margin"][train], visual[train], findings
                ),
            }
            for family, sizes in (("centroid_scalar", (1,)), ("pca_interaction", PCS)):
                transform = transforms[family]
                for size in sizes:
                    v_train = visual_block(
                        transform, family, data["finding"][train], data["margin"][train],
                        visual[train], size
                    )
                    v_valid = visual_block(
                        transform, family, data["finding"][valid], data["margin"][valid],
                        visual[valid], size
                    )
                    e_train, e_valid = np.concatenate([n_train, v_train], 1), np.concatenate([n_valid, v_valid], 1)
                    for c_value in REGULARIZATION:
                        enhanced_scores[(representation, family, size, c_value)][valid] = fit_logistic(
                            e_train, data["votes"][train], c_value
                        ).predict_proba(e_valid)[:, 1]
    base_nll = {
        str(c): float(reader_nll(data["votes"], probability).mean())
        for c, probability in base_scores.items()
    }
    enhanced_nll = {
        f"representation={r};family={f};size={s};C={c}": float(
            reader_nll(data["votes"], enhanced_scores[(r, f, s, c)]).mean()
        )
        for r, f, s, c in configurations
    }
    selected = min(
        configurations,
        key=lambda value: enhanced_nll[
            f"representation={value[0]};family={value[1]};size={value[2]};C={value[3]}"
        ],
    )
    return {
        "base_C": float(min(REGULARIZATION, key=lambda value: base_nll[str(value)])),
        "base_cv_log_loss": base_nll,
        "enhanced": {
            "representation": selected[0],
            "family": selected[1],
            "size": int(selected[2]),
            "C": float(selected[3]),
        },
        "enhanced_cv_log_loss": enhanced_nll,
    }


def selection_identity(
    model: str, dev: dict[str, Any], raw: dict[str, Any], findings: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "model": model,
        "findings": list(findings),
        "development_directory": dev["directory"],
        "development_config_sha256": dev["config_sha256"],
        "development_metadata_sha256": dev["metadata_sha256"],
        "development_hidden_sha256": dev["hidden_sha256"],
        "development_record_keys_sha256": dev["record_keys_sha256"],
        "development_labels_sha256": dev["labels_sha256"],
        "raw_directory": raw["directory"],
        "raw_config_sha256": raw["config_sha256"],
        "raw_features_sha256": raw["features_sha256"],
        "raw_metadata_sha256": raw["metadata_sha256"],
    }


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {args.output}")
    findings = require_findings(args.findings)
    holdout = load_holdout_contract(args.holdout_lock, args.exposure_audit)
    dev = artifact_rows(args.dev, args.model, findings, DEV_PER_CELL)
    raw = strict_raw(args.raw, args.model)
    bind_hidden_raw(dev, raw, "development")
    data = join_raw(dev, raw)
    selection = select_nested(data, findings)
    result = {
        "protocol": PROTOCOL,
        "mode": "development_selection",
        "status": "frozen",
        "analysis_contract": analysis_contract(),
        "holdout_contract": holdout,
        "identity": selection_identity(args.model, dev, raw, findings),
        "grouped_folds": FOLDS,
        "selection": selection,
        "confirmation_parameters": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "permutations": PERMUTATIONS,
        },
        "confirmation_opened": False,
    }
    atomic_json(args.output, result)
    return result


def fit_frozen(
    dev: dict[str, Any], test: dict[str, Any], findings: tuple[str, ...], selection: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, NuisanceTransform, Any, Any, np.ndarray, np.ndarray]:
    nuisance = fit_nuisance(dev, np.arange(len(dev["votes"])), findings)
    n_dev = nuisance.design(
        dev["finding"], dev["margin"], dev["nuisance_numeric"], dev["nuisance_categorical"]
    )
    n_test = nuisance.design(
        test["finding"], test["margin"], test["nuisance_numeric"], test["nuisance_categorical"]
    )
    base_model = fit_logistic(n_dev, dev["votes"], selection["base_C"])
    base_probability = base_model.predict_proba(n_test)[:, 1]
    chosen = selection["enhanced"]
    visual_dev = dev["representations"][chosen["representation"]]
    visual_test = test["representations"][chosen["representation"]]
    if chosen["family"] == "centroid_scalar":
        transform = fit_centroid(
            dev["finding"], dev["votes"], dev["margin"], visual_dev, findings
        )
    else:
        transform = fit_pca(dev["finding"], dev["margin"], visual_dev, findings)
    v_dev = visual_block(
        transform, chosen["family"], dev["finding"], dev["margin"],
        visual_dev, chosen["size"]
    )
    v_test = visual_block(
        transform, chosen["family"], test["finding"], test["margin"],
        visual_test, chosen["size"]
    )
    enhanced_model = fit_logistic(
        np.concatenate([n_dev, v_dev], axis=1), dev["votes"], chosen["C"]
    )
    enhanced_probability = enhanced_model.predict_proba(
        np.concatenate([n_test, v_test], axis=1)
    )[:, 1]
    return base_probability, enhanced_probability, nuisance, transform, enhanced_model, v_dev, v_test


def condition_design(
    base: Any, finding: np.ndarray, margin: np.ndarray
) -> np.ndarray:
    one_hot = base.one_hot(finding)
    indices = np.argmax(one_hot, axis=1)
    scaled = (margin - base.margin_mean[indices]) / base.margin_std[indices]
    return np.concatenate(
        [one_hot, one_hot * scaled[:, None], one_hot * scaled[:, None] ** 2, one_hot * scaled[:, None] ** 3],
        axis=1,
    )


def conditional_randomization(
    dev: dict[str, Any],
    test: dict[str, Any],
    nuisance: NuisanceTransform,
    enhanced_model: Any,
    v_dev: np.ndarray,
    v_test: np.ndarray,
    aligned_probability: np.ndarray,
    permutations: int,
) -> dict[str, Any]:
    nuisance_dev = nuisance.design(
        dev["finding"], dev["margin"],
        dev["nuisance_numeric"], dev["nuisance_categorical"]
    )
    nuisance_test = nuisance.design(
        test["finding"], test["margin"], test["nuisance_numeric"], test["nuisance_categorical"]
    )
    residualizer = Ridge(alpha=1.0, fit_intercept=False).fit(nuisance_dev, v_dev)
    predicted = residualizer.predict(nuisance_test)
    residual = v_test - predicted
    numeric = test["nuisance_numeric"].astype(np.float64)
    numeric = (numeric - numeric.mean(axis=0)) / np.maximum(numeric.std(axis=0), 1e-8)
    standardized_residual = (
        residual - residual.mean(axis=0)
    ) / np.maximum(residual.std(axis=0), 1e-8)
    residual_numeric_correlation = np.abs(
        standardized_residual.T @ numeric / max(len(numeric) - 1, 1)
    )
    categorical_diagnostics: dict[str, Any] = {}
    categorical = test["nuisance_categorical"]
    for column in range(categorical.shape[1]):
        groups = []
        for value in sorted(set(str(item) for item in categorical[:, column])):
            rows = np.flatnonzero(categorical[:, column].astype(str) == value)
            if len(rows) < 2:
                continue
            groups.append(
                {
                    "value": value,
                    "n": int(len(rows)),
                    "maximum_absolute_residual_mean": float(
                        np.abs(standardized_residual[rows].mean(axis=0)).max()
                    ),
                    "residual_std_p95": float(
                        np.quantile(standardized_residual[rows].std(axis=0), 0.95)
                    ),
                }
            )
        categorical_diagnostics[str(column)] = groups
    strata: list[np.ndarray] = []
    frozen_edges: dict[str, list[float]] = {}
    for finding in sorted(set(dev["finding"])):
        dev_values = dev["margin"][dev["finding"] == finding]
        edges = np.quantile(dev_values, [0.25, 0.5, 0.75])
        frozen_edges[str(finding)] = edges.tolist()
        test_rows = np.flatnonzero(test["finding"] == finding)
        bins = np.digitize(test["margin"][test_rows], edges, right=True)
        for value in range(4):
            rows = test_rows[bins == value]
            if len(rows) > 1:
                strata.append(rows)
    aligned_nll = float(reader_nll(test["votes"], aligned_probability).mean())
    rng = np.random.default_rng(20260811)
    permuted_nll = []
    for _ in range(permutations):
        shuffled = residual.copy()
        for rows in strata:
            shuffled[rows] = residual[rng.permutation(rows)]
        probability = enhanced_model.predict_proba(
            np.concatenate([nuisance_test, predicted + shuffled], axis=1)
        )[:, 1]
        permuted_nll.append(float(reader_nll(test["votes"], probability).mean()))
    values = np.asarray(permuted_nll)
    p_value = float((1 + np.sum(values <= aligned_nll)) / (permutations + 1))
    return {
        "method": "F+M+N-ridge-residual permutation within finding_x_dev-margin-quartile",
        "one_claim_per_image_required": True,
        "permutations": permutations,
        "plus_one_p_value": p_value,
        "aligned_log_loss": aligned_nll,
        "permuted_log_loss_mean": float(values.mean()),
        "aligned_advantage_ci95": np.quantile(values - aligned_nll, [0.025, 0.975]).tolist(),
        "dev_margin_edges": frozen_edges,
        "permutation_stratum_sizes": [int(len(rows)) for rows in strata],
        "residualizer": "multi-output ridge alpha=1 on the full frozen F+M+N design",
        "validity_scope": (
            "conditional residual-permutation control; plus-one p-value relies on "
            "adequate ridge conditional mean and residual exchangeability within strata"
        ),
        "residual_numeric_abs_correlation_max": float(
            residual_numeric_correlation.max()
        ),
        "residual_numeric_abs_correlation_p95": float(
            np.quantile(residual_numeric_correlation, 0.95)
        ),
        "residual_categorical_group_diagnostics": categorical_diagnostics,
    }


def paired_bootstrap(
    test: dict[str, Any], baseline: np.ndarray, enhanced: np.ndarray, draws: int
) -> dict[str, Any]:
    base_nll, enhanced_nll = reader_nll(test["votes"], baseline), reader_nll(test["votes"], enhanced)
    base_brier, enhanced_brier = brier(test["votes"], baseline), brier(test["votes"], enhanced)
    cells = [
        np.flatnonzero((test["finding"] == finding) & (test["votes"] == vote))
        for finding in FROZEN_FINDINGS
        for vote in range(4)
    ]
    if any(len(rows) != FRESH_PER_CELL for rows in cells):
        raise ValueError("stratified bootstrap requires exactly 19 rows in every frozen cell")
    rng = np.random.default_rng(20260811)
    nll_delta, brier_delta = [], []
    for _ in range(draws):
        rows = np.concatenate(
            [rng.choice(cell, FRESH_PER_CELL, replace=True) for cell in cells]
        )
        nll_delta.append(float(base_nll[rows].mean() - enhanced_nll[rows].mean()))
        brier_delta.append(float(base_brier[rows].mean() - enhanced_brier[rows].mean()))
    return {
        "draws": draws,
        "method": "paired stratified bootstrap; resample 19 rows within each of 28 fixed finding_x_vote cells",
        "log_loss_delta_ci95": np.quantile(nll_delta, [0.025, 0.975]).tolist(),
        "brier_delta_ci95": np.quantile(brier_delta, [0.025, 0.975]).tolist(),
    }


def write_or_validate_opened_marker(path: Path, contract: dict[str, Any]) -> str:
    payload = {
        "status": "opened_locked",
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("confirmation was already opened under a different contract")
        return "exact_contract_resume_after_interruption"
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return "first_open"


def run_confirm(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.output.with_suffix(".predictions.jsonl")
    if args.output.exists() or prediction_path.exists():
        raise FileExistsError("confirmation receipt/output already exists; refusing overwrite")
    if args.bootstrap_draws != BOOTSTRAP_DRAWS or args.permutations != PERMUTATIONS:
        raise ValueError("confirmation resampling parameters are frozen and cannot be changed")
    receipt = json.loads(args.selection.read_text(encoding="utf-8"))
    current_analysis = analysis_contract()
    holdout = load_holdout_contract(args.holdout_lock, args.exposure_audit)
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("status") != "frozen"
        or receipt.get("analysis_contract") != current_analysis
        or receipt.get("identity", {}).get("model") != args.model
        or receipt.get("holdout_contract") != holdout
        or receipt.get("confirmation_parameters")
        != {"bootstrap_draws": BOOTSTRAP_DRAWS, "permutations": PERMUTATIONS}
        or receipt.get("confirmation_opened") is not False
    ):
        raise ValueError("selection receipt is stale, mismatched, or already opened")
    findings = require_findings(args.findings)
    dev = artifact_rows(args.dev, args.model, findings, DEV_PER_CELL)
    raw = strict_raw(args.raw, args.model)
    bind_hidden_raw(dev, raw, "development")
    identity = selection_identity(args.model, dev, raw, findings)
    if identity != receipt["identity"]:
        raise ValueError("development/raw identity differs from frozen selection")
    # Recompute the deterministic dev-only argmin before opening confirmation.
    # This prevents a modified receipt from creating a new one-shot attempt.
    dev_data = join_raw(dev, raw)
    recomputed_selection = select_nested(dev_data, findings)
    if recomputed_selection != receipt["selection"]:
        raise ValueError("frozen selection is not the deterministic development argmin")
    confirmation_config_path = args.confirmation / "config.json"
    confirmation_summary_path = args.confirmation / "summary.json"
    confirmation_config = json.loads(confirmation_config_path.read_text(encoding="utf-8"))
    if confirmation_config.get("manifest_sha256") != holdout["manifest_sha256"]:
        raise ValueError("confirmation hidden artifact is not derived from the locked holdout")
    repository_root = Path(__file__).resolve().parents[2]
    opened_marker = (
        repository_root
        / "corrected_runs/evidence_addressability_gate_v2/one_shot_registry"
        / f"{PROTOCOL}-{args.model}-{holdout['manifest_sha256']}.json"
    )
    opened_marker.parent.mkdir(parents=True, exist_ok=True)
    open_contract = {
        "protocol": PROTOCOL,
        "analysis_contract_sha256": current_analysis["sha256"],
        "selection_receipt_sha256": sha256(args.selection),
        "holdout_lock_sha256": holdout["lock_sha256"],
        "exposure_audit_sha256": holdout["exposure_sha256"],
        "confirmation_directory": str(args.confirmation.resolve()),
        "confirmation_config_sha256": sha256(confirmation_config_path),
        "confirmation_summary_sha256": sha256(confirmation_summary_path),
        "raw_config_sha256": raw["config_sha256"],
        "output": str(args.output.resolve()),
        "predictions": str(prediction_path.resolve()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutations": PERMUTATIONS,
    }
    opened_mode = write_or_validate_opened_marker(opened_marker, open_contract)
    # This is the first line that opens reader labels from the prospective endpoint holdout.
    test = artifact_rows(args.confirmation, args.model, findings, FRESH_PER_CELL)
    bind_hidden_raw(test, raw, "confirmation")
    if set(dev["image"]) & set(test["image"]):
        raise ValueError("development/fresh-confirmation image overlap")
    if len(set(test["image"])) != len(test["image"]):
        raise ValueError("fresh confirmation must contain one claim per image")
    test_data = join_raw(test, raw)
    baseline, enhanced, nuisance, transform, model, v_dev, v_test = fit_frozen(
        dev_data, test_data, findings, receipt["selection"]
    )
    base_metrics, enhanced_metrics = metric_summary(test_data, baseline), metric_summary(test_data, enhanced)
    nll_delta = base_metrics["reader_nll"] - enhanced_metrics["reader_nll"]
    brier_delta = base_metrics["reader_support_brier"] - enhanced_metrics["reader_support_brier"]
    bootstrap = paired_bootstrap(test_data, baseline, enhanced, args.bootstrap_draws)
    permutation = conditional_randomization(
        dev_data, test_data, nuisance, model, v_dev, v_test, enhanced, args.permutations
    )
    positive_findings = sum(
        enhanced_metrics["by_finding"][finding]["nll"]
        < base_metrics["by_finding"][finding]["nll"]
        for finding in findings
    )
    model_pass = bool(
        nll_delta / base_metrics["reader_nll"] >= 0.05
        and bootstrap["log_loss_delta_ci95"][0] > 0
        and brier_delta / base_metrics["reader_support_brier"] >= 0.05
        and bootstrap["brier_delta_ci95"][0] > 0
        and positive_findings >= 5
        and permutation["plus_one_p_value"] <= 0.05
    )
    predictions = "".join(
        json.dumps(
            {
                "record_key": str(key), "image_id": str(image), "finding": str(finding),
                "positive_votes": int(vote), "base_probability": float(left),
                "enhanced_probability": float(right),
            },
            sort_keys=True,
        ) + "\n"
        for key, image, finding, vote, left, right in zip(
            test["record_key"], test["image"], test["finding"], test["votes"], baseline, enhanced
        )
    )
    temporary_predictions = prediction_path.with_suffix(
        prediction_path.suffix + f".{os.getpid()}.tmp"
    )
    temporary_predictions.write_text(predictions, encoding="utf-8")
    os.replace(temporary_predictions, prediction_path)
    result = {
        "protocol": PROTOCOL,
        "mode": "fresh_confirmation_once",
        "status": "complete",
        "claim_boundary": (
            "This tests frozen-backbone supervised incremental decodability only; "
            "it is not localization, causal control, correction, or mitigation. "
            "The holdout is prospective for this endpoint relative to the old direct-CE "
            "manifest, but is not image-unseen across the repository."
        ),
        "analysis_contract": current_analysis,
        "analysis_contract_sha256": current_analysis["sha256"],
        "selection_receipt": str(args.selection.resolve()),
        "selection_receipt_sha256": sha256(args.selection),
        "opened_marker": str(opened_marker.resolve()),
        "opened_marker_sha256": sha256(opened_marker),
        "opened_mode": opened_mode,
        "holdout_contract": holdout,
        "holdout_lock_sha256": holdout["lock_sha256"],
        "exposure_audit_sha256": holdout["exposure_sha256"],
        "model": args.model,
        "findings": list(findings),
        "development_record_keys_sha256": dev["record_keys_sha256"],
        "development_labels_sha256": dev["labels_sha256"],
        "confirmation_record_keys_sha256": test["record_keys_sha256"],
        "confirmation_labels_sha256": test["labels_sha256"],
        "confirmation_unique_images": len(set(test["image"])),
        "base_model": "finding + final_margin + frozen simple image statistics",
        "enhanced_model": "base model + selected raw visual representation",
        "development_selection": receipt["selection"],
        "confirmation": {
            "base": base_metrics,
            "enhanced": enhanced_metrics,
            "per_reader_log_loss_delta": nll_delta,
            "per_reader_log_loss_relative_improvement": nll_delta / base_metrics["reader_nll"],
            "reader_support_brier_delta": brier_delta,
            "reader_support_brier_relative_improvement": brier_delta / base_metrics["reader_support_brier"],
            "positive_nll_findings": int(positive_findings),
            "bootstrap": bootstrap,
            "conditional_randomization": permutation,
        },
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": sha256(prediction_path),
        "model_gate": {
            "model_pass": model_pass,
            "requires_joint_two_model_authorizer": True,
        },
    }
    atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    select = subparsers.add_parser("select")
    confirm = subparsers.add_parser("confirm")
    for subparser in (select, confirm):
        subparser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
        subparser.add_argument("--dev", type=Path, required=True)
        subparser.add_argument("--raw", type=Path, required=True)
        subparser.add_argument("--findings", nargs="+", required=True)
        subparser.add_argument("--holdout-lock", type=Path, required=True)
        subparser.add_argument("--exposure-audit", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
    confirm.add_argument("--confirmation", type=Path, required=True)
    confirm.add_argument("--selection", type=Path, required=True)
    confirm.add_argument("--bootstrap-draws", type=int, default=5000)
    confirm.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    result = run_select(args) if args.mode == "select" else run_confirm(args)
    compact = {
        "protocol": result["protocol"],
        "mode": result["mode"],
        "status": result["status"],
    }
    if args.mode == "confirm":
        compact["model_gate"] = result["model_gate"]
        compact["confirmation"] = {
            key: value
            for key, value in result["confirmation"].items()
            if key not in {"base", "enhanced"}
        }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
