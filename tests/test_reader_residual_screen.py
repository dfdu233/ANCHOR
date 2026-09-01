import json

import numpy as np
import pytest

from corrected_sgta.screen_reader_residual_v1 import (
    NuisanceTransform,
    ResidualTransform,
    admissible_pca_dimensions,
    grouped_splits,
    fit_predict_models,
    nested_oof,
    reader_targets,
    scoped_metrics,
    validate_inputs,
)


def balanced_problem(seed: int = 7):
    rng = np.random.default_rng(seed)
    votes = np.tile(np.repeat(np.arange(4), 15), 2)
    findings = np.repeat(np.asarray(["effusion", "nodule"], dtype=object), 60)
    target, stratum = reader_targets(votes)
    evidence = votes.astype(float) + rng.normal(scale=0.65, size=len(votes))
    features = rng.normal(size=(len(votes), 24))
    features[:, 0] += 1.5 * target
    maybe = -np.abs(evidence - 1.5) + rng.normal(scale=0.2, size=len(votes))
    groups = np.asarray([f"image-{index}" for index in range(len(votes))], dtype=object)
    return target, stratum, findings, evidence, features, maybe, groups


def test_reader_targets_are_unanimity_within_polarity_strata():
    target, stratum = reader_targets([0, 1, 2, 3])
    assert target.tolist() == [1, 0, 0, 1]
    assert stratum.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        reader_targets([4])


def test_pca_dimension_rule_uses_smallest_training_fold():
    assert admissible_pca_dimensions([1, 2, 4, 8, 16], [101, 80, 95], 100) == [1, 2, 4]
    assert admissible_pca_dimensions([1, 2, 4], [80], 2) == [1, 2]


def test_grouped_splits_are_image_disjoint_and_stratified():
    target, stratum, findings, _, _, _, groups = balanced_problem()
    splits = grouped_splits(target, stratum, findings, groups, 5, 42)
    assert len(splits) == 5
    observed = np.zeros(len(target), dtype=int)
    for train, test in splits:
        assert set(groups[train]).isdisjoint(groups[test])
        observed[test] += 1
        assert set(target[test]) == {0, 1}
    assert observed.tolist() == [1] * len(target)


def test_random_projection_and_train_only_transforms_are_reproducible():
    target, stratum, findings, evidence, features, _, _ = balanced_problem()
    train = np.arange(90)
    nuisance_fit = NuisanceTransform.fit(evidence[train], stratum[train], findings[train])
    nuisance = nuisance_fit.transform(evidence[train], stratum[train], findings[train])
    left = ResidualTransform.fit(features[train], nuisance, 4, "random", 19, 10.0)
    right = ResidualTransform.fit(features[train], nuisance, 4, "random", 19, 10.0)
    assert np.allclose(left.projector, right.projector)
    assert np.allclose(
        left.transform(features[train], nuisance), right.transform(features[train], nuisance)
    )


def test_nested_oof_is_complete_and_reports_both_polarity_strata():
    target, stratum, findings, evidence, features, maybe, groups = balanced_problem()
    predictions = nested_oof(
        features=features,
        target=target,
        evidence=evidence,
        maybe_margin=maybe,
        stratum=stratum,
        findings=findings,
        groups=groups,
        requested_k=[1, 2, 4, 8],
        outer_folds=5,
        inner_folds=3,
        seed=42,
        residual_alpha=10.0,
    )
    for name in ("baseline", "candidate", "random_control", "direct_maybe"):
        assert predictions[name].shape == (len(target),)
        assert np.isfinite(predictions[name]).all()
        assert ((predictions[name] >= 0) & (predictions[name] <= 1)).all()
    assert max(predictions["outer_selected_k"]) <= 4
    metrics = scoped_metrics(target, stratum, predictions["baseline"], predictions["candidate"])
    assert set(("negative_0v1", "positive_2v3")).issubset(metrics)
    assert np.isfinite(metrics["macro_delta_auc"])


def test_direct_maybe_control_predicts_unanimity_in_both_strata():
    rng = np.random.default_rng(12)
    votes = np.repeat(np.arange(4), 60)
    target, stratum = reader_targets(votes)
    evidence = rng.normal(size=len(votes))
    # High for both disagreement bins, hence directly predictive of lower
    # unanimity in both polarity strata.
    maybe = np.isin(votes, (1, 2)).astype(float) * 5 + rng.normal(scale=0.1, size=len(votes))
    findings = np.asarray(["finding"] * len(votes), dtype=object)
    train = np.concatenate([np.arange(value * 60, value * 60 + 45) for value in range(4)])
    test = np.concatenate([np.arange(value * 60 + 45, value * 60 + 60) for value in range(4)])
    baseline, augmented = fit_predict_models(
        train=train,
        test=test,
        target=target,
        evidence=evidence,
        stratum=stratum,
        findings=findings,
        features=None,
        k=None,
        mode="pca",
        seed=42,
        residual_alpha=10.0,
        maybe_margin=maybe,
    )
    assert np.mean((target[test] - augmented) ** 2) < np.mean((target[test] - baseline) ** 2)


def test_validate_inputs_rejects_non_dev_and_non_disjoint(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    (features / "config.json").write_text(json.dumps({"split": "pilot"}))
    (features / "summary.json").write_text(json.dumps({"status": "complete"}))
    sampling = tmp_path / "sampling.json"
    sampling.write_text(
        json.dumps({"split_contract": {"image_disjoint": True, "claim_rows": {"dev": 10}}})
    )
    with pytest.raises(ValueError, match="dev-only"):
        validate_inputs(features, sampling)

    (features / "config.json").write_text(json.dumps({"split": "dev"}))
    sampling.write_text(
        json.dumps({"split_contract": {"image_disjoint": False, "claim_rows": {"dev": 10}}})
    )
    with pytest.raises(ValueError, match="image-disjoint"):
        validate_inputs(features, sampling)
