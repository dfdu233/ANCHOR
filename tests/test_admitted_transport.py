from corrected_sgta.analyze_admitted_transport import analyze


def _metric(tp: int, fp: int, fn: int) -> dict:
    return {
        "n_claims": tp + fp + fn,
        "n_true_claims": tp + fn,
        "n_predicted_claims": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def _images(gain: bool = True) -> list[dict]:
    rows = []
    for index in range(12):
        baseline = _metric(1, 1, 1)
        candidate = _metric(2, 0, 0) if gain else _metric(0, 2, 2)
        rows.append({
            "image": f"image-{index}",
            "baseline": baseline,
            "original_margin": candidate,
            "null_centered_margin": candidate,
        })
    return rows


def test_paired_admission_gate_passes_shared_gain() -> None:
    models = {"a": _images(), "b": _images(), "c": _images()}
    branches = {"a": "original_margin", "b": "null_centered_margin", "c": "original_margin"}
    result = analyze(models, branches, draws=200, seed=3)
    assert result["holdout_gate"]["passed"] is True
    assert result["holdout_gate"]["nonnegative_models"] == 3


def test_paired_admission_gate_rejects_material_model_harm() -> None:
    models = {"a": _images(), "b": _images(), "c": _images(False)}
    branches = {name: "original_margin" for name in models}
    result = analyze(models, branches, draws=200, seed=4)
    assert result["holdout_gate"]["passed"] is False
    assert result["holdout_gate"]["no_material_drop"] is False


def test_rejects_misaligned_claim_universes() -> None:
    models = {"a": _images(), "b": _images(), "c": _images()}
    models["c"][0]["baseline"] = _metric(1, 1, 2)
    branches = {name: "original_margin" for name in models}
    try:
        analyze(models, branches, draws=20, seed=4)
    except ValueError as error:
        assert "claim universe differs" in str(error)
    else:
        raise AssertionError("misaligned claim universes must be rejected")
