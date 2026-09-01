from corrected_sgta.analyze_draft_score_reranking import analyze


def test_draft_conditioned_analysis_detects_fixed_k_gain() -> None:
    records = []
    for image in range(8):
        for qid, truth, draft, centered in (
            (0, "No", "yes", 0.0),
            (1, "Yes", "no", 2.0),
            (2, "Yes", "yes", 1.0),
        ):
            records.append({
                "status": "ok",
                "question_id": image * 10 + qid,
                "image": f"image-{image}",
                "truth": truth,
                "draft": {"prediction": draft},
                "scores": {
                    "original_margin": centered,
                    "null_margin": 0.0,
                    "null_centered_margin": centered,
                },
            })
    result = analyze(records, draws=200, seed=5)
    assert result["metrics"]["null_centered_margin"]["tp"] == 16
    assert result["metrics"]["baseline"]["tp"] == 8
    assert result["screening_gate"]["passed"] is True
