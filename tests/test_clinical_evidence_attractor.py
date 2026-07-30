import json
from pathlib import Path

from anchor.corrected_sgta.analyze_clinical_evidence_attractor import (
    analyze_files,
)


def write_probe(path: Path, contraction: float) -> None:
    rows = []
    for case_index in range(4):
        real = float(case_index)
        for view, value in (
            ("real", real),
            ("null", 0.0),
            ("style_0", (1 - contraction) * real + contraction * 1.5),
        ):
            for disease_index, disease in enumerate(("a", "b")):
                evidence = value + disease_index
                for polarity, nll in (
                    ("positive", 1.0 - evidence / 2),
                    ("negative", 1.0 + evidence / 2),
                ):
                    rows.append(
                        {
                            "case_id": f"case-{case_index}",
                            "image_relative": (
                                f"p00/patient-{case_index}/study/image.jpg"
                            ),
                            "view": view,
                            "disease": disease,
                            "polarity": polarity,
                            "sequence_nll": nll,
                            "model": path.stem,
                        }
                    )
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_evidence_analysis_detects_contraction(tmp_path: Path) -> None:
    huatuo = tmp_path / "huatuo.jsonl"
    base = tmp_path / "base.jsonl"
    write_probe(huatuo, contraction=0.5)
    write_probe(base, contraction=0.2)
    result = analyze_files(huatuo, base, repeats=20)
    assert (
        result["models"]["huatuo"][
            "mean_log_squared_centroid_distance_ratio"
        ]
        < 0
    )
    assert result["decision"]["centroid_projection_exceeds_null_both_models"]
