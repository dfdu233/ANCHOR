from corrected_sgta.prepare_vindr_cecd_ontology_listing_v1 import (
    NONE_TOKEN,
    PANEL,
    TARGET_FINDINGS,
    build_image_reference,
    orbit_cells,
    prompt_texts,
)
from corrected_sgta.audit_vindr_cecd_ontology_listing_tracks_v1 import (
    summarize_balanced_v2,
    summarize_breadth,
)
from corrected_sgta.validate_vindr_cecd_ontology_listing_v1 import (
    parse_ontology_listing,
    validate_outputs,
)


def _vectors(*, positives: dict[str, tuple[int, int, int]], no_finding=(0, 0, 0)):
    label_names = [source for _, source in TARGET_FINDINGS] + ["Edema", "No finding"]
    readers = {}
    for index, rad_id in enumerate(PANEL):
        row = {name: 0 for name in label_names}
        row["No finding"] = no_finding[index]
        for name, values in positives.items():
            row[name] = values[index]
        readers[rad_id] = row
    return readers, label_names


def test_closed_ontology_has_fourteen_claims_and_relative_empty_token() -> None:
    assert len(TARGET_FINDINGS) == 14
    assert NONE_TOKEN not in {source for _, source in TARGET_FINDINGS}
    prompts = prompt_texts()
    assert len(prompts) == 3
    assert all(NONE_TOKEN in text for text in prompts.values())
    assert all("Allowed ontology" in text for text in prompts.values())


def test_orbit_is_exact_five_by_three_plus_four_controls() -> None:
    cells = orbit_cells()
    assert len(cells) == 19
    assert len({row["cell_id"] for row in cells}) == 19
    assert sum(row["role"] == "science_factorial" for row in cells) == 15
    assert sum(row["role"] == "identity_image_control" for row in cells) == 3
    assert sum(row["role"] == "exact_duplicate_prompt_control" for row in cells) == 1


def test_reader_reference_retains_required_optional_refuted_and_outside() -> None:
    vectors, names = _vectors(
        positives={
            "Aortic enlargement": (1, 1, 1),
            "Cardiomegaly": (1, 0, 0),
            "Pleural effusion": (1, 1, 1),
            "Edema": (1, 1, 1),
        }
    )
    row = build_image_reference("x", vectors, names)
    assert row["sampling_stratum"] == "multiple_unanimous_target_findings"
    assert row["required_finding_ids"] == ["aortic_enlargement", "pleural_effusion"]
    assert row["optional_finding_ids"] == ["cardiomegaly"]
    assert "pneumothorax" in row["refuted_finding_ids"]
    assert row["outside_target_ontology_reader_positive"] == [
        {"source_name": "Edema", "positive_votes": 3, "reader_support": 1.0}
    ]


def test_unanimous_no_finding_is_empty_set_not_fifteenth_claim() -> None:
    vectors, names = _vectors(positives={}, no_finding=(1, 1, 1))
    row = build_image_reference("normal", vectors, names)
    assert row["sampling_stratum"] == "unanimous_no_finding"
    assert len(row["claims"]) == 14
    assert row["required_finding_ids"] == []
    assert row["optional_finding_ids"] == []
    assert len(row["refuted_finding_ids"]) == 14


def test_strict_parser_never_silently_drops_out_of_ontology_text() -> None:
    valid = parse_ontology_listing("Cardiomegaly, Pleural effusion")
    assert valid == {
        "status": "valid",
        "finding_ids": ["cardiomegaly", "pleural_effusion"],
        "unknown_items": [],
        "duplicate_finding_ids": [],
        "empty_set": False,
    }
    assert parse_ontology_listing(NONE_TOKEN)["finding_ids"] == []
    prose = parse_ontology_listing("There is Cardiomegaly")
    assert prose["status"] == "format_violation"
    assert prose["unknown_items"] == ["There is Cardiomegaly"]
    outside = parse_ontology_listing("Cardiomegaly, Edema")
    assert outside["status"] == "format_violation"
    assert outside["finding_ids"] == ["cardiomegaly"]
    assert outside["unknown_items"] == ["Edema"]
    duplicate = parse_ontology_listing("ILD, ILD")
    assert duplicate["status"] == "format_violation"
    assert duplicate["duplicate_finding_ids"] == ["ild"]
    mixed_none = parse_ontology_listing(f"{NONE_TOKEN}, Cardiomegaly")
    assert mixed_none["status"] == "format_violation"


def test_future_output_validator_requires_two_models_and_complete_cells() -> None:
    rows = [
        {
            "model": model,
            "image_id": "x",
            "cell_id": cell,
            "status": "ok",
            "text": NONE_TOKEN,
        }
        for model in ("huatuo", "hulu")
        for cell in ("a", "b")
    ]
    audit = validate_outputs(rows, image_ids={"x"}, cell_ids={"a", "b"})
    assert audit["full_two_model_product_orbit"] is True
    assert audit["all_cells_strictly_parseable"] is True
    assert audit["efficacy_or_truth_comparison_performed"] is False


def test_two_track_auditor_distinguishes_balanced_and_breadth_claims() -> None:
    findings = [f"f{index}" for index in range(8)]
    v2_rows = []
    for image_id, split in (("a", "pilot"), ("b", "dev"), ("c", "confirmation")):
        for index, finding in enumerate(findings):
            positive = 3 if index < 2 else 0
            v2_rows.append(
                {
                    "image_id": image_id,
                    "experiment_split": split,
                    "finding": finding,
                    "positive_votes": positive,
                    "reader_count": 3,
                    "reader_panel": ["R8", "R9", "R10"],
                    "reference_relevance": "required" if positive == 3 else "out_of_scope",
                }
            )
    v2 = summarize_balanced_v2(
        {"eligible_findings": findings, "oe_claim_rows": len(v2_rows)}, v2_rows
    )
    assert v2["images"] == 3
    assert v2["true_multiclaim_images"] == 3
    assert v2["complete_fixed_claim_universe"] is True
    assert v2["patient_disjoint_verifiable"] is False

    breadth = summarize_breadth(
        {
            "task_contract": {
                "formal_task_type": "ontology_constrained_open_cardinality_listing",
                "free_form_oe": False,
                "target_finding_ids": [f"x{index}" for index in range(14)],
            },
            "reference_contract": {"reference_rows": 1},
            "source_census": {
                "fixed_panel_images": 5501,
                "images_with_at_least_two_unanimous_target_findings": 1360,
                "images_with_any_reader_positive_outside_target_ontology": 4191,
                "images_with_unanimous_positive_outside_target_ontology": 3620,
            },
        },
        [
            {
                "image_id": "z",
                "experiment_split": "dev",
                "sampling_stratum": "multiple_unanimous_target_findings",
                "required_finding_ids": ["x1", "x2"],
            }
        ],
    )
    assert breadth["finding_count"] == 14
    assert breadth["selected_true_multiclaim_images"] == 1
    assert breadth["free_oe_authorized"] is False
