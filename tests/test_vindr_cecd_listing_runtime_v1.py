import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image
import anchor.corrected_sgta.run_vindr_cecd_listing_runtime_v1 as listing_runtime

from anchor.corrected_sgta.run_vindr_cecd_listing_runtime_v1 import (
    ADMISSION_VERSION,
    MANIFEST_VERSION,
    NONE_TOKEN,
    ONTOLOGY,
    PACK_VERSION,
    FakeListingAdapter,
    RuntimeContractError,
    canonical_json_sha256,
    evaluate_run,
    evaluate_outputs,
    parse_listing,
    run_runtime,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim(finding_id: str, votes: int) -> dict:
    return {
        "finding_id": finding_id,
        "positive_votes": votes,
        "reader_support": votes / 3.0,
    }


def test_real_model_cannot_select_a_noncanonical_gpu_lock(tmp_path: Path) -> None:
    with pytest.raises(RuntimeContractError, match="GPU lock drift"):
        listing_runtime._require_canonical_gpu_lock("huatuo", tmp_path / "other.lock")
    # CPU fake fixtures remain injectable and never claim a scientific model.
    listing_runtime._require_canonical_gpu_lock("fake", tmp_path / "other.lock")


@pytest.fixture(autouse=True)
def _legacy_runtime_fixture_uses_a_stubbed_strict_gate(monkeypatch):
    """Old runtime mechanics fixtures are not fake human adjudications."""

    def validated(**kwargs):
        return {"receipt": json.loads(Path(kwargs["receipt_path"]).read_text())}

    monkeypatch.setattr(listing_runtime, "validate_scientific_admission", validated)


def _fixture(tmp_path: Path):
    prompt_text = {
        "inspect_and_list": "P0 exact ontology comma grammar",
        "which_are_visible": "P1 exact ontology comma grammar",
        "report_all_from_ontology": "P2 exact ontology comma grammar",
        "inspect_and_list_exact_duplicate": "P0 exact ontology comma grammar",
    }
    cells = []
    renders = (
        "baseline_percentile",
        "native_linear",
        "center_minus_0p05w",
        "center_plus_0p05w",
        "width_x1p25",
    )
    prompts = ("inspect_and_list", "which_are_visible", "report_all_from_ontology")
    for render in renders:
        for prompt in prompts:
            cells.append(
                {
                    "cell_id": f"science__{render}__{prompt}",
                    "render_id": render,
                    "prompt_id": prompt,
                    "prompt_text": prompt_text[prompt],
                    "role": "science_factorial",
                }
            )
    for prompt in prompts:
        cells.append(
            {
                "cell_id": f"control_identity_image__{prompt}",
                "render_id": "identity_lossless_duplicate",
                "prompt_id": prompt,
                "prompt_text": prompt_text[prompt],
                "role": "identity_image_control",
            }
        )
    cells.append(
        {
            "cell_id": "control_duplicate_prompt__inspect_and_list_exact_duplicate",
            "render_id": "baseline_percentile",
            "prompt_id": "inspect_and_list_exact_duplicate",
            "prompt_text": prompt_text["inspect_and_list_exact_duplicate"],
            "role": "exact_duplicate_prompt_control",
        }
    )
    references = [
        {
            "image_id": "good",
            "experiment_split": "pilot",
            "inverse_sampling_weight": 1.0,
            "claims": [_claim(fid, 3 if fid == "cardiomegaly" else 0) for fid, _ in ONTOLOGY],
        },
        {
            "image_id": "guard-bad",
            "experiment_split": "pilot",
            "inverse_sampling_weight": 1.0,
            "claims": [_claim(fid, 0) for fid, _ in ONTOLOGY],
        },
    ]
    reference = tmp_path / "reference.jsonl"
    reference.write_text("".join(json.dumps(row) + "\n" for row in references))
    experiment = {
        "schema_version": MANIFEST_VERSION,
        "reference_contract": {"reference_file_sha256": _sha(reference)},
        "orbit_contract": {"cells": cells},
        "source": {"image_root": "/unused", "bbox": {"path": "/unused"}},
    }
    experiment_path = tmp_path / "experiment.json"
    _write_json(experiment_path, experiment)
    failure_ids = ["pair-bad"]
    pack_dir = tmp_path / "pack"
    _write_json(
        pack_dir / "sealed_mapping.json",
        {
            "clinical_pairs": [
                {
                    "pair_id": "pair-bad",
                    "image_id": "guard-bad",
                    "transform": "center_plus_0p05w",
                    "transform_guard": {"clinical_guard_pass": False},
                }
            ]
        },
    )
    pack = {
        "version": PACK_VERSION,
        "clinical_review": {
            "computational_guard_failure_pair_ids_sha256": canonical_json_sha256(failure_ids)
        },
        "artifact_sha256": {"sealed_mapping.json": _sha(pack_dir / "sealed_mapping.json")},
    }
    _write_json(pack_dir / "manifest.json", pack)
    admission = {
        "schema_version": ADMISSION_VERSION,
        "status": "independently_admitted_for_model_scoring",
        "four_independent_human_returns_validated": True,
        "listing_render_equivalence_admitted": True,
        "listing_prompt_equivalence_admitted": True,
        "adjudication_complete": True,
        "upstream_binary_ce_gate_authorized": True,
        "upstream_binary_ce_authorization_sha256": "1" * 64,
        "model_scoring_authorized": True,
        "gpu_authorized": True,
        "model_outputs_read_for_admission": False,
        "authorized_model_ids": ["huatuo", "hulu"],
        "pack_manifest_sha256": _sha(pack_dir / "manifest.json"),
        "experiment_manifest_sha256": _sha(experiment_path),
        "reference_file_sha256": _sha(reference),
        "computational_guard_failure_pair_ids_sha256": canonical_json_sha256(failure_ids),
    }
    admission_path = tmp_path / "admission.json"
    _write_json(admission_path, admission)
    return experiment_path, pack_dir, reference, admission_path, references


def test_admission_absence_and_hash_mismatch_fail_before_adapter_factory(tmp_path: Path) -> None:
    experiment, pack, reference, admission, _ = _fixture(tmp_path)
    calls = []
    lock_calls = []

    def factory(_: str):
        calls.append("loaded")
        return FakeListingAdapter()

    def lock_factory(path):
        lock_calls.append(path)
        raise AssertionError("lock must not be attempted before admission")

    with pytest.raises(RuntimeContractError, match="absent"):
        run_runtime(
            experiment_manifest_path=experiment,
            pack_dir=pack,
            admission_path=tmp_path / "missing.json",
            expected_admission_sha256="2" * 64,
            reference_path=reference,
            output_dir=tmp_path / "out-a",
            model_id="huatuo",
            split="pilot",
            adapter_factory=factory,
            lock_factory=lock_factory,
        )
    with pytest.raises(RuntimeContractError, match="pinned hash"):
        run_runtime(
            experiment_manifest_path=experiment,
            pack_dir=pack,
            admission_path=admission,
            expected_admission_sha256="2" * 64,
            reference_path=reference,
            output_dir=tmp_path / "out-b",
            model_id="huatuo",
            split="pilot",
            adapter_factory=factory,
            lock_factory=lock_factory,
        )
    assert calls == []
    assert lock_calls == []
    assert not (tmp_path / "out-a").exists()
    assert not (tmp_path / "out-b").exists()


def test_strict_parser_preserves_format_and_out_of_ontology() -> None:
    good = parse_listing("Cardiomegaly, Pleural effusion")
    assert good.valid and good.finding_ids == ("cardiomegaly", "pleural_effusion")
    assert parse_listing(NONE_TOKEN).valid
    duplicate = parse_listing("Cardiomegaly, Cardiomegaly")
    assert not duplicate.valid and duplicate.duplicate_finding_ids == ("cardiomegaly",)
    outside = parse_listing("Cardiomegaly, pneumonia likely")
    assert not outside.valid
    assert outside.out_of_ontology == ("pneumonia likely",)
    assert outside.raw_text == "Cardiomegaly, pneumonia likely"
    mixed = parse_listing(f"{NONE_TOKEN}, Cardiomegaly")
    assert not mixed.valid and "mixed_empty_set_token" in mixed.violations
    hedge = parse_listing("Possibly Cardiomegaly")
    assert hedge.hedge and not hedge.valid


def test_fake_end_to_end_atomic_resume_and_guard_exclusion(tmp_path: Path) -> None:
    experiment, pack, reference, admission, _ = _fixture(tmp_path)
    adapters = []

    def factory(_: str):
        adapter = FakeListingAdapter({("good", "inspect_and_list"): "Cardiomegaly"})
        adapters.append(adapter)
        return adapter

    def render(row, cell):
        image = Image.new("L", (4, 4))
        return image

    output = tmp_path / "run"
    first = run_runtime(
        experiment_manifest_path=experiment,
        pack_dir=pack,
        admission_path=admission,
        expected_admission_sha256=_sha(admission),
        reference_path=reference,
        output_dir=output,
        model_id="fake",
        split="pilot",
        adapter_factory=factory,
        render_provider=render,
    )
    assert first["cell_shards"] == 19
    assert first["excluded_guard_invalid_images"] == 1
    assert first["guard_invalid_images_entering_complete_orbit"] == 0
    assert len(adapters[0].calls) == 19
    assert not (output / "cell_shards" / "guard-bad").exists()
    second = run_runtime(
        experiment_manifest_path=experiment,
        pack_dir=pack,
        admission_path=admission,
        expected_admission_sha256=_sha(admission),
        reference_path=reference,
        output_dir=output,
        model_id="fake",
        split="pilot",
        adapter_factory=factory,
        render_provider=render,
    )
    assert second["invocation"] == {"new_shards": 0, "resumed_shards": 19}
    assert len(adapters) == 1
    evaluation = evaluate_run(
        experiment_manifest_path=experiment,
        reference_path=reference,
        run_dir=output,
        output_dir=tmp_path / "evaluation",
    )
    assert set(evaluation["outputs"]) == {"fixed_k", "matched_coverage"}
    assert (tmp_path / "evaluation" / "fixed_k.json").is_file()
    shard = next((output / "cell_shards" / "good").glob("*.json"))
    shard.write_text("{}\n")
    with pytest.raises(RuntimeContractError, match="invalid existing shard"):
        run_runtime(
            experiment_manifest_path=experiment,
            pack_dir=pack,
            admission_path=admission,
            expected_admission_sha256=_sha(admission),
            reference_path=reference,
            output_dir=output,
            model_id="fake",
            split="pilot",
            adapter_factory=factory,
            render_provider=render,
        )


def test_scientific_adapter_is_constructed_and_held_inside_shared_lock(tmp_path: Path) -> None:
    experiment, pack, reference, admission, _ = _fixture(tmp_path)
    state = {"locked": False, "factory": 0, "generated": 0}

    @contextmanager
    def lock_factory(_):
        assert not state["locked"]
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    class Adapter(FakeListingAdapter):
        model_id = "huatuo"

        def generate(self, image, prompt, max_new_tokens, seed):
            assert state["locked"]
            state["generated"] += 1
            return super().generate(image, prompt, max_new_tokens, seed)

        def close(self):
            assert state["locked"]

    def factory(model_id):
        assert model_id == "huatuo" and state["locked"]
        state["factory"] += 1
        return Adapter()

    run_runtime(
        experiment_manifest_path=experiment,
        pack_dir=pack,
        admission_path=admission,
        expected_admission_sha256=_sha(admission),
        reference_path=reference,
        output_dir=tmp_path / "scientific-run",
        model_id="huatuo",
        split="pilot",
        adapter_factory=factory,
        render_provider=lambda row, cell: Image.new("L", (2, 2)),
        lock_factory=lock_factory,
    )
    assert state == {"locked": False, "factory": 1, "generated": 19}


def _science_shards(image_id: str, labels: dict[tuple[str, str], list[str]]) -> list[dict]:
    rows = []
    for render in (
        "baseline_percentile", "native_linear", "center_minus_0p05w",
        "center_plus_0p05w", "width_x1p25",
    ):
        for prompt in ("inspect_and_list", "which_are_visible", "report_all_from_ontology"):
            finding_ids = labels.get((render, prompt), [])
            raw = NONE_TOKEN if not finding_ids else ", ".join(dict(ONTOLOGY)[finding] for finding in finding_ids)
            rows.append(
                {
                    "image_id": image_id,
                    "render_id": render,
                    "prompt_id": prompt,
                    "parsed": parse_listing(raw).as_dict(),
                }
            )
    return rows


def test_fixed_k_and_matched_coverage_preserve_claim_budget_and_joint_metrics() -> None:
    refs = [
        {
            "image_id": "a", "inverse_sampling_weight": 2.0,
            "claims": [_claim(fid, 3 if fid == "cardiomegaly" else 0) for fid, _ in ONTOLOGY],
        },
        {
            "image_id": "b", "inverse_sampling_weight": 1.0,
            "claims": [_claim(fid, 2 if fid == "pleural_effusion" else 0) for fid, _ in ONTOLOGY],
        },
    ]
    labels_a = {(render, prompt): ["cardiomegaly"] for render in (
        "baseline_percentile", "native_linear", "center_minus_0p05w", "center_plus_0p05w", "width_x1p25"
    ) for prompt in ("inspect_and_list", "which_are_visible", "report_all_from_ontology")}
    labels_b = {(render, prompt): ["pleural_effusion"] for render in (
        "baseline_percentile", "native_linear", "center_minus_0p05w", "center_plus_0p05w", "width_x1p25"
    ) for prompt in ("inspect_and_list", "which_are_visible", "report_all_from_ontology")}
    shards = _science_shards("a", labels_a) + _science_shards("b", labels_b)
    fixed = evaluate_outputs(reference_rows=refs, shard_rows=shards, mode="fixed_k")
    matched = evaluate_outputs(reference_rows=refs, shard_rows=shards, mode="matched_coverage")
    assert fixed["intention_to_evaluate_science_orbits"] == 2
    assert fixed["canonical_claim_budget"] == 2
    assert all(row["selected_claims"] == 2 for row in fixed["methods"].values())
    assert all(row["selected_claims"] == 2 for row in matched["methods"].values())
    assert fixed["methods"]["canonical"]["required_omission_rate"] == 0.0
    assert fixed["methods"]["canonical"]["disagreement_overcommitment_rate"] == 0.5
    assert fixed["all_cell_format_violation_rate"] == 0.0


def test_invalid_cell_stays_in_intention_to_evaluate_clinical_denominator() -> None:
    refs = [
        {
            "image_id": "a", "inverse_sampling_weight": 1.0,
            "claims": [_claim(fid, 3 if fid == "cardiomegaly" else 0) for fid, _ in ONTOLOGY],
        }
    ]
    labels = {(render, prompt): ["cardiomegaly"] for render in (
        "baseline_percentile", "native_linear", "center_minus_0p05w",
        "center_plus_0p05w", "width_x1p25",
    ) for prompt in ("inspect_and_list", "which_are_visible", "report_all_from_ontology")}
    shards = _science_shards("a", labels)
    # Preserve the recognized exact label but add an out-of-ontology segment.
    bad = next(
        row for row in shards
        if row["render_id"] == "native_linear" and row["prompt_id"] == "which_are_visible"
    )
    bad["parsed"] = parse_listing("Cardiomegaly, pneumonia likely").as_dict()
    result = evaluate_outputs(reference_rows=refs, shard_rows=shards, mode="fixed_k")
    assert result["intention_to_evaluate_science_orbits"] == 1
    assert result["missing_science_shard_orbits"] == 0
    assert result["orbits_with_any_parser_failure"] == 1
    assert result["all_cell_format_violation_rate"] == pytest.approx(1 / 15)
    assert result["all_cell_out_of_ontology_rate"] == pytest.approx(1 / 15)
    assert result["methods"]["canonical"]["images"] == 1
    assert result["methods"]["canonical"]["required_omission_rate"] == 0.0
