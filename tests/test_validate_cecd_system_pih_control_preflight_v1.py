import copy
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.validate_cecd_system_pih_control_preflight_v1 import (
    ARCHITECTURES,
    CONTRACT_FINGERPRINTS,
    PreflightError,
    derive_query_head_width,
    dynamic_expanded_partition,
    positional_surrogate_spans,
    sha256_file,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/cecd_system_pih_control_preflight_v1.json"


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _ready_payload(tmp_path: Path) -> dict:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    for name in payload["implementation_bindings"]:
        source = tmp_path / f"{name}.py"
        source.write_text("# test-only independent common-protocol binding\n", encoding="utf-8")
        payload["implementation_bindings"][name] = _record(source)
    for family in ("huatuo", "hulu"):
        payload["runtime_integrations"][family]["status"] = "ready"
        for name in (
            "system_attention_runtime_patch",
            "native_eager_canary_artifact",
            "pih_o_proj_runtime_integration",
        ):
            source = tmp_path / f"{family}_{name}.json"
            source.write_text("{}\n", encoding="utf-8")
            payload["runtime_integrations"][family][name] = _record(source)
    for family in ("huatuo", "hulu"):
        payload["pih_selection"][family]["status"] = "ready"
        payload["pih_selection"][family]["selected_heads_artifact"] = (
            f"sha256:{family}-selected"
        )
        payload["pih_selection"][family]["random_heads_artifact"] = (
            f"sha256:{family}-random"
        )
    payload["control_output_root"] = str(tmp_path / "control_outputs")
    return payload


def test_current_contract_has_bound_components_but_selected_heads_remain_blocked() -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    result = validate_plan(payload, root=ROOT)
    assert result["passed"] is False
    assert result["paper_native_reproduction_authorized"] is False
    assert result["official_code_port_authorized"] is False
    assert result["true_system_attention_test_available_on_locked_prompts"] is False
    assert result["three_stage_thresholds_modified"] is False
    assert result["worker_or_listing_modified"] is False
    assert not any(
        blocker.startswith("implementation_not_bound:") for blocker in result["blockers"]
    )
    assert "huatuo:pih_selection_not_ready" in result["blockers"]
    assert "hulu:pih_selection_not_ready" in result["blockers"]
    assert "huatuo:runtime_integration_not_ready" in result["blockers"]
    assert "hulu:runtime_integration_not_ready" in result["blockers"]
    assert result["per_model_runtime_integration_ready"] is False


def test_ready_common_protocol_never_authorizes_paper_native(tmp_path: Path) -> None:
    result = validate_plan(_ready_payload(tmp_path), root=ROOT)
    assert result["passed"] is True
    assert result["control_execution_ready"] is True
    assert result["common_protocol_label_required"] is True
    assert result["paper_native_reproduction_authorized"] is False
    assert result["official_code_port_authorized"] is False


def test_dynamic_spans_accept_variable_visual_length_and_are_exhaustive() -> None:
    for visual_length in (4, 17, 576, 1024):
        labels = ["user_text"] * 3 + ["image"] * visual_length + ["user_text"] * 5
        spans = dynamic_expanded_partition(
            labels, image_start=3, image_end=3 + visual_length
        )
        assert len(spans["image"]) == visual_length
        flattened = set().union(*(set(value) for value in spans.values()))
        assert flattened == set(range(len(labels)))


def test_noncontiguous_or_magic_image_span_is_rejected() -> None:
    labels = ["user_text", "image", "user_text", "image"]
    with pytest.raises(PreflightError, match="contiguous dynamic span"):
        dynamic_expanded_partition(labels, image_start=1, image_end=2)
    with pytest.raises(PreflightError, match="invalid dynamic image span"):
        positional_surrogate_spans(20, image_start=35, image_end=611)


def test_pre_image_user_delimiter_cannot_be_role_labeled_as_system() -> None:
    labels = ["user_text", "user_text", "image", "image", "user_text"]
    role_spans = dynamic_expanded_partition(labels, image_start=2, image_end=4)
    positional = positional_surrogate_spans(5, image_start=2, image_end=4)
    assert role_spans["system"] == ()
    assert positional["prefix_before_image"] == (0, 1)
    assert set(positional["prefix_before_image"]) == set(role_spans["user_text"][:2])


def test_head_width_uses_o_proj_input_not_hidden_size() -> None:
    huatuo = ARCHITECTURES["huatuo"]
    hulu = ARCHITECTURES["hulu"]
    assert derive_query_head_width(
        o_proj_in_features=huatuo["attention_output_width"],
        num_query_heads=huatuo["num_query_heads"],
    ) == 128
    assert derive_query_head_width(
        o_proj_in_features=hulu["attention_output_width"],
        num_query_heads=hulu["num_query_heads"],
    ) == 128
    assert hulu["hidden_size"] // hulu["num_query_heads"] == 80
    assert hulu["attention_output_width"] != hulu["hidden_size"]


def test_cross_model_head_set_reuse_is_blocked(tmp_path: Path) -> None:
    payload = _ready_payload(tmp_path)
    payload["pih_selection"]["hulu"]["selected_heads_artifact"] = payload[
        "pih_selection"
    ]["huatuo"]["selected_heads_artifact"]
    result = validate_plan(payload, root=ROOT)
    assert result["passed"] is False
    assert "pih_head_set_reused_across_models" in result["blockers"]


def test_test_split_selection_and_wrong_hulu_width_are_blocked(tmp_path: Path) -> None:
    payload = _ready_payload(tmp_path)
    payload["pih_selection"]["hulu"]["selection_split"] = "test"
    payload["pih_selection"]["hulu"]["locked_test_scanned"] = True
    payload["pih_selection"]["hulu"]["head_width"] = 80
    result = validate_plan(payload, root=ROOT)
    assert result["passed"] is False
    assert "hulu:selection_not_dev_only" in result["blockers"]
    assert "hulu:locked_test_scanned" in result["blockers"]
    assert "hulu:wrong_query_head_width" in result["blockers"]


def test_contract_fingerprint_drift_fails_closed() -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["contract_fingerprints"] = copy.deepcopy(CONTRACT_FINGERPRINTS)
    payload["contract_fingerprints"]["papers"] = "0" * 64
    with pytest.raises(PreflightError, match="contract drift"):
        validate_plan(payload, root=ROOT)
