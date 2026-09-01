from __future__ import annotations

import json

from corrected_sgta.capture_huatuo_specificity_native_v1 import (
    _capture_status,
    run_capture,
)
from tests.test_specificity_ratchet_visible_replay_v1 import _fixture


class FakeNativeCaptureAdapter:
    def __init__(self, *, text: str, visual_counts=None):
        self.text = text
        self.visual_counts = visual_counts or {"own.jpg": 576, "swap1.jpg": 576, "swap2.jpg": 576}
        self.generation_calls = 0

    def fingerprint(self):
        return {
            "model_family": "huatuogpt-vision-7b",
            "adapter_version": "fake-native-capture-v1",
        }

    def generate_native_identity(self, *, image_path, question, seed, max_new_tokens):
        self.generation_calls += 1
        return {
            "text": self.text,
            "direct_output_sequence_ids": [100, 101, 102, 103, 104, 99],
            "directly_captured_output_sequences": True,
            "terminal_special_token_ids": [99],
            "decode_contract": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "repetition_penalty": 1.2,
            },
            "hit_max_new_tokens": False,
        }

    def contextual_target_ids(self, *, target):
        return [100, 101, 102, 103, 104]

    def visual_token_count(self, *, image_path, question):
        return self.visual_counts[image_path.name]


def test_case_complete_capture_trims_only_terminal_special_and_resumes(tmp_path):
    manifest, metadata, _, image_root, _ = _fixture(tmp_path)
    target = "A left lesion is present."
    adapter = FakeNativeCaptureAdapter(text=target)
    output = tmp_path / "capture_run"
    result = run_capture(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=output,
        adapter=adapter,
    )
    assert result["status"] == "complete_passed"
    assert result["n_captured_cases"] == 1
    assert adapter.generation_calls == 1
    case = result["cases"][0]
    assert case["raw_output_sequence_token_ids"] == [100, 101, 102, 103, 104, 99]
    assert case["native_generation_token_ids"] == [100, 101, 102, 103, 104]
    assert case["identity_passed"] is True
    persisted = json.loads((output / "native_capture.json").read_text())
    assert "resumed_cases" not in persisted

    second = FakeNativeCaptureAdapter(text=target)
    resumed = run_capture(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=output,
        adapter=second,
    )
    assert resumed["resumed_cases"] == 1
    assert second.generation_calls == 0


def test_capture_records_text_or_visual_identity_failure_fail_closed(tmp_path):
    manifest, metadata, _, image_root, _ = _fixture(tmp_path)
    adapter = FakeNativeCaptureAdapter(
        text="A different answer.",
        visual_counts={"own.jpg": 576, "swap1.jpg": 576, "swap2.jpg": 288},
    )
    result = run_capture(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=tmp_path / "failed_capture",
        adapter=adapter,
    )
    assert result["status"] == "complete_with_identity_failures"
    assert result["n_identity_failures"] == 1
    assert result["cases"][0]["decoded_text_exact_frozen_match"] is False
    assert result["cases"][0]["visual_token_count_equal_across_own_swaps"] is False


def test_limit_writes_canary_that_cannot_pose_as_complete_capture(tmp_path):
    manifest, metadata, _, image_root, _ = _fixture(tmp_path)
    # One manifest case means a limit of one is complete, not a partial canary.
    result = run_capture(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=tmp_path / "one_case",
        adapter=FakeNativeCaptureAdapter(text="A left lesion is present."),
        limit_cases=1,
    )
    assert result["status"] == "complete_passed"
    assert (tmp_path / "one_case" / "native_capture.json").is_file()


def test_partial_canary_identity_failure_is_never_success_status():
    assert _capture_status(is_canary=True, failures=[]) == "canary_passed"
    assert _capture_status(is_canary=True, failures=["case-bad"]) == "canary_failed"
