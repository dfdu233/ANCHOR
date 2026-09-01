import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    ContractError,
    TeacherForcedTrace,
)
from corrected_sgta.specificity_ratchet_visible_replay_v1 import (
    CAPTURE_PROTOCOL_ID,
    compute_replay_signals,
    run_runtime,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FakeReplayAdapter:
    def __init__(self, boost_constraint=True):
        self.calls = []
        self.boost_constraint = boost_constraint

    def fingerprint(self):
        return {
            "model_family": "huatuogpt-vision-7b",
            "adapter_version": "fake-replay-v1",
        }

    def score(self, *, image_path, question, target, condition):
        self.calls.append((None if image_path is None else image_path.name, condition))
        matches = list(re.finditer(r"\S+", target))
        ids = [100 + index for index in range(len(matches))]
        offsets = [(match.start(), match.end()) for match in matches]
        values = np.full((3, len(ids)), -2.0)
        if self.boost_constraint and image_path is not None and image_path.name == "own.jpg":
            # "left" is token one. Its own-image commitment rises late.
            values[:, 1] += np.asarray([0.0, 0.5, 1.0])
        image_hash = _sha(image_path.read_bytes()) if image_path is not None else None
        serialized = _sha(
            json.dumps(
                {"image": image_hash, "condition": condition, "target": target},
                sort_keys=True,
            ).encode()
        )
        return TeacherForcedTrace(
            condition=condition,
            target=target,
            token_ids=ids,
            token_offsets=offsets,
            offset_unit="unicode_character",
            layer_ids=["l1", "l2", "l3"],
            layer_gold_logp=values.tolist(),
            serialized_input_sha256=serialized,
            prompt_sha256=_sha(question.encode()),
            target_sha256=_sha(target.encode()),
            image_sha256=image_hash,
            template_id="fake-native-template",
            contextual_offsets_certified=True,
        )


def _span(target, text):
    start = target.index(text)
    return {
        "char_start": start,
        "char_end_exclusive": start + len(text),
        "text": text,
        "utf8_sha256": _sha(text.encode()),
    }


def _fixture(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("own.jpg", "swap1.jpg", "swap2.jpg"):
        (image_root / name).write_bytes(name.encode())
    target = "A left lesion is present."
    rows = []
    for index in range(2):
        rows.append(
            {
                "manifest_protocol_id": "specificity-ratchet-visible-replay-v1",
                "sample_id": f"sample-{index}",
                "case_id": "case-own",
                "edge_id": f"edge-{index}",
                "source_question_id": "qid-own",
                "target_model_family": "huatuogpt-vision-7b",
                "image_relpath": "own.jpg",
                "question": "What is present?",
                "full_visible_answer": target,
                "full_visible_answer_sha256": _sha(target.encode()),
                "native_generation_ids_certified": False,
                "child_char_span_in_full_answer": _span(target, target),
                "constraint_char_spans_in_full_answer": [_span(target, "left")],
                "model_input_contract": "complete frozen visible OE answer only",
                "edge_type": "laterality" if index == 0 else "size_morph",
                "modality_stratum": "XR",
                "anatomy_stratum": "thorax",
                "prompt_requested_increment": False,
                "scientific_role": "supported_specificity_control" if index == 0 else "causal_escalation_error",
                "split": "dev",
                "matched_image_swaps": [
                    {"case_id": "swap-1", "image_relpath": "swap1.jpg"},
                    {"case_id": "swap-2", "image_relpath": "swap2.jpg"},
                ],
            }
        )
    manifest = tmp_path / "samples.jsonl"
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    manifest.write_bytes(payload)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest_protocol_id": "specificity-ratchet-visible-replay-v1",
                "status": "physician_admitted_visible_answer_replay",
                "target_model_family": "huatuogpt-vision-7b",
                "native_capture_required_before_scientific_runtime": True,
                "manifest_sha256": _sha(payload),
                "n_scientific_edges": 2,
            }
        )
    )
    adapter = FakeReplayAdapter()
    native_ids = [100, 101, 102, 103, 104]
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "capture_protocol_id": CAPTURE_PROTOCOL_ID,
                "status": "complete_passed",
                "manifest_sha256": _sha(payload),
                "metadata_sha256": _sha(metadata.read_bytes()),
                "target_model_family": "huatuogpt-vision-7b",
                "adapter_fingerprint": adapter.fingerprint(),
                "cases": [
                    {
                        "case_id": "case-own",
                        "source_question_id": "qid-own",
                        "frozen_visible_answer_sha256": _sha(target.encode()),
                        "directly_captured_output_sequences": True,
                        "decoded_text_exact_frozen_match": True,
                        "native_ids_equal_contextual_target_ids": True,
                        "visual_token_counts_own_swap1_swap2": [576, 576, 576],
                        "visual_token_count_equal_across_own_swaps": True,
                        "native_generation_token_ids": native_ids,
                        "native_generation_token_ids_sha256": _sha(
                            json.dumps(native_ids, separators=(",", ":")).encode()
                        ),
                    }
                ],
            }
        )
    )
    return manifest, metadata, capture, image_root, adapter


def test_runtime_scores_each_case_once_and_resumes(tmp_path):
    manifest, metadata, capture, image_root, adapter = _fixture(tmp_path)
    output = tmp_path / "run"
    result = run_runtime(
        manifest=manifest,
        metadata=metadata,
        native_capture=capture,
        image_root=image_root,
        output_dir=output,
        adapter=adapter,
    )
    assert result["rows"] == 2
    assert result["cases"] == 1
    assert result["scored_cases_this_invocation"] == 1
    assert adapter.calls == [
        ("own.jpg", "image"),
        ("swap1.jpg", "image"),
        ("swap2.jpg", "image"),
        (None, "text_only"),
    ]
    shards = sorted((output / "shards").glob("*.json"))
    signal = json.loads(shards[0].read_text())["payload"]["signals"]
    assert signal["primary_own_minus_matched_swaps"][
        "constraint_minus_matched_difference_in_differences"
    ] == [0.0, 0.5, 1.0]
    second = FakeReplayAdapter()
    resumed = run_runtime(
        manifest=manifest,
        metadata=metadata,
        native_capture=capture,
        image_root=image_root,
        output_dir=output,
        adapter=second,
    )
    assert resumed["resumed_rows"] == 2
    assert second.calls == []


def test_identical_own_swap_and_text_traces_have_zero_residual(tmp_path):
    manifest, metadata, capture, image_root, _ = _fixture(tmp_path)
    adapter = FakeReplayAdapter(boost_constraint=False)
    # Capture fingerprint is structurally identical despite a different fake object.
    result = run_runtime(
        manifest=manifest,
        metadata=metadata,
        native_capture=capture,
        image_root=image_root,
        output_dir=tmp_path / "zero",
        adapter=adapter,
    )
    assert result["status"] == "complete"
    shard = next((tmp_path / "zero" / "shards").glob("*.json"))
    signals = json.loads(shard.read_text())["payload"]["signals"]
    assert signals["primary_own_minus_matched_swaps"][
        "constraint_minus_matched_difference_in_differences"
    ] == [0.0, 0.0, 0.0]
    assert signals["text_only_secondary"]["difference_in_differences"] == [0.0, 0.0, 0.0]


def test_missing_or_nonidentical_native_capture_refuses_before_scoring(tmp_path):
    manifest, metadata, capture, image_root, adapter = _fixture(tmp_path)
    payload = json.loads(capture.read_text())
    payload["cases"][0]["native_ids_equal_contextual_target_ids"] = False
    capture.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="native/contextual token IDs differ"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            native_capture=capture,
            image_root=image_root,
            output_dir=tmp_path / "refused",
            adapter=adapter,
        )


def test_native_capture_is_bound_to_exact_replay_metadata(tmp_path):
    manifest, metadata, capture, image_root, adapter = _fixture(tmp_path)
    payload = json.loads(metadata.read_text())
    payload["post_capture_drift"] = True
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="different replay metadata"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            native_capture=capture,
            image_root=image_root,
            output_dir=tmp_path / "must_not_score",
            adapter=adapter,
        )
    assert adapter.calls == []


def test_partial_or_visual_length_mismatched_capture_refuses_before_scoring(tmp_path):
    manifest, metadata, capture, image_root, adapter = _fixture(tmp_path)
    payload = json.loads(capture.read_text())
    payload["status"] = "canary_complete"
    capture.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="partial or contains identity failures"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            native_capture=capture,
            image_root=image_root,
            output_dir=tmp_path / "partial",
            adapter=adapter,
        )
    assert adapter.calls == []

    payload["status"] = "complete_passed"
    payload["cases"][0]["visual_token_counts_own_swap1_swap2"] = [576, 576, 288]
    capture.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="visual token lengths differ"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            native_capture=capture,
            image_root=image_root,
            output_dir=tmp_path / "visual-mismatch",
            adapter=adapter,
        )
    assert adapter.calls == []


def test_cross_model_adapter_is_refused(tmp_path):
    manifest, metadata, capture, image_root, adapter = _fixture(tmp_path)
    adapter.fingerprint = lambda: {"model_family": "hulu-med-4b"}
    with pytest.raises(ContractError, match="another model-family"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            native_capture=capture,
            image_root=image_root,
            output_dir=tmp_path / "wrong-model",
            adapter=adapter,
        )
