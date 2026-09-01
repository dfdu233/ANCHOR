from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from anchor.corrected_sgta.specificity_full_replay_runtime_v1 import (
    ContractError,
    IDENTITY_PROTOCOL_ID,
    MANIFEST_PROTOCOL_ID,
    compute_full_replay_signals,
    load_full_replay_manifest,
    run_full_replay,
)
from anchor.corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    TeacherForcedTrace,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FakeFullReplayAdapter:
    def fingerprint(self):
        return {"model_family": "fake-full-replay", "revision": "v1"}

    def visual_token_count(self, *, image_path, question):
        return 16

    def score(self, *, image_path, question, target, condition):
        matches = list(re.finditer(r"\S+", target))
        offsets = [(match.start(), match.end()) for match in matches]
        token_ids = list(range(100, 100 + len(matches)))
        image_hash = _sha_bytes(image_path.read_bytes()) if image_path else None
        image_delta = 0.2 if image_path and image_path.name == "own.bin" else 0.0
        base = np.asarray([-2.0 + 0.03 * index for index in range(len(matches))])
        layers = np.vstack((base, base + image_delta, base + 2 * image_delta))
        return TeacherForcedTrace(
            condition=condition,
            target=target,
            token_ids=token_ids,
            token_offsets=offsets,
            offset_unit="unicode_character",
            layer_ids=["decoder.0", "decoder.1", "decoder.2"],
            layer_gold_logp=layers.tolist(),
            serialized_input_sha256=_sha_bytes(
                f"{condition}|{image_hash}|{target}".encode()
            ),
            prompt_sha256=_sha_bytes(question.encode()),
            target_sha256=_sha_bytes(target.encode()),
            image_sha256=image_hash,
            template_id="fake-full-answer-template",
            contextual_offsets_certified=True,
        )


def _files(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("own.bin", "swap1.bin", "swap2.bin", "swap3.bin"):
        (image_root / name).write_bytes(name.encode())
    target = "A small left lesion is present."
    left = target.index("left")
    row = {
        "manifest_protocol_id": MANIFEST_PROTOCOL_ID,
        "sample_id": "SRF1-test",
        "case_id": "CASE-own",
        "edge_id": "EDGE-own",
        "image_relpath": "own.bin",
        "question": "What is present?",
        "full_visible_answer": target,
        "full_visible_answer_sha256": _sha_bytes(target.encode()),
        "constraint_char_spans_in_visible_answer": [
            {
                "char_start": left,
                "char_end_exclusive": left + 4,
                "text": "left",
                "utf8_sha256": _sha_bytes(b"left"),
            }
        ],
        "scientific_role": "causal_escalation_error",
        "split": "dev",
        "edge_type": "laterality",
        "modality_stratum": "XR",
        "anatomy_stratum": "thorax",
        "prompt_requested_increment": False,
        "source_generation_fingerprint": "source-fingerprint",
        "swap_candidates": [
            {"case_id": f"CASE-swap{index}", "image_relpath": f"swap{index}.bin", "split": "dev"}
            for index in (1, 2, 3)
        ],
    }
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest_protocol_id": MANIFEST_PROTOCOL_ID,
                "status": "physician_admitted_full_visible_replay",
                "manifest_sha256": _sha_bytes(manifest.read_bytes()),
                "rows": 1,
                "source_model": "huatuo",
                "source_generation_fingerprint": "source-fingerprint",
                "native_generation_sequence_certified": False,
                "isolated_parent_child_runtime_prohibited": True,
            }
        )
    )
    canary = tmp_path / "identity.json"
    canary.write_text(
        json.dumps(
            {
                "protocol": IDENTITY_PROTOCOL_ID,
                "status": "passed",
                "manifest_sha256": _sha_bytes(manifest.read_bytes()),
                "metadata_sha256": _sha_bytes(metadata.read_bytes()),
                "source_model": "huatuo",
                "source_generation_fingerprint": "source-fingerprint",
                "adapter_fingerprint": {
                    "model_family": "fake-full-replay",
                    "revision": "v1",
                },
                "sample_id": row["sample_id"],
                "directly_captured_output_sequences": True,
                "decoded_visible_text_identity": True,
                "gpu_scoring_authorized": True,
            }
        )
    )
    return manifest, metadata, canary, image_root, row


def test_manifest_requires_native_identity_canary_before_model_loading(tmp_path):
    manifest, metadata, _, _, _ = _files(tmp_path)
    with pytest.raises(ContractError, match="identity canary is required"):
        load_full_replay_manifest(manifest, metadata)


def test_manifest_rejects_identity_sidecar_from_different_metadata(tmp_path):
    manifest, metadata, canary, _, _ = _files(tmp_path)
    payload = json.loads(canary.read_text())
    payload["metadata_sha256"] = "0" * 64
    canary.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="different metadata"):
        load_full_replay_manifest(
            manifest, metadata, identity_canary=canary, require_identity=True
        )


def test_full_replay_scores_complete_answer_with_exact_swap_controls(tmp_path):
    manifest, metadata, canary, image_root, _ = _files(tmp_path)
    result = run_full_replay(
        manifest=manifest,
        metadata=metadata,
        identity_canary=canary,
        image_root=image_root,
        output_dir=tmp_path / "run",
        adapter=FakeFullReplayAdapter(),
        split="dev",
        swaps_per_row=2,
        command=["fake-command"],
    )
    assert result["status"] == "complete"
    shard = json.loads(next((tmp_path / "run" / "shards").glob("*.json")).read_text())
    payload = shard["payload"]
    assert payload["status"] == "ok"
    assert len(payload["selected_swaps"]) == 2
    assert payload["signals"]["token_counts"]["constraint"] == 1
    assert payload["signals"]["swap_images"]["count"] == 2
    assert payload["signals"]["text_only_secondary"]["lexical_sensitivity_only"] is True


def test_full_replay_rejects_adapter_different_from_identity_canary(tmp_path):
    manifest, metadata, canary, image_root, _ = _files(tmp_path)
    payload = json.loads(canary.read_text())
    payload["adapter_fingerprint"] = {"model_family": "different"}
    canary.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="differs from native identity"):
        run_full_replay(
            manifest=manifest,
            metadata=metadata,
            identity_canary=canary,
            image_root=image_root,
            output_dir=tmp_path / "must_not_run",
            adapter=FakeFullReplayAdapter(),
        )


def test_signal_contract_refuses_swap_template_drift(tmp_path):
    _, _, _, image_root, row = _files(tmp_path)
    adapter = FakeFullReplayAdapter()
    own = adapter.score(
        image_path=image_root / "own.bin",
        question=row["question"],
        target=row["full_visible_answer"],
        condition="image",
    )
    swap1 = adapter.score(
        image_path=image_root / "swap1.bin",
        question=row["question"],
        target=row["full_visible_answer"],
        condition="image",
    )
    swap2 = adapter.score(
        image_path=image_root / "swap2.bin",
        question=row["question"],
        target=row["full_visible_answer"],
        condition="image",
    )
    drift = TeacherForcedTrace(**{**swap2.__dict__, "template_id": "drift"})
    text = adapter.score(
        image_path=None,
        question=row["question"],
        target=row["full_visible_answer"],
        condition="text_only",
    )
    with pytest.raises(ContractError, match="template drift"):
        compute_full_replay_signals(
            row=row,
            own_trace=own,
            swap_traces=[swap1, drift],
            text_trace=text,
            own_image_sha256=_sha_bytes((image_root / "own.bin").read_bytes()),
            swap_image_sha256=[
                _sha_bytes((image_root / "swap1.bin").read_bytes()),
                _sha_bytes((image_root / "swap2.bin").read_bytes()),
            ],
        )
