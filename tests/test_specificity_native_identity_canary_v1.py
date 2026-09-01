from __future__ import annotations

import hashlib
import json

import pytest

from anchor.corrected_sgta.run_specificity_native_identity_canary_v1 import (
    run_identity_canary,
)
from anchor.corrected_sgta.specificity_full_replay_runtime_v1 import (
    ContractError,
    IDENTITY_PROTOCOL_ID,
)
from tests.test_specificity_full_replay_runtime_v1 import _files


class FakeNativeAdapter:
    def __init__(self, text: str):
        self.text = text

    def fingerprint(self):
        return {"model_family": "fake-huatuo", "revision": "v1"}

    def generate_native_identity(
        self, *, image_path, question, seed, max_new_tokens
    ):
        return {
            "text": self.text,
            "direct_output_sequence_ids": [11, 12, 13],
            "directly_captured_output_sequences": True,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "seed": seed,
            "decode_contract": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "repetition_penalty": 1.2,
            },
            "hit_max_new_tokens": False,
        }


def _prepare(tmp_path):
    manifest, metadata, _, image_root, row = _files(tmp_path)
    generation = {
        "model": "huatuo",
        "seed": 42,
        "max_new_tokens": 512,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "min_new_tokens": 1,
            "repetition_penalty": 1.2,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            generation, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    row_payload = json.loads(manifest.read_text())
    row_payload["source_question_id"] = "Q-1"
    row_payload["source_generation_fingerprint"] = fingerprint
    manifest.write_text(json.dumps(row_payload, sort_keys=True) + "\n")
    config = tmp_path / "generation_config.json"
    config.write_text(json.dumps({**generation, "fingerprint": fingerprint}))
    meta = json.loads(metadata.read_text())
    meta["source_generation_fingerprint"] = fingerprint
    meta["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    meta["provenance"] = {
        "source_generation_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()
    }
    metadata.write_text(json.dumps(meta))
    return manifest, metadata, image_root, row_payload, config


def test_identity_canary_passes_only_exact_visible_text(tmp_path):
    manifest, metadata, image_root, row, config = _prepare(tmp_path)
    output = tmp_path / "identity_pass.json"
    result = run_identity_canary(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        source_generation_config=config,
        output=output,
        adapter=FakeNativeAdapter(row["full_visible_answer"]),
    )
    assert result["protocol"] == IDENTITY_PROTOCOL_ID
    assert result["status"] == "passed"
    assert result["decoded_visible_text_identity"] is True
    assert result["gpu_scoring_authorized"] is True


def test_identity_canary_freezes_mismatch_without_authorization(tmp_path):
    manifest, metadata, image_root, _, config = _prepare(tmp_path)
    output = tmp_path / "identity_fail.json"
    result = run_identity_canary(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        source_generation_config=config,
        output=output,
        adapter=FakeNativeAdapter("A different answer."),
    )
    assert result["status"] == "failed"
    assert result["decoded_visible_text_identity"] is False
    assert result["gpu_scoring_authorized"] is False
    assert output.is_file()


def test_identity_canary_rejects_generation_config_tampering(tmp_path):
    manifest, metadata, image_root, row, config = _prepare(tmp_path)
    payload = json.loads(config.read_text())
    payload["generation"]["repetition_penalty"] = 1.1
    config.write_text(json.dumps(payload))
    meta = json.loads(metadata.read_text())
    meta["provenance"]["source_generation_config_sha256"] = hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ContractError, match="self-consistent"):
        run_identity_canary(
            manifest=manifest,
            metadata=metadata,
            image_root=image_root,
            source_generation_config=config,
            output=tmp_path / "must_not_exist.json",
            adapter=FakeNativeAdapter(row["full_visible_answer"]),
        )
    assert not (tmp_path / "must_not_exist.json").exists()
