import hashlib
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.run_natural_oe_diagnostic_completion_pilot_v1 import (
    validate_inputs,
)
import anchor.corrected_sgta.run_natural_oe_diagnostic_completion_pilot_v1 as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    dicom = tmp_path / "image.dicom"
    dicom.write_bytes(b"dicom-placeholder")
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "item_id": "item",
        "image_id": "image",
        "dicom_path": str(dicom),
        "prompt_id": "natural_abnormality_listing_v1",
        "selection_uses_reader_labels": True,
        "selection_uses_model_outputs": False,
        "generation_receives_reader_labels": False,
        "generation_receives_target_edge": False,
    }
    manifest.write_text(json.dumps(row) + "\n")
    contract = tmp_path / "contract.json"
    payload = {
        "version": "natural-oe-diagnostic-completion-pilot-v1",
        "prompt": "What abnormalities are visible in this chest X-ray? Answer in one concise sentence.",
        "generation_receives_reader_labels": False,
        "generation_receives_target_edge": False,
        "generation_manifest": str(manifest),
        "generation_manifest_sha256": _sha(manifest),
        "images": 1,
    }
    contract.write_text(json.dumps(payload))
    progression = tmp_path / "progression.json"
    progression.write_text(
        json.dumps(
            {
                "allowed_next_stage": {
                    "name": "natural_oe_bounded_construct_pilot",
                    "maximum_images": 128,
                }
            }
        )
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "version": "natural-oe-pilot-launch-authorization-v1",
                "generation_authorized": True,
                "pilot_contract_sha256": _sha(contract),
                "maximum_images": 1,
                "runner_sha256": _sha(Path(runner.__file__)),
                "progression_gate": str(progression),
                "progression_gate_sha256": _sha(progression),
            }
        )
    )
    return contract, authorization, manifest


def test_validation_enforces_sealed_generation_view(tmp_path):
    contract, authorization, _ = _fixture(tmp_path)
    _, _, rows = validate_inputs(contract, authorization, limit=1)
    assert len(rows) == 1


def test_validation_rejects_reader_label_leak(tmp_path):
    contract, authorization, manifest = _fixture(tmp_path)
    row = json.loads(manifest.read_text())
    row["child_votes"] = 0
    manifest.write_text(json.dumps(row) + "\n")
    payload = json.loads(contract.read_text())
    payload["generation_manifest_sha256"] = _sha(manifest)
    contract.write_text(json.dumps(payload))
    auth = json.loads(authorization.read_text())
    auth["pilot_contract_sha256"] = _sha(contract)
    authorization.write_text(json.dumps(auth))
    with pytest.raises(ValueError, match="leaked"):
        validate_inputs(contract, authorization, limit=1)


def test_validation_enforces_authorized_image_ceiling(tmp_path):
    contract, authorization, _ = _fixture(tmp_path)
    auth = json.loads(authorization.read_text())
    auth["maximum_images"] = 0
    authorization.write_text(json.dumps(auth))
    with pytest.raises(PermissionError, match="exceeds"):
        validate_inputs(contract, authorization, limit=1)
