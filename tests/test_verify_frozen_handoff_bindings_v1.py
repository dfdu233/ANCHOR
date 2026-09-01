import hashlib
import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.verify_frozen_handoff_bindings_v1 import verify_handoff


SCHEMA = "unit-test-frozen-handoff-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff(tmp_path: Path) -> Path:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    image = tmp_path / "image.bin"
    image.write_bytes(b"dicom-like-payload")
    payload = {
        "schema_version": SCHEMA,
        "source_bindings": {
            "source": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": _sha(source),
            }
        },
        "frozen_input": {"image_path": str(image), "image_sha256": _sha(image)},
    }
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    return handoff


def test_valid_handoff_is_verified(tmp_path: Path) -> None:
    result = verify_handoff(_handoff(tmp_path), expected_schema=SCHEMA)
    assert result["verified"] is True
    assert len(result["bindings_checked"]) == 1


def test_source_drift_fails_closed(tmp_path: Path) -> None:
    handoff = _handoff(tmp_path)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    Path(payload["source_bindings"]["source"]["path"]).write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source drift"):
        verify_handoff(handoff, expected_schema=SCHEMA)


def test_image_drift_fails_closed(tmp_path: Path) -> None:
    handoff = _handoff(tmp_path)
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    Path(payload["frozen_input"]["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen image drift"):
        verify_handoff(handoff, expected_schema=SCHEMA)


def test_schema_drift_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema drift"):
        verify_handoff(_handoff(tmp_path), expected_schema="wrong-schema")
