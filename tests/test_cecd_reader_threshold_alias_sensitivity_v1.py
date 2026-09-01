import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/cecd_reader_threshold_alias_sensitivity_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_sensitivity_contract_is_outcome_blind_and_cannot_change_primary_gate():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cecd-reader-threshold-alias-sensitivity-v1"
    assert payload["frozen_before_formal_ce_outcomes"] is True
    assert payload["formal_ce_outcomes_consumed"] is False
    assert payload["primary_ce_gate_modified"] is False
    assert payload["falsification"]["thresholds_may_change_primary_ce_gate"] is False
    assert payload["execution_policy"]["score_dependent_analysis"] == (
        "only_after_locked_primary_cecd_go"
    )
    assert payload["bindings"]["dev_factorial_rows"] is None
    assert payload["bindings"]["confirmation_factorial_rows"] is None


def test_sensitivity_closure_matches_formal_ce_and_requires_named_reader_join():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["closure"]["findings"] == [
        "aortic_enlargement",
        "cardiomegaly",
        "pleural_effusion",
        "pulmonary_fibrosis",
    ]
    assert payload["closure"]["reader_panel"] == ["R8", "R9", "R10"]
    assert payload["closure"]["positional_reader_vote_lists_allowed"] is False
    assert payload["inference"]["primary_cluster_unit"] == "whole_image_id"
    assert payload["inference"]["models_resampled_synchronously"] is True
    assert payload["inference"]["reader_bootstrap"] is False


def test_every_available_source_binding_is_hash_exact():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, record in payload["bindings"].items():
        if record is None:
            continue
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), name
        assert _sha256(path) == record["sha256"], name

