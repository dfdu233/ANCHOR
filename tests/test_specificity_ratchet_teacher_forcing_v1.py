import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from anchor.corrected_sgta.specificity_ratchet_teacher_forcing_v1 import (
    ContractError,
    TeacherForcedTrace,
    build_control_plan,
    load_admitted_manifest,
    map_constraint_spans,
    run_runtime,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _span(target: str, occurrence: str, start: int = 0) -> dict:
    left = target.index(occurrence, start)
    return {
        "char_start": left,
        "char_end_exclusive": left + len(occurrence),
        "text": occurrence,
        "utf8_sha256": _sha(occurrence.encode("utf-8")),
    }


class FakeContextualAdapter:
    def fingerprint(self):
        return {
            "model_family": "cpu-fake",
            "model_revision": "fake-v1",
            "tokenizer_revision": "regex-v1",
            "teacher_forcing_template_sha256": _sha(b"fake-template-v1"),
            "intermediate_logit_rule": "final-norm-then-tied-head",
        }

    def score(self, *, image_path, question, target, condition):
        matches = list(re.finditer(r"\S+", target))
        token_ids = [100 + index for index in range(len(matches))]
        offsets = [(match.start(), match.end()) for match in matches]
        base = np.asarray([-0.2 - 0.03 * index for index in range(len(matches))])
        layers = np.vstack((base - 0.2, base - 0.1, base))
        image_sha = _sha(image_path.read_bytes()) if image_path is not None else None
        serialized = json.dumps(
            {"condition": condition, "question": question, "target": target, "image": image_sha},
            sort_keys=True,
        ).encode()
        return TeacherForcedTrace(
            condition=condition,
            target=target,
            token_ids=token_ids,
            token_offsets=offsets,
            offset_unit="unicode_character",
            layer_ids=["decoder.0", "decoder.1", "decoder.2"],
            layer_gold_logp=layers.tolist(),
            serialized_input_sha256=_sha(serialized),
            prompt_sha256=_sha(question.encode()),
            target_sha256=_sha(target.encode()),
            image_sha256=image_sha,
            template_id="fake-chat-template-v1",
            contextual_offsets_certified=True,
        )


class UncertifiedAdapter(FakeContextualAdapter):
    def score(self, **kwargs):
        trace = super().score(**kwargs)
        values = trace.__dict__.copy()
        values["contextual_offsets_certified"] = False
        return TeacherForcedTrace(**values)


class TargetDependentTemplateAdapter(FakeContextualAdapter):
    """Invalid adapter that changes the template for only the child target."""

    def score(self, **kwargs):
        trace = super().score(**kwargs)
        if kwargs["target"] == "A left lesion is present.":
            values = trace.__dict__.copy()
            values["template_id"] = "fake-chat-template-child-v2"
            return TeacherForcedTrace(**values)
        return trace


def _manifest(tmp_path: Path, count: int = 4):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    roles = ["supported_specificity_control", "causal_escalation_error"]
    for index in range(count):
        image = image_root / f"{index}.bin"
        image.write_bytes(f"image-{index}".encode())
        child = "A left lesion is present."
        rows.append(
            {
                "manifest_protocol_id": "specificity-ratchet-mechanism-v1",
                "sample_id": f"SR-{index}",
                "case_id": f"CASE-{index}",
                "edge_id": f"EDGE-{index}",
                "image_relpath": image.name,
                "question": "What is present?",
                "parent_target": "A lesion is present.",
                "child_target": child,
                "constraint_char_spans_in_child": [_span(child, "left")],
                "scientific_role": roles[index % 2],
                "split": "dev",
                "edge_type": "laterality",
                "modality_stratum": "XR",
                "anatomy_stratum": "thorax",
                "prompt_requested_increment": False,
            }
        )
    manifest = tmp_path / "samples.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    manifest.write_bytes(payload)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest_protocol_id": "specificity-ratchet-mechanism-v1",
                "status": "physician_admitted",
                "image_disjoint": True,
                "manifest_sha256": _sha(payload),
                "n_scientific_edges": count,
            }
        )
    )
    return manifest, metadata, image_root, rows


def test_utf8_repeated_spans_map_every_occurrence():
    target = "左 lower lesion and 左 upper lesion"
    second = target.index("左", 1)
    spans = [_span(target, "左"), _span(target, "左", second)]
    offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", target)]
    assert map_constraint_spans(target, spans, offsets, "unicode_character") == [0, 4]


def test_constraint_boundary_spill_into_nonwhitespace_is_refused():
    target = "A left-sided lesion"
    offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", target)]
    with pytest.raises(ContractError, match="spills"):
        map_constraint_spans(target, [_span(target, "left")], offsets, "unicode_character")


def test_invalid_utf8_byte_boundary_is_refused():
    target = "左 lesion"
    with pytest.raises(ContractError, match="UTF-8 boundary"):
        map_constraint_spans(target, [_span(target, "左")], [(0, 1), (4, 10)], "utf8_byte")


def test_runtime_scores_same_image_parent_child_and_resumes_atomically(tmp_path):
    manifest, metadata, image_root, _ = _manifest(tmp_path)
    output = tmp_path / "run"
    first = run_runtime(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=output,
        adapter=FakeContextualAdapter(),
        split="dev",
        command=["fake-command"],
        _historical_contract_test_only=True,
    )
    assert first["status"] == "complete"
    assert first["rows"] == 4
    assert first["resumed_rows"] == 0
    shards = sorted((output / "shards").glob("*.json"))
    assert len(shards) == 4
    one = json.loads(shards[0].read_text())["payload"]
    assert one["signals"]["token_counts"] == {
        "constraint": 1,
        "parent_target": 4,
        "child_target": 5,
        "matched_parent": 1,
        "matched_child_nonconstraint": 1,
    }
    assert len(one["signals"]["image_layer_signals"]["constraint_minus_parent_sequence"]) == 3
    assert one["signals"]["text_only_nuisance"]["lexical_proxy_only_not_clinical_evidence"] is True
    second = run_runtime(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=output,
        adapter=FakeContextualAdapter(),
        split="dev",
        command=["fake-command"],
        _historical_contract_test_only=True,
    )
    assert second["resumed_rows"] == 4
    controls = json.loads((output / "controls.json").read_text())
    assert controls["shuffled_parent_pairing"]["coverage"] == 1.0
    assert controls["sequence_length_role_permutation"]["coverage"] == 1.0


def test_f6_rejected_runtime_refuses_by_default_before_outputs(tmp_path):
    manifest, metadata, image_root, _ = _manifest(tmp_path, count=2)
    output = tmp_path / "run"
    with pytest.raises(ContractError, match="F6-rejected"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            image_root=image_root,
            output_dir=output,
            adapter=FakeContextualAdapter(),
            command=["fake-command"],
        )
    assert not output.exists()


def test_runtime_refuses_uncertified_standalone_offsets_without_shard(tmp_path):
    manifest, metadata, image_root, _ = _manifest(tmp_path, count=2)
    output = tmp_path / "run"
    with pytest.raises(ContractError, match="contextual offsets"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            image_root=image_root,
            output_dir=output,
            adapter=UncertifiedAdapter(),
            command=["fake-command"],
            _historical_contract_test_only=True,
        )
    assert not list((output / "shards").glob("*.json"))


def test_runtime_refuses_parent_child_template_drift_without_complete_artifact(tmp_path):
    manifest, metadata, image_root, _ = _manifest(tmp_path, count=2)
    output = tmp_path / "run"
    with pytest.raises(ContractError, match="parent/child teacher-forcing templates differ"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            image_root=image_root,
            output_dir=output,
            adapter=TargetDependentTemplateAdapter(),
            command=["fake-command"],
            _historical_contract_test_only=True,
        )
    assert not (output / "COMPLETE.json").exists()


def test_resume_detects_corrupt_payload(tmp_path):
    manifest, metadata, image_root, _ = _manifest(tmp_path, count=2)
    output = tmp_path / "run"
    arguments = dict(
        manifest=manifest,
        metadata=metadata,
        image_root=image_root,
        output_dir=output,
        adapter=FakeContextualAdapter(),
        command=["fake-command"],
        _historical_contract_test_only=True,
    )
    run_runtime(**arguments)
    shard_path = sorted((output / "shards").glob("*.json"))[0]
    shard = json.loads(shard_path.read_text())
    shard["payload"]["case_id"] = "tampered"
    shard_path.write_text(json.dumps(shard))
    with pytest.raises(ContractError, match="checksum"):
        run_runtime(**arguments)


def test_manifest_metadata_is_inseparable_and_physician_admitted(tmp_path):
    manifest, metadata, _, _ = _manifest(tmp_path)
    altered = json.loads(metadata.read_text())
    altered["status"] = "candidate_only"
    metadata.write_text(json.dumps(altered))
    with pytest.raises(ContractError, match="physician-admitted"):
        load_admitted_manifest(manifest, metadata)


def test_sparse_control_bins_are_reported_not_relaxed():
    payload = {
        "sample_id": "only",
        "case_id": "CASE",
        "split": "dev",
        "edge_type": "subtype",
        "scientific_role": "causal_escalation_error",
        "signals": {"token_counts": {"parent_target": 3, "child_target": 4, "constraint": 1}},
    }
    plan = build_control_plan([payload], 7)
    assert plan["shuffled_parent_pairing"]["coverage"] == 0.0
    assert plan["sequence_length_role_permutation"]["coverage"] == 0.0
    assert plan["shuffled_parent_pairing"]["no_caliper_relaxation"] is True
