import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from anchor.corrected_sgta.analyze_clinical_autoregressive_lockin_v1 import (
    analyze_run,
    analyze_payloads,
)
from anchor.corrected_sgta.clinical_autoregressive_lockin_probe_v1 import (
    ContextualContinuationTrace,
    ContractError,
    GreedyGenerationTrace,
    PromptEndTrace,
    _validate_prompt_end_trace,
    run_runtime,
)
from anchor.corrected_sgta.validate_clinical_lockin_stimulus_contract_v1 import (
    ConstructContractError,
    FUTURE_MODE,
    FUTURE_PROTOCOL,
    _canonical as construct_canonical,
    audit_legacy_v4,
    validate_v5_contract,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _offsets(text: str):
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


class FakeLockinAdapter:
    def __init__(self, consume_response=False):
        self.consume_response = consume_response

    def fingerprint(self):
        return {
            "model_family": "cpu-fake",
            "model_artifact_fingerprint": "fake-model-v1",
            "tokenizer_fingerprint": "whitespace-v1",
            "chat_template_sha256": _sha(b"fake-chat-template"),
            "multimodal_expansion_contract": "fake-image-placeholder-expansion-v1",
            "prompt_end_position_contract": "last_expanded_prompt_token_before_first_assistant_response_token",
            "layer_logit_lens_contract": "fake-final-norm-tied-head-every-layer",
            "generation_decode_contract": "greedy-num_beams1-sampling_false-max_new_tokens256",
        }

    def generate(self, *, image_path, prompt):
        image = image_path.read_bytes()
        image_hash = _sha(image)
        return GreedyGenerationTrace(
            text="The chest X-ray shows opacity.",
            generated_token_ids=[10, 11, 12],
            image_sha256=image_hash,
            prompt_sha256=_sha(prompt.encode()),
            serialized_prompt_sha256=_sha((prompt + image_hash).encode()),
            template_id="fake-template-v1",
            decode_contract="greedy-num_beams1-sampling_false-max_new_tokens256",
            hit_max_new_tokens=False,
        )

    def prompt_end(self, *, image_path, prompt, condition):
        image = image_path.read_bytes() if image_path else b""
        sign = 0.0 if condition == "text_only" else 1.0 if b"positive" in image else -1.0
        image_hash = _sha(image) if image_path else None
        return PromptEndTrace(
            condition=condition,
            prompt=prompt,
            layer_ids=["l1", "l2", "l3", "l4"],
            layer_fractions=[0.25, 0.5, 0.75, 1.0],
            layer_prompt_end_hidden=[[sign, 0.1], [sign, 0.2], [sign, 0.3], [sign, 0.4]],
            serialized_prompt_sha256=_sha((prompt + str(image_hash)).encode()),
            prompt_sha256=_sha(prompt.encode()),
            image_sha256=image_hash,
            template_id="fake-template-v1",
            prompt_end_position_contract="last_expanded_prompt_token_before_first_assistant_response_token",
            first_response_token_consumed=self.consume_response,
            multimodal_expansion_certified=True,
        )

    def score(self, *, image_path, prompt, prefix, continuation, condition):
        image_hash = _sha(image_path.read_bytes()) if image_path else None
        sign = 0.1 if image_path and b"positive" in image_path.read_bytes() else -0.1
        if condition == "text_only":
            sign = 0.0
        prefix_offsets = _offsets(prefix)
        continuation_offsets = _offsets(continuation)
        token_count = len(continuation_offsets)
        values = np.asarray(
            [[-0.8 + sign] * token_count, [-0.7 + sign] * token_count,
             [-0.6 + sign] * token_count, [-0.5 + sign] * token_count]
        )
        serialized = json.dumps(
            {
                "prompt": prompt,
                "prefix": prefix,
                "continuation": continuation,
                "image": image_hash,
            },
            sort_keys=True,
        ).encode()
        return ContextualContinuationTrace(
            condition=condition,
            prompt=prompt,
            prefix=prefix,
            continuation=continuation,
            prefix_token_ids=list(range(10, 10 + len(prefix_offsets))),
            prefix_token_offsets=prefix_offsets,
            continuation_token_ids=list(range(100, 100 + token_count)),
            continuation_token_offsets=continuation_offsets,
            offset_unit="unicode_character",
            layer_ids=["l1", "l2", "l3", "l4"],
            layer_fractions=[0.25, 0.5, 0.75, 1.0],
            layer_gold_logp=values.tolist(),
            serialized_input_sha256=_sha(serialized),
            prompt_sha256=_sha(prompt.encode()),
            prefix_sha256=_sha(prefix.encode()),
            continuation_sha256=_sha(continuation.encode()),
            image_sha256=image_hash,
            template_id="fake-template-v1",
            contextual_offsets_certified=True,
            final_layer_matches_standard_logits=True,
        )


def _runtime_manifest(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    prompt = "What abnormalities are present?"
    for block in range(2):
        refs = {}
        for support, word in ((0, "negative"), (3, "positive")):
            for role in ("anchor", "same"):
                path = image_root / f"b{block}-{support}-{role}.bin"
                path.write_bytes(f"{word}-{block}-{role}".encode())
                refs[(support, role)] = {
                    "image_id": path.stem,
                    "dicom_relpath": path.name,
                    "dicom_sha256": _sha(path.read_bytes()),
                    "positive_votes": support,
                    "reader_count": 3,
                }
        for support in (0, 3):
            ladder = [
                {
                    "step": step,
                    "prefix": prefix,
                    "prefix_utf8_sha256": _sha(prefix.encode()),
                    "phase": "empty" if step == 0 else "common_clinical_prefix" if step <= 2 else "claim_specific_modifier",
                    "claim_begins_after_prefix": True,
                }
                for step, prefix in enumerate(
                    ["", "The chest ", "The chest shows ", "The chest shows a ", "The chest shows a right "]
                )
            ]
            control_prefix = "Regarding opacity, the finding is"
            rows.append(
                {
                    "manifest_protocol_id": "clinical-autoregressive-lockin-manifest-v4-claim-specific-prompt",
                    "sample_id": f"B{block}-v{support}",
                    "block_id": f"B{block}",
                    "prompt_end_probe_role": "probe_fit" if block == 0 else "probe_eval",
                    "split": "dev",
                    "finding": "lung_opacity",
                    "positive_votes": support,
                    "prompt_condition": "negative_obligation",
                    "prompt": prompt,
                    "prompt_utf8_sha256": _sha(prompt.encode()),
                    "embedded_claim": "opacity",
                    "embedded_polarity": "present",
                    "prefix_ladder": ladder,
                    "non_attractor_preclaim_template_control": {
                        "prefix": control_prefix,
                        "prefix_utf8_sha256": _sha(control_prefix.encode()),
                        "present_continuation": " present.",
                        "absent_continuation": " absent.",
                        "role": "teacher-forced-control-only",
                    },
                    "image_conditions": {
                        "original": refs[(support, "anchor")],
                        "same_support_swap": refs[(support, "same")],
                        "opposite_support_swap": refs[(3 - support, "anchor")],
                    },
                }
            )
    manifest = tmp_path / "manifest.jsonl"
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in sorted(rows, key=lambda row: row["sample_id"])
    )
    manifest.write_bytes(payload)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest_protocol_id": "clinical-autoregressive-lockin-manifest-v4-claim-specific-prompt",
                "status": "dev_frozen_gpu_not_run",
                "split": "dev",
                "dev_model_output_used_for_selection": False,
                "confirmation_split_locked": True,
                "manifest_sha256": _sha(payload),
                "anchor_rows": len(rows),
            }
        )
    )
    return manifest, metadata, image_root


def test_rejected_fixed_continuation_runtime_fails_before_adapter_or_shard(tmp_path):
    manifest, metadata, image_root = _runtime_manifest(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(ContractError, match="F6|f6|rejected"):
        run_runtime(
            manifest=manifest,
            metadata=metadata,
            image_root=image_root,
            output_dir=output,
            adapter=FakeLockinAdapter(),
            command=["fake"],
        )
    assert not output.exists()


def test_runtime_refuses_prompt_end_trace_after_response_token(tmp_path):
    image = tmp_path / "positive.bin"
    image.write_bytes(b"positive")
    prompt = "What abnormalities are present?"
    trace = FakeLockinAdapter(consume_response=True).prompt_end(
        image_path=image, prompt=prompt, condition="image"
    )
    with pytest.raises(ContractError, match="consumed"):
        _validate_prompt_end_trace(
            trace,
            prompt=prompt,
            condition="image",
            expected_image_sha256=_sha(image.read_bytes()),
        )


def _synthetic_payloads():
    rows = []
    profiles = [1.0, 1.0, 0.9, 0.35, 0.1]
    for finding_index, finding in enumerate(("lung_opacity", "pleural_effusion")):
        for block in range(12):
            role = "probe_fit" if block < 6 else "probe_eval"
            for support in (0, 3):
                sign = 1.0 if support == 3 else -1.0
                hidden = [[sign, 0.01 * (block + 1), 0.02 * finding_index] for _ in range(4)]
                opposite = [[-sign, 0.01 * (block + 1), 0.02 * finding_index] for _ in range(4)]
                ladder = []
                lengths = [0, 5, 6, 7, 9]
                for step, profile in enumerate(profiles):
                    original_logp = [-0.5] * 4
                    text_logp = [-1.0 + 0.15 * step] * 4
                    ladder.append(
                        {
                            "step": step,
                            "prefix_token_ids": list(range(lengths[step])),
                            "continuation_token_ids": [1, 2],
                            "effects": {
                                "causal_excess_over_same_support": [profile] * 4,
                                "absolute_original_minus_same": [0.0] * 4,
                            },
                            "layer_mean_logp": {
                                "original": original_logp,
                                "text_only": text_logp,
                            },
                        }
                    )
                rows.append(
                    {
                        "status": "ok",
                        "sample_id": f"{finding}-{block}-{support}",
                        "block_id": f"{finding}-{block}",
                        "finding": finding,
                        "positive_votes": support,
                        "prompt_end_probe_role": role,
                        "layer_ids": ["l1", "l2", "l3", "l4"],
                        "layer_fractions": [0.25, 0.5, 0.75, 1.0],
                        "template_id": "t",
                        "generation_endpoint": {
                            "original_opposite_same_embedded_claim_surface": True,
                            "original_opposite_exact_full_text_collision": False,
                            "clinical_correctness_assigned": False,
                        },
                        "prompt_end_readout": {
                            "layer_hidden": {
                                "original": hidden,
                                "same_support_swap": hidden,
                                "opposite_support_swap": opposite,
                                "text_only": [[0.0, 0.1, 0.0] for _ in range(4)],
                            },
                            "text_only_control": {
                                "same_prompt_no_image": True,
                                "used_to_fit_or_select_prompt_end_probe": False,
                                "token_identity_required": False,
                                "layer_and_hidden_dimension_alignment_required": True,
                            },
                        },
                        "prefix_ladder": ladder,
                        "non_attractor_preclaim_template_control": {
                            "causal_excess_over_same_support": [1.0] * 4,
                            "not_pre_response_hidden_decoding": True,
                        },
                    }
                )
    return rows


def test_legacy_numeric_analyzer_is_forensic_and_cannot_authorize_even_on_synthetic_signal():
    result = analyze_payloads(_synthetic_payloads(), bootstrap_replicates=100, seed=7)
    assert result["directional_admission_pass"] is True
    assert result["legacy_counterfactual_lockin_gate"] is True
    assert result["lockin_mechanism_pass"] is False
    assert result["decision"] == "rejected_f6_construct_invalid"
    assert result["confirmation_or_patching_authorized"] is False
    for name in ("q25", "q50", "q75"):
        probe = result["layer_results"][name]["point"]["prompt_end_probe"]
        assert probe["fit_uses_only_prompt_end_hidden"] is True
        assert probe["teacher_forced_likelihood_used_for_gate1"] is False


def test_formal_v4_analyze_run_is_rejected_before_output(tmp_path):
    run = tmp_path / "runtime"
    shards = run / "shards"
    shards.mkdir(parents=True)
    payloads = _synthetic_payloads()
    config = {"config_fingerprint": "fake-runtime-fingerprint"}
    (run / "config.json").write_text(json.dumps(config))
    (run / "COMPLETE.json").write_text(
        json.dumps(
            {
                "runtime_protocol_id": "clinical-autoregressive-lockin-probe-v1",
                "analysis_input_complete": True,
                "scientific_gate_authorized": False,
                "analyzable_rows": len(payloads),
            }
        )
    )
    length_matches = {
        str(step): {
            "eligible": True,
            "exact_common_counts": [[0, 5, 6, 7, 9][step]],
        }
        for step in range(5)
    }
    (run / "controls.json").write_text(
        json.dumps(
            {
                "random_pair_control": {
                    "mapping_target_to_source": {
                        row["sample_id"]: next(
                            candidate["sample_id"]
                            for candidate in payloads
                            if candidate["finding"] == row["finding"]
                            and candidate["positive_votes"] == row["positive_votes"]
                            and candidate["block_id"] != row["block_id"]
                        )
                        for row in payloads
                    }
                },
                "length_control": {
                    "cross_claim_step_matches": length_matches,
                }
            }
        )
    )
    for index, payload in enumerate(payloads):
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        (shards / f"{index:04d}.json").write_text(
            json.dumps(
                {
                    "config_fingerprint": config["config_fingerprint"],
                    "payload_sha256": _sha(encoded),
                    "payload": payload,
                }
            )
        )
    output = tmp_path / "analysis.json"
    with pytest.raises(ContractError, match="F6|f6|forbidden"):
        analyze_run(
            run,
            output,
            bootstrap_replicates=100,
            seed=7,
            command=["frozen-analysis", "--seed", "7"],
        )
    assert not output.exists()


def test_exact_string_audit_rejects_v4_and_exposes_problematic_concatenations(tmp_path):
    rows = [
        {
            "finding": "pleural_effusion",
            "embedded_claim": "pleural effusion",
            "prefix_ladder": [
                {"step": 0, "prefix": ""},
                {"step": 1, "prefix": "The chest X-ray "},
            ],
        },
        {
            "finding": "lung_opacity",
            "embedded_claim": "opacity",
            "prefix_ladder": [
                {"step": 2, "prefix": "This chest X-ray shows no common abnormalities "},
                {
                    "step": 3,
                    "prefix": (
                        "This chest X-ray shows no common abnormalities such as consolidation, "
                        "effusion or pneumothorax. "
                    ),
                },
            ],
        },
    ]
    manifest = tmp_path / "v4.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest_protocol_id": "clinical-autoregressive-lockin-manifest-v4-claim-specific-prompt",
                "manifest_sha256": _sha(manifest.read_bytes()),
            }
        )
    )
    result = audit_legacy_v4(manifest, metadata)
    strings = {row["exact_concatenation"] for row in result["exact_serialized_stimuli"]}
    assert "The chest X-ray pleural effusion" in strings
    assert (
        "This chest X-ray shows no common abnormalities such as consolidation, "
        "effusion or pneumothorax. opacity"
    ) in strings
    assert result["status"] == "rejected_f6_construct_invalid"
    assert result["gpu_authorized"] is False


def _v5_contract_with_review(pilot):
    body = {
        "manifest_protocol_id": FUTURE_PROTOCOL,
        "measurement_mode": FUTURE_MODE,
        "fixed_continuation_across_manually_assembled_prefixes": False,
        "context_changes_across_token_positions_acknowledged": True,
        "prefix_policy": "tokenizer_boundaries_of_one_frozen_full_sequence_only",
        "stimuli": [
            {
                "finding": finding,
                "full_sequence": payload["text"],
                "full_sequence_sha256": _sha(payload["text"].encode()),
                "pilot_full_sequence_sha256": payload["text_sha256"],
                "manual_prefixes": [],
            }
            for finding, payload in sorted(pilot.items())
        ],
        "natural_control_sequences": [
            {
                "finding": finding,
                "full_sequence": f"A complete natural non-target control for {finding}.",
                "full_sequence_sha256": _sha(
                    f"A complete natural non-target control for {finding}.".encode()
                ),
                "role": "token_position_matched_non_target_natural_control",
                "manual_prefixes": [],
            }
            for finding in sorted(pilot)
        ],
    }
    review = {
        "reviewer_id": "independent-reviewer-1",
        "reviewer_role": "clinical_language_reviewer",
        "reviewed_contract_sha256": _sha(construct_canonical(body)),
        "all_full_sequences_natural": True,
        "proposition_leakage_assessed": True,
        "token_position_context_change_accepted": True,
        "approved_for_mechanistic_dev_only": True,
        "attestation": "I reviewed the exact complete sequences without using model scores.",
    }
    return {**body, "independent_construct_review": review}


def test_v5_requires_human_construct_admission_and_never_auto_authorizes_gpu():
    text = "The chest X-ray shows a right-sided pleural effusion."
    pilot = {
        "pleural_effusion": {
            "text": text,
            "text_sha256": _sha(text.encode()),
        }
    }
    contract = _v5_contract_with_review(pilot)
    missing_review = dict(contract)
    missing_review.pop("independent_construct_review")
    with pytest.raises(ConstructContractError, match="review"):
        validate_v5_contract(missing_review, pilot_exact_surfaces=pilot)
    admitted = validate_v5_contract(contract, pilot_exact_surfaces=pilot)
    assert admitted["status"] == "construct_admitted_cpu_only"
    assert admitted["gpu_authorized"] is False
    assert "runtime has not been implemented" in admitted["reason_gpu_still_false"]


def test_v5_refuses_manual_prefixes_even_with_positive_review():
    text = "A complete natural sequence."
    pilot = {"lung_opacity": {"text": text, "text_sha256": _sha(text.encode())}}
    contract = _v5_contract_with_review(pilot)
    contract["stimuli"][0]["manual_prefixes"] = ["A complete "]
    body = {key: value for key, value in contract.items() if key != "independent_construct_review"}
    contract["independent_construct_review"]["reviewed_contract_sha256"] = _sha(
        construct_canonical(body)
    )
    with pytest.raises(ConstructContractError, match="manually"):
        validate_v5_contract(contract, pilot_exact_surfaces=pilot)


def test_v5_refuses_missing_natural_position_control_before_review():
    text = "A complete natural sequence."
    pilot = {"lung_opacity": {"text": text, "text_sha256": _sha(text.encode())}}
    contract = _v5_contract_with_review(pilot)
    contract["natural_control_sequences"] = []
    body = {key: value for key, value in contract.items() if key != "independent_construct_review"}
    contract["independent_construct_review"]["reviewed_contract_sha256"] = _sha(
        construct_canonical(body)
    )
    with pytest.raises(ConstructContractError, match="control"):
        validate_v5_contract(contract, pilot_exact_surfaces=pilot)
