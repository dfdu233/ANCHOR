#!/usr/bin/env python3
"""Outcome-blind compatibility gate for two 2026 attention controls.

This module does not run either model and never reads experimental outcomes.
It freezes the boundary between an official-paper implementation and an
independent, architecture-neutral control for:

* System-Mediated Attention Imbalances Make VLMs Say Yes; and
* Mechanisms of Prompt-Induced Hallucination in VLMs.

The gate is intentionally separate from the CECD workers and three-stage
decision thresholds.  A passing contract authorizes only a common-protocol
control run, never a claim of paper-native reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "cecd-system-pih-control-preflight-v1"
ROOT = Path("/home/dbw/ANCHOR")
HEX64 = set("0123456789abcdef")
MODEL_FAMILIES = ("huatuo", "hulu")

PAPERS = {
    "system_mediated_attention": {
        "paper_id": "acl:2026.findings-acl.1940",
        "paper_url": "https://aclanthology.org/2026.findings-acl.1940/",
        "paper_pdf_sha256": "dc1a8486fe0fb55d891c17af91f10a5df6be841e7af3e35594f9b160d3db52ba",
        "official_repository": "https://github.com/anisha0325/vlm-hallucination-yes-bias",
        "official_commit": "343ad9da36bd23e555816b7531da626234aeda22",
        "official_tree_sha256": "1d4a13510262d0e2539d28eb47d39757a1f49471d13a663b4be2bcec200aeb6d",
        "license_status": "no_license_file_or_explicit_grant_at_frozen_commit",
        "official_model_scope": ["llava-v1.5-7b"],
        "official_source_hashes": {
            "README.md": "bf8bc832a85357136c01c03f5554f150f02ebfcf8f7bd77c7e1ce77bcc34268b",
            "LLaVA/llava/model/language_model/modeling_llama.py": "0fc835999482c3b026db801c0e15903ef62a929a9c34558f5485978fbad41c17",
            "LLaVA/eval_scripts/analyze_attention_reweight_matrices_fm_sys.py": "9fc9e1a9b01588fd1fb636605c1803df57fdbebe9adb9cc4beaf14e7d46f1717",
            "LLaVA/results/coco/llava_3000/identify_attention_head/layers25to32_llava157b.json": "1530d06d9a6990261292e6bfb08a31dc24fe10bf57bb88e81745857d4f74a9d0",
            "LLaVA/bash_scripts/y25_fm_sys.sh": "73f556e1cb6d90ad3d7cb410fce01fe6fb96a3a7fbeb3dc4fba82b4ee72dd3c0",
        },
        "release_defects": [
            "tracked_shell_entrypoint_references_untracked_all_heads_llava157b_json",
            "readme_clone_target_differs_from_acl_linked_author_repository",
        ],
    },
    "prompt_induced_hallucination": {
        "paper_id": "acl:2026.acl-long.1941",
        "paper_url": "https://aclanthology.org/2026.acl-long.1941/",
        "paper_pdf_sha256": "e4036d8fdfc82abd8b13a53084942562b098a82a55fbcdd67eac441b7691bac4",
        "official_repository": "https://github.com/michalg04/prompt-induced_hallucinations",
        "official_commit": "f301d68ed3743417c29c384c632838523610c9c4",
        "official_tree_sha256": "cfd1f2983c760b23f757a0edcb1919d51403c3df33e46b2fc240f9dd52c5c73f",
        "license_status": "no_license_file_or_explicit_grant_at_frozen_commit",
        "official_model_scope": [
            "Qwen2-VL-7B-Instruct",
            "LLaVA-OneVision-Qwen2-7B",
            "Janus-Pro-7B",
        ],
        "official_source_hashes": {
            "README.md": "d5060e617506c17b5508c5216678988772222fbed26a056df37298ce2b18dc05",
            "knockout_utils.py": "1e4cb59172100e4826ed51d1f6725edd14e674f681a0269c5a64b56e43a1dc09",
            "3_knockouts.py": "9dad1bc6e635475aae108c7c61b8a9dc0483c75224520e46c80b9acc83ab0d03",
            "4_evaluate_knockouts.py": "c16ff61656de3baacbf7a47902616012124e1dab742c2ce41e1fb71fcbb7f145",
            "5_attention_mass.py": "5d1866b277f999281f93480c686ce2e033ff97feedf0c5648264f8b54cf1758e",
            "requirements.txt": "73bf4de4e24152da6efaa02e01f05a5b36a69c07350077151ad9095602a22b8d",
        },
        "release_defects": [
            "official_head_discovery_and_reporting_do_not_freeze_an_independent_test_split",
            "official_mean_hook_averages_across_batch_and_tokens",
        ],
    },
}

ARCHITECTURES = {
    "huatuo": {
        "model_class": "LlavaQwen2ForCausalLM",
        "decoder_class": "Qwen2Model",
        "transformers_version": "4.37.2",
        "decoder_layers_path": "model.model.layers",
        "attention_path_template": "model.model.layers[{layer}].self_attn",
        "o_proj_path_template": "model.model.layers[{layer}].self_attn.o_proj",
        "num_hidden_layers": 28,
        "num_query_heads": 28,
        "num_key_value_heads": 4,
        "hidden_size": 3584,
        "attention_output_width": 3584,
        "head_dim": 128,
        "native_attention_backend": "eager",
        "visual_token_semantics": "one_negative_placeholder_expands_to_fixed_clip_patch_sequence",
        "observed_primary_visual_length_contract": 576,
        "locked_prompt_has_true_system_role": False,
        "pre_image_content": "huatuo_user_role_delimiter_only",
    },
    "hulu": {
        "model_class": "HulumedQwen3ForCausalLM",
        "decoder_class": "Qwen3Model",
        "transformers_version": "4.51.2",
        "decoder_layers_path": "model.model.layers",
        "attention_path_template": "model.model.layers[{layer}].self_attn",
        "o_proj_path_template": "model.model.layers[{layer}].self_attn.o_proj",
        "num_hidden_layers": 36,
        "num_query_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 2560,
        "attention_output_width": 4096,
        "head_dim": 128,
        "native_attention_backend": "sdpa",
        "visual_token_semantics": "processor_materializes_variable_contiguous_image_token_run_before_embedding_replacement",
        "observed_primary_visual_length_contract": "dynamic_from_image_token_run",
        "locked_prompt_has_true_system_role": False,
        "pre_image_content": "qwen_user_role_delimiter_only",
    },
}

SPAN_CONTRACT = {
    "source_of_truth": "expanded_prefix_token_provenance_after_multimodal_preparation",
    "required_role_labels": ["system", "image", "user_text"],
    "role_aware_partition_must_be_disjoint_and_exhaustive": True,
    "image_span_must_be_nonempty_and_contiguous": True,
    "padding_excluded": True,
    "generated_tokens_excluded_from_frozen_prefix_spans": True,
    "official_positional_surrogate": {
        "prefix_before_image": "all_expanded_prefix_keys_strictly_before_image",
        "image": "expanded_visual_keys_only",
        "suffix_after_image": "all_expanded_prefix_keys_strictly_after_image",
        "may_be_named_system_only_if_all_prefix_before_image_has_system_role": True,
    },
    "forbid_magic_boundaries": [35, 576],
}

SYSTEM_CONTROL = {
    "official_semantics": {
        "hook": "post_softmax_pre_value_matmul_attention_weights",
        "source": "official_positional_prefix_before_image",
        "alpha": 0.0,
        "recipients": ["image", "suffix_after_image"],
        "recipient_modality_mass": "proportional_to_original_rowwise_modality_mass",
        "within_recipient_modality": "uniform_per_token_increment",
        "second_softmax": False,
        "official_layers_zero_indexed": list(range(24, 32)),
        "official_heads": "all_32_query_heads",
        "official_magic_image_start": 35,
        "official_magic_image_length": 576,
    },
    "common_protocol": {
        "name": "pre_image_prefix_attention_redistribution_control",
        "paper_native": False,
        "official_code_port": False,
        "hook": "inside_eager_attention_after_fp32_softmax_before_value_matmul",
        "attention_backend": "eager_required",
        "query_rows": "last_frozen_prefix_query_only",
        "source": "prefix_before_image_never_relabel_as_system_without_role_proof",
        "alpha": 0.0,
        "recipients": ["image", "suffix_after_image"],
        "recipient_modality_mass": "proportional_to_original_rowwise_modality_mass",
        "within_recipient_modality": "uniform_per_token_increment",
        "eligible_row_rule": "both_source_and_combined_recipient_mass_finite_and_recipient_mass_gt_1e-6",
        "mass_conservation_absolute_tolerance": 1e-6,
        "primary_layers": {
            "huatuo": list(range(21, 28)),
            "hulu": list(range(27, 36)),
        },
        "primary_heads": "all_query_heads_no_outcome_based_selection",
        "required_controls": [
            "identity_alpha_one",
            "source_zero_without_redistribution",
            "prefix_to_image_only",
            "prefix_to_text_only",
            "random_equal_width_key_span",
            "native_vs_eager_first_token_logit_canary",
        ],
        "interpretation_boundary": "positional_prefix_control_not_system_instruction_mechanism",
    },
}

PIH_CONTROL = {
    "official_semantics": {
        "hook": "forward_pre_hook_on_attention_o_proj",
        "tensor": "concatenated_query_head_outputs_B_T_H_before_o_proj",
        "replacement": "selected_head_replaced_by_its_token_mean",
        "official_code_reduction_axes": ["batch", "token"],
        "official_candidate_group_sizes": [1, 3, 5, 10],
        "official_random_control": "same_layer_distribution_and_same_head_count",
    },
    "common_protocol": {
        "name": "dev_selected_prompt_copy_head_mean_ablation_control",
        "paper_native": False,
        "official_code_port": False,
        "hook": "forward_pre_hook_on_attention_o_proj",
        "head_width_source": "o_proj_in_features_divided_by_num_query_heads",
        "require_exact_divisibility": True,
        "replacement": "per_sample_mean_over_frozen_prefix_token_axis_only",
        "batch_size": 1,
        "cross_sample_mean_forbidden": True,
        "candidate_group_sizes": [1, 3, 5, 10],
        "selection_split": "patient_or_image_disjoint_dev_only",
        "selection_unit": "model_specific_layer_query_head_pair",
        "selection_score": "reader_grounded_prompt_copy_correction_subject_to_aligned_prompt_non_degradation",
        "head_sets_reused_across_models": False,
        "locked_test_scanned_during_selection": False,
        "random_control": "same_model_same_selected_layer_multiset_same_head_count_seed_locked",
        "required_artifacts": [
            "dev_manifest_hash",
            "candidate_sweep_hash",
            "ranked_head_table_hash",
            "selected_head_set_hash",
            "random_head_set_hash",
        ],
        "attention_mass_diagnostic": "last_frozen_prefix_query_role_aware_system_image_user_text_partition",
    },
}

FIDELITY = {
    "paper_native_reproduction": {"huatuo": False, "hulu": False},
    "official_code_faithful_port": {"huatuo": False, "hulu": False},
    "system_common_protocol_feasibility": {
        "huatuo": "source_patch_feasible_but_true_system_source_absent",
        "hulu": "source_patch_feasible_but_true_system_source_absent_and_backend_must_change_to_eager",
    },
    "pih_common_protocol_feasibility": {
        "huatuo": "mechanically_feasible_with_model_specific_dev_head_discovery",
        "hulu": "mechanically_feasible_only_with_o_proj_width_not_hidden_size",
    },
    "allowed_claim": "independent_architecture_neutral_control_inspired_by_the_papers",
    "forbidden_claims": [
        "paper_native",
        "official_implementation",
        "faithful_official_code_port",
        "system_attention_mechanism_when_source_span_has_no_true_system_tokens",
    ],
}

RUNTIME_INTEGRATION_CONTRACT = {
    "generic_tensor_bindings_do_not_authorize_model_execution": True,
    "required_models": list(MODEL_FAMILIES),
    "per_model_required_artifacts": [
        "system_attention_runtime_patch",
        "native_eager_canary_artifact",
        "pih_o_proj_runtime_integration",
    ],
    "native_eager_canary_must_precede_intervention": True,
    "hulu_runtime_backend_for_system_control": "eager",
    "selected_head_artifacts_are_separately_required": True,
}


class PreflightError(ValueError):
    """Raised when the frozen compatibility contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CONTRACT_FINGERPRINTS = {
    "papers": canonical_sha256(PAPERS),
    "architectures": canonical_sha256(ARCHITECTURES),
    "span_contract": canonical_sha256(SPAN_CONTRACT),
    "system_control": canonical_sha256(SYSTEM_CONTROL),
    "pih_control": canonical_sha256(PIH_CONTROL),
    "fidelity": canonical_sha256(FIDELITY),
    "runtime_integration": canonical_sha256(RUNTIME_INTEGRATION_CONTRACT),
}


def _hex64(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and set(text) <= HEX64


def dynamic_expanded_partition(
    role_labels: Sequence[str], *, image_start: int, image_end: int
) -> dict[str, tuple[int, ...]]:
    """Validate and return a role-aware partition of one unpadded prefix.

    ``role_labels`` must already describe the expanded multimodal sequence.
    This deliberately cannot infer roles from magic token positions.
    """

    labels = tuple(str(value) for value in role_labels)
    if not labels:
        raise PreflightError("expanded prefix cannot be empty")
    if not (0 <= image_start < image_end <= len(labels)):
        raise PreflightError("image span must be nonempty and in bounds")
    allowed = set(SPAN_CONTRACT["required_role_labels"])
    if set(labels) - allowed:
        raise PreflightError("expanded prefix contains an unknown role label")
    expected_image = set(range(image_start, image_end))
    actual_image = {index for index, label in enumerate(labels) if label == "image"}
    if actual_image != expected_image:
        raise PreflightError("image labels must form exactly one contiguous dynamic span")
    partition = {
        role: tuple(index for index, label in enumerate(labels) if label == role)
        for role in SPAN_CONTRACT["required_role_labels"]
    }
    union = set().union(*(set(indices) for indices in partition.values()))
    if union != set(range(len(labels))):
        raise PreflightError("role-aware spans are not exhaustive")
    if sum(len(indices) for indices in partition.values()) != len(union):
        raise PreflightError("role-aware spans overlap")
    return partition


def positional_surrogate_spans(
    prefix_length: int, *, image_start: int, image_end: int
) -> dict[str, tuple[int, ...]]:
    if not (0 <= image_start < image_end <= prefix_length):
        raise PreflightError("invalid dynamic image span")
    return {
        "prefix_before_image": tuple(range(image_start)),
        "image": tuple(range(image_start, image_end)),
        "suffix_after_image": tuple(range(image_end, prefix_length)),
    }


def derive_query_head_width(*, o_proj_in_features: int, num_query_heads: int) -> int:
    if o_proj_in_features <= 0 or num_query_heads <= 0:
        raise PreflightError("attention output width and query-head count must be positive")
    if o_proj_in_features % num_query_heads:
        raise PreflightError("o_proj input width is not divisible by query-head count")
    return o_proj_in_features // num_query_heads


def _validate_file_record(record: Any, *, root: Path, label: str) -> list[str]:
    expected = {"path", "sha256", "bytes"}
    if not isinstance(record, Mapping) or set(record) != expected:
        raise PreflightError(f"{label}: file record schema drift")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = root / path
    blockers: list[str] = []
    if not path.is_file():
        blockers.append(f"{label}:missing:{path}")
        return blockers
    if not _hex64(record["sha256"]) or sha256_file(path) != record["sha256"]:
        blockers.append(f"{label}:sha256_mismatch:{path}")
    if record["bytes"] != path.stat().st_size:
        blockers.append(f"{label}:size_mismatch:{path}")
    return blockers


def _validate_selection(record: Any, *, family: str) -> list[str]:
    expected = {
        "status",
        "selection_split",
        "locked_test_scanned",
        "head_width",
        "selected_heads_artifact",
        "random_heads_artifact",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise PreflightError(f"{family}: PIH selection schema drift")
    blockers: list[str] = []
    if record["selection_split"] != "patient_or_image_disjoint_dev_only":
        blockers.append(f"{family}:selection_not_dev_only")
    if record["locked_test_scanned"] is not False:
        blockers.append(f"{family}:locked_test_scanned")
    expected_width = derive_query_head_width(
        o_proj_in_features=ARCHITECTURES[family]["attention_output_width"],
        num_query_heads=ARCHITECTURES[family]["num_query_heads"],
    )
    if record["head_width"] != expected_width:
        blockers.append(f"{family}:wrong_query_head_width")
    if record["status"] != "ready":
        blockers.append(f"{family}:pih_selection_not_ready")
    for name in ("selected_heads_artifact", "random_heads_artifact"):
        artifact = record[name]
        if record["status"] == "ready" and artifact is None:
            blockers.append(f"{family}:{name}_missing")
    return blockers


def _validate_runtime_integration(
    record: Any, *, family: str, root: Path
) -> list[str]:
    expected = {
        "status",
        "system_attention_runtime_patch",
        "native_eager_canary_artifact",
        "pih_o_proj_runtime_integration",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise PreflightError(f"{family}: runtime integration schema drift")
    blockers: list[str] = []
    if record["status"] != "ready":
        blockers.append(f"{family}:runtime_integration_not_ready")
    for name in RUNTIME_INTEGRATION_CONTRACT["per_model_required_artifacts"]:
        artifact = record[name]
        if artifact is None:
            blockers.append(f"{family}:{name}_missing")
        else:
            blockers.extend(
                _validate_file_record(
                    artifact,
                    root=root,
                    label=f"{family}:runtime:{name}",
                )
            )
    return blockers


def validate_plan(payload: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    required = {
        "schema_version",
        "frozen_before_control_outputs",
        "control_outputs_consumed",
        "gpu_used",
        "contract_fingerprints",
        "architecture_source_files",
        "implementation_bindings",
        "runtime_integrations",
        "pih_selection",
        "control_output_root",
    }
    if set(payload) != required:
        raise PreflightError(
            f"preflight fields missing={sorted(required-set(payload))} "
            f"extra={sorted(set(payload)-required)}"
        )
    if payload["schema_version"] != VERSION:
        raise PreflightError("schema version mismatch")
    if payload["frozen_before_control_outputs"] is not True:
        raise PreflightError("contract must be frozen before control outputs")
    if payload["control_outputs_consumed"] is not False or payload["gpu_used"] is not False:
        raise PreflightError("compatibility audit must be outcome-blind and CPU/source-only")
    if payload["contract_fingerprints"] != CONTRACT_FINGERPRINTS:
        raise PreflightError("paper/method/architecture contract drift")

    blockers: list[str] = []
    source_files = payload["architecture_source_files"]
    if not isinstance(source_files, Mapping) or set(source_files) != set(MODEL_FAMILIES):
        raise PreflightError("architecture source closure drift")
    for family, records in source_files.items():
        if not isinstance(records, list) or not records:
            blockers.append(f"{family}:architecture_source_not_hash_bound")
            continue
        for index, record in enumerate(records):
            blockers.extend(
                _validate_file_record(record, root=root, label=f"{family}:source:{index}")
            )

    bindings = payload["implementation_bindings"]
    expected_bindings = {
        "dynamic_span_builder",
        "system_attention_source_patch",
        "system_numerical_canary",
        "pih_mean_ablation_hook",
        "pih_selection_runner",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise PreflightError("implementation binding closure drift")
    for name, record in bindings.items():
        if record is None:
            blockers.append(f"implementation_not_bound:{name}")
        else:
            blockers.extend(_validate_file_record(record, root=root, label=f"binding:{name}"))

    runtime_integrations = payload["runtime_integrations"]
    if not isinstance(runtime_integrations, Mapping) or set(runtime_integrations) != set(
        MODEL_FAMILIES
    ):
        raise PreflightError("runtime integration model closure drift")
    for family, record in runtime_integrations.items():
        blockers.extend(
            _validate_runtime_integration(record, family=family, root=root)
        )

    selections = payload["pih_selection"]
    if not isinstance(selections, Mapping) or set(selections) != set(MODEL_FAMILIES):
        raise PreflightError("PIH selection model closure drift")
    for family, record in selections.items():
        blockers.extend(_validate_selection(record, family=family))

    ready_selected = {
        family: selections[family]["selected_heads_artifact"]
        for family in MODEL_FAMILIES
        if selections[family]["status"] == "ready"
    }
    if len(ready_selected) == 2 and ready_selected["huatuo"] == ready_selected["hulu"]:
        blockers.append("pih_head_set_reused_across_models")

    for family in MODEL_FAMILIES:
        architecture = ARCHITECTURES[family]
        if architecture["locked_prompt_has_true_system_role"] is not False:
            raise PreflightError("audited locked-prompt system-role status drift")
        if derive_query_head_width(
            o_proj_in_features=architecture["attention_output_width"],
            num_query_heads=architecture["num_query_heads"],
        ) != architecture["head_dim"]:
            raise PreflightError(f"{family}: frozen head geometry is inconsistent")

    output_root = Path(str(payload["control_output_root"]))
    if not output_root.is_absolute():
        output_root = root / output_root
    if output_root.exists() and any(output_root.iterdir()):
        blockers.append("control_output_root_not_empty_preflight_too_late")

    return {
        "schema_version": VERSION,
        "passed": not blockers,
        "control_execution_ready": not blockers,
        "paper_native_reproduction_authorized": False,
        "official_code_port_authorized": False,
        "common_protocol_label_required": True,
        "generic_tensor_components_bound": not any(
            blocker.startswith("implementation_not_bound:")
            or ":missing:" in blocker
            or ":sha256_mismatch:" in blocker
            or ":size_mismatch:" in blocker
            for blocker in blockers
            if blocker.startswith("binding:")
            or blocker.startswith("implementation_not_bound:")
        ),
        "per_model_runtime_integration_ready": not any(
            ":runtime_integration_not_ready" in blocker
            or ":system_attention_runtime_patch_missing" in blocker
            or ":native_eager_canary_artifact_missing" in blocker
            or ":pih_o_proj_runtime_integration_missing" in blocker
            for blocker in blockers
        ),
        "true_system_attention_test_available_on_locked_prompts": False,
        "three_stage_thresholds_modified": False,
        "worker_or_listing_modified": False,
        "blockers": sorted(set(blockers)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    result = validate_plan(payload)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
