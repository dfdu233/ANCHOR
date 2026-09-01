#!/usr/bin/env python3
"""Outcome-blind scientific-worker boundary for the CECD envelope.

Implemented in this first part:

* real, CPU-only Huatuo/Hulu runtime descriptors;
* full input, checkpoint, processor, prompt, generation, hook, vision-token
  transport and source hash closure;
* conformance-tested architecture-neutral kernels for seven controls.

Not implemented: real Huatuo/Hulu arm adapters, both Treble semantic variants,
and CECD hidden intervention.  Formal arm execution fails before importing
Torch, loading a model, creating an output directory or touching CUDA.  This is
intentional: a mathematical kernel is not a scientifically valid model port.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from anchor.corrected_sgta.cecd_dual_semantics_kernels_v1 import (
    IMPLEMENTED_KERNEL_METHODS,
    synthetic_adapter_conformance,
)


VERSION = "cecd-dual-semantics-scientific-worker-v1"
RUNTIME_DESCRIPTOR_SCHEMA = "cecd-dual-semantics-runtime-descriptor-v1"
INPUT_SIDECAR_SCHEMA = "cecd-dual-semantics-input-bindings-v1"
ROOT = Path("/home/dbw/ANCHOR")
INPUT_NAMES = (
    "calibration_manifest",
    "evaluation_manifest",
    "record_keys",
    "claim_contract",
)
FORMAL_METHOD_STATUS = {
    method: "formal_centered_logit_ce_implemented_oe_not_implemented"
    for method in IMPLEMENTED_KERNEL_METHODS
}
FORMAL_METHOD_STATUS.update(
    {
        "cecd_interaction_projection": "method_not_implemented",
        "treble_proceedings": "method_not_implemented",
        "treble_released": "method_not_implemented",
    }
)


class ScientificWorkerError(RuntimeError):
    """Base fail-closed scientific-worker error."""


class MethodNotImplementedError(ScientificWorkerError):
    """Raised before model/GPU access for an unimplemented formal arm."""


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ScientificWorkerError(f"required regular file is missing or symlinked: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def relative_file_record(path: Path, root: Path) -> dict[str, Any]:
    record = file_record(path)
    record["path"] = str(path.resolve().relative_to(root.resolve()))
    return record


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScientificWorkerError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ScientificWorkerError(f"{label} must be a JSON object")
    return payload


def _sidecar_path(preflight: Path) -> Path:
    return preflight.with_name(f"{preflight.stem}.inputs.json")


def load_input_sidecar(preflight: Path) -> dict[str, Any]:
    sidecar_path = _sidecar_path(preflight)
    payload = load_object(sidecar_path, "dual-semantics input sidecar")
    required = {
        "schema_version",
        "preflight_sha256",
        "model_dirs",
        "huatuo_source_root",
        "input_bindings",
    }
    if set(payload) != required:
        raise ScientificWorkerError("input sidecar fields are not closed")
    if (
        payload["schema_version"] != INPUT_SIDECAR_SCHEMA
        or payload["preflight_sha256"] != sha256_file(preflight)
    ):
        raise ScientificWorkerError("input sidecar does not bind the current preflight")
    if not isinstance(payload["model_dirs"], Mapping) or set(payload["model_dirs"]) != {
        "huatuo",
        "hulu",
    }:
        raise ScientificWorkerError("input sidecar must bind exactly two model directories")
    if not isinstance(payload["input_bindings"], Mapping) or set(
        payload["input_bindings"]
    ) != set(INPUT_NAMES):
        raise ScientificWorkerError("input sidecar file closure is incomplete")
    return payload


def _inventory(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ScientificWorkerError(f"empty file inventory: {root}")
    return [relative_file_record(path, root) for path in unique]


def full_model_inventory(model_dir: Path) -> list[dict[str, Any]]:
    files = [path for path in model_dir.rglob("*") if path.is_file()]
    return _inventory(files, model_dir)


def _processor_paths(family: str, model_dir: Path) -> list[Path]:
    common = (
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    names = list(common)
    if family == "huatuo":
        names.extend(
            (
                "vit/clip_vit_large_patch14_336/config.json",
                "vit/clip_vit_large_patch14_336/preprocessor_config.json",
            )
        )
    else:
        names.extend(
            (
                "chat_template.json",
                "preprocessor_config.json",
                "processing_hulumed.py",
                "image_processing_hulumed.py",
            )
        )
    paths = [model_dir / name for name in names if (model_dir / name).is_file()]
    if not paths:
        raise ScientificWorkerError(f"{family} processor inventory is empty")
    return paths


def _load_config(model_dir: Path) -> dict[str, Any]:
    return load_object(model_dir / "config.json", "model config")


def prompt_template_contract(family: str, model_dir: Path) -> dict[str, Any]:
    tokenizer_config = (
        load_object(model_dir / "tokenizer_config.json", "tokenizer config")
        if (model_dir / "tokenizer_config.json").is_file()
        else {}
    )
    chat_template_file = model_dir / "chat_template.json"
    return {
        "version": "cecd-dual-semantics-prompt-template-contract-v1",
        "family": family,
        "polar_prompt_semantics": {
            "proposition": "present(finding)",
            "speech_act": "polar_diagnostic_question",
            "answer_space": ["Yes", "No", "Maybe"],
        },
        "oe_prompt_semantics": {
            "task": "fixed-candidate atomic-claim teacher forcing",
            "free_generation_claim_exchange_allowed": False,
        },
        "tokenizer_chat_template": tokenizer_config.get("chat_template"),
        "chat_template_file_sha256": (
            sha256_file(chat_template_file) if chat_template_file.is_file() else None
        ),
    }


def generation_contract() -> dict[str, Any]:
    return {
        "version": "cecd-dual-semantics-generation-contract-v1",
        "do_sample": False,
        "temperature": None,
        "ce_max_new_tokens": 1,
        "oe_mode": "teacher_forced_aligned_atomic_claims",
        "positive_claim_count_exchange_allowed": False,
        "length_exchange_allowed": False,
        "refusal_exchange_allowed": False,
    }


def hook_contract(family: str, model_dir: Path) -> dict[str, Any]:
    config = _load_config(model_dir)
    if family == "huatuo":
        vision_config = load_object(
            model_dir / "vit/clip_vit_large_patch14_336/config.json",
            "Huatuo vision config",
        )
        decoder_layers = int(config.get("num_hidden_layers", -1))
        nested_vision = vision_config.get("vision_config")
        vision_layers = int(
            vision_config.get(
                "num_hidden_layers",
                nested_vision.get("num_hidden_layers", -1)
                if isinstance(nested_vision, Mapping)
                else -1,
            )
        )
        paths = {
            "vision_mlp": "bot.model.get_vision_tower().vision_tower.vision_model.encoder.layers[*].mlp",
            "decoder_mlp": "bot.model.model.layers[*].mlp",
        }
    else:
        decoder_layers = int(config.get("num_hidden_layers", -1))
        vision = config.get("vision_encoder_config")
        vision_layers = int(vision.get("num_hidden_layers", -1)) if isinstance(vision, Mapping) else -1
        paths = {
            "vision_mlp": "runtime.model.get_vision_encoder().encoder.layers[*].mlp",
            "decoder_mlp": "runtime.model.model.layers[*].mlp",
        }
    if decoder_layers <= 0 or vision_layers <= 0:
        raise ScientificWorkerError(f"{family} layer counts are unavailable")
    return {
        "version": "cecd-dual-semantics-hook-contract-v1",
        "family": family,
        "vision_layers": vision_layers,
        "decoder_layers": decoder_layers,
        "paths": paths,
        "zero_shift_logit_identity_required": True,
        "hook_count_assertion_required": True,
        "real_hook_adapter_implemented": False,
    }


def vision_token_transport_contract(family: str) -> dict[str, Any]:
    if family == "huatuo":
        return {
            "version": "cecd-vision-token-transport-v1",
            "family": family,
            "grid": "fixed_clip_vit_l14_336",
            "transport": "same_token_index_only_after_exact_grid_assertion",
            "treble_visual_adapter_implemented": False,
        }
    return {
        "version": "cecd-vision-token-transport-v1",
        "family": family,
        "grid": "adaptive_hulumed_grid",
        "transport": "not_admitted_not_implemented",
        "forbidden_substitutions": [
            "silent_square_resize",
            "unregistered_token_interpolation",
            "pooled_direction_relabelled_as_source_faithful_treble",
        ],
        "treble_visual_adapter_implemented": False,
    }


def compute_model_fingerprint(family: str, model_dir: Path) -> dict[str, Any]:
    if family not in {"huatuo", "hulu"} or not model_dir.is_dir():
        raise ScientificWorkerError(f"invalid {family} model directory: {model_dir}")
    checkpoint_inventory = full_model_inventory(model_dir)
    processor_inventory = _inventory(_processor_paths(family, model_dir), model_dir)
    return {
        "model_id": f"{family}:{model_dir.name}",
        "checkpoint_sha256": canonical_sha256(checkpoint_inventory),
        "processor_sha256": canonical_sha256(processor_inventory),
        "template_sha256": canonical_sha256(
            prompt_template_contract(family, model_dir)
        ),
        "generation_contract_sha256": canonical_sha256(generation_contract()),
        "hook_contract_sha256": canonical_sha256(hook_contract(family, model_dir)),
        "vision_token_transport_contract_sha256": canonical_sha256(
            vision_token_transport_contract(family)
        ),
    }


def _source_closure(
    *, family: str, model_dir: Path, huatuo_source_root: Path, sidecar_path: Path
) -> list[dict[str, Any]]:
    local = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        local / "cecd_dual_semantics_kernels_v1.py",
        local / "cecd_dual_semantics_ce_adapter_v1.py",
        local / "run_cecd_dual_semantics_controlled_v1.py",
        local / "treble_collision_contract.py",
        sidecar_path,
    ]
    if family == "huatuo":
        paths.extend(path for path in huatuo_source_root.rglob("*.py") if path.is_file())
    else:
        paths.extend(path for path in model_dir.glob("*.py") if path.is_file())
    records = [file_record(path) for path in sorted({path.resolve() for path in paths})]
    worker_path = str(Path(__file__).resolve())
    if sum(record["path"] == worker_path for record in records) != 1:
        raise ScientificWorkerError("worker must occur exactly once in source closure")
    return records


def runtime_versions() -> dict[str, str]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "numpy": version("numpy"),
        "pillow": version("pillow"),
        "cuda_initialized": "false_descriptor_is_cpu_only",
    }


def describe_runtime(*, family: str, preflight_path: Path) -> dict[str, Any]:
    preflight = load_object(preflight_path, "dual-semantics preflight")
    sidecar = load_input_sidecar(preflight_path)
    model_dir = Path(str(sidecar["model_dirs"][family])).resolve()
    huatuo_source_root = Path(str(sidecar["huatuo_source_root"])).resolve()
    actual_fingerprint = compute_model_fingerprint(family, model_dir)
    expected = preflight.get("model_fingerprints", {}).get(family)
    if actual_fingerprint != expected:
        changed = sorted(
            key
            for key in set(actual_fingerprint) | set(expected or {})
            if actual_fingerprint.get(key) != (expected or {}).get(key)
        )
        raise ScientificWorkerError(
            f"{family} real model fingerprint disagrees with preflight: {changed}"
        )
    inputs: dict[str, Any] = {}
    expected_hash_fields = {
        "calibration_manifest": "calibration_manifest_sha256",
        "evaluation_manifest": "evaluation_manifest_sha256",
        "record_keys": "record_keys_sha256",
        "claim_contract": "claim_contract_sha256",
    }
    for name, field in expected_hash_fields.items():
        record = file_record(Path(str(sidecar["input_bindings"][name])))
        if record["sha256"] != preflight.get(field):
            raise ScientificWorkerError(f"{family} {name} hash disagrees with preflight")
        inputs[name] = record
    return {
        "schema_version": RUNTIME_DESCRIPTOR_SCHEMA,
        "model_family": family,
        "model_id": actual_fingerprint["model_id"],
        "model_fingerprint": actual_fingerprint,
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_versions": runtime_versions(),
        "source_files": _source_closure(
            family=family,
            model_dir=model_dir,
            huatuo_source_root=huatuo_source_root,
            sidecar_path=_sidecar_path(preflight_path),
        ),
        "input_bindings": inputs,
    }


def validate_formal_request(
    *,
    authorization: Path,
    preflight: Path,
    run_contract: Path,
    family: str,
    method: str,
    output_dir: Path,
    task: str = "full",
) -> dict[str, Any]:
    # Import only the CPU-side validator after all CLI fields are present. It
    # hashes authorization/preflight inputs but does not parse sealed outcomes.
    from anchor.corrected_sgta.run_cecd_dual_semantics_controlled_v1 import (
        load_object as load_runner_object,
        sha256_file as runner_sha256_file,
        validate_authorization_and_preflight,
    )

    authorization_payload, preflight_payload, output_root = (
        validate_authorization_and_preflight(
            authorization_path=authorization,
            preflight_path=preflight,
            root=ROOT,
        )
    )
    contract = load_runner_object(run_contract, "write-once run contract")
    contract_fingerprint = contract.get("fingerprint")
    body = {key: value for key, value in contract.items() if key != "fingerprint"}
    if canonical_sha256(body) != contract_fingerprint:
        raise ScientificWorkerError("run-contract fingerprint mismatch")
    if (
        contract.get("authorization_fingerprint")
        != authorization_payload.get("fingerprint")
        or contract.get("preflight", {}).get("sha256") != runner_sha256_file(preflight)
        or contract.get("method_output_root") != str(output_root)
        or family not in contract.get("models", {})
        or method not in preflight_payload.get("methods", [])
        or {item.get("model_family") for item in contract.get("arm_order", [])}
        != {"huatuo", "hulu"}
    ):
        raise ScientificWorkerError("formal arm is outside the frozen run contract")
    resolved_output = output_dir.resolve()
    if output_root not in resolved_output.parents:
        raise ScientificWorkerError("formal output directory escapes method-output root")
    status = FORMAL_METHOD_STATUS.get(method, "method_not_implemented")
    if status == "method_not_implemented":
        raise MethodNotImplementedError(
            f"{method}: strict scientific implementation is absent; refusing before model/GPU/output"
        )
    if task == "ce":
        return {
            "authorization": authorization_payload,
            "preflight": preflight_payload,
            "run_contract": contract,
            "output_root": output_root,
        }
    raise MethodNotImplementedError(
        f"{method}: formal centered-logit CE adapter exists, but aligned OE is not "
        "implemented; refusing a full arm before model/GPU/output"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-runtime", action="store_true")
    parser.add_argument("--synthetic-conformance", action="store_true")
    parser.add_argument("--model-family", choices=("huatuo", "hulu"))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task", choices=("ce", "full"), default="full")
    parser.add_argument("--shared-cache-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    modes = int(args.describe_runtime) + int(args.synthetic_conformance)
    if modes > 1:
        raise ScientificWorkerError("descriptor and synthetic modes are mutually exclusive")
    if args.synthetic_conformance:
        result = synthetic_adapter_conformance(args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["passed"] is not True:
            raise ScientificWorkerError("synthetic factorial conformance failed")
        return
    if args.describe_runtime:
        if args.model_family is None or args.preflight is None:
            raise ScientificWorkerError("runtime descriptor requires model family and preflight")
        print(
            json.dumps(
                describe_runtime(
                    family=args.model_family, preflight_path=args.preflight
                ),
                sort_keys=True,
            )
        )
        return
    required = {
        "model_family": args.model_family,
        "preflight": args.preflight,
        "authorization": args.authorization,
        "run_contract": args.run_contract,
        "method": args.method,
        "output_dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ScientificWorkerError(f"formal arm request is missing: {missing}")
    context = validate_formal_request(
        authorization=args.authorization,
        preflight=args.preflight,
        run_contract=args.run_contract,
        family=args.model_family,
        method=args.method,
        output_dir=args.output_dir,
        task=args.task,
    )
    if args.task != "ce":
        raise AssertionError("full task must fail before this point")
    if args.shared_cache_root is None:
        raise ScientificWorkerError("formal CE task requires --shared-cache-root")
    output_root = context["output_root"]
    if output_root not in args.shared_cache_root.resolve().parents:
        raise ScientificWorkerError("shared CE cache escapes method-output root")
    sidecar = load_input_sidecar(args.preflight)
    model_dir = Path(str(sidecar["model_dirs"][args.model_family])).resolve()
    bindings = context["run_contract"]["runtime_descriptors"][args.model_family][
        "input_bindings"
    ]
    from anchor.corrected_sgta.cecd_dual_semantics_ce_adapter_v1 import (
        run_formal_ce_method,
    )

    result = run_formal_ce_method(
        family=args.model_family,
        method=args.method,
        model_dir=model_dir,
        evaluation_manifest=Path(bindings["evaluation_manifest"]["path"]),
        record_keys_path=Path(bindings["record_keys"]["path"]),
        claim_contract_path=Path(bindings["claim_contract"]["path"]),
        run_contract=context["run_contract"],
        shared_cache_root=args.shared_cache_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
