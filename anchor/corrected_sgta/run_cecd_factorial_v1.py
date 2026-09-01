#!/usr/bin/env python3
"""Crash-safe Clinical-Equivalence Composition Defect (CECD) factorial.

This runner deliberately measures a discrete render-by-wording ANOVA
interaction.  It does *not* call that quantity a mathematical commutator.  The
scientific cells cross five frozen DICOM renders with three proposition- and
speech-act-preserving polar questions.  Exact image and prompt duplicates are
engineering controls and never enter the scientific interaction.

Until the render families and question templates receive the preregistered
blinded clinical/language admission, every artifact produced here is an
engineering screen only.  In particular, a passing pixel guard is not evidence
of clinical equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    BASELINE_VIEW,
    balanced_rows,
    build_render_views,
    canonical_json_sha256,
    model_artifact_fingerprint,
    read_dicom_pixels,
    safe_record_key,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    atomic_json,
    import_huatuo,
    resolve_image,
    sha256_file,
)


VERSION = "clinical-equivalence-composition-defect-factorial-v1"
MEASUREMENT_NAME = "discrete_render_by_prompt_anova_interaction"
DEFAULT_MANIFEST = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "manifests_v2/reader_vote_manifest_v2.jsonl"
)
DEFAULT_BBOXES = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "manifests_v2/bbox_annotations_v2.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata/train")
FROZEN_FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "pleural_effusion",
    "pulmonary_fibrosis",
)
FROZEN_VOTES = (0, 1, 2, 3)
FROZEN_PER_BIN = 10
FROZEN_SEED = 42
STAGE_SPECS = {
    "pilot_screen": {"manifest_split": "pilot", "per_bin": 10, "claims": 160},
    "dev_fit": {"manifest_split": "dev", "per_bin": 20, "claims": 320},
    "confirmation_locked": {
        "manifest_split": "confirmation", "per_bin": 60, "claims": 960,
    },
}
SCIENCE_RENDER_NAMES = (
    BASELINE_VIEW,
    "native_linear",
    "center_minus_0p05w",
    "center_plus_0p05w",
    "width_x1p25",
)
IDENTITY_RENDER_NAME = "identity_lossless_duplicate"
PROMPT_SUFFIX = "Answer with exactly one word: Yes, No, or Maybe."
PROMPT_TEMPLATES = (
    (
        "existential",
        "Is there {finding} on this chest X-ray? " + PROMPT_SUFFIX,
    ),
    (
        "radiograph_subject",
        "Does this chest X-ray show {finding}? " + PROMPT_SUFFIX,
    ),
    (
        "visibility",
        "Can {finding} be seen on this chest X-ray? " + PROMPT_SUFFIX,
    ),
)
DUPLICATE_PROMPT_NAME = "existential_exact_duplicate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def readable_finding(finding: str) -> str:
    return str(finding).replace("_", " ")


def individual_reader_votes(row: Mapping[str, Any]) -> list[int]:
    """Normalize the manifest's audited per-reader records to binary votes."""

    raw = row.get("reader_votes", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("reader_votes must be a sequence")
    values: list[int] = []
    for item in raw:
        value = item.get("vote") if isinstance(item, Mapping) else item
        try:
            vote = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid individual reader vote: {item!r}") from error
        if vote not in (0, 1):
            raise ValueError(f"individual reader vote must be binary, got {vote}")
        values.append(vote)
    expected_count = int(row.get("reader_count", len(values)))
    expected_positive = int(row["positive_votes"])
    if len(values) != expected_count or sum(values) != expected_positive:
        raise ValueError(
            "individual reader votes disagree with manifest summary: "
            f"n={len(values)}/{expected_count}, positives={sum(values)}/{expected_positive}"
        )
    return values


def prompts_for(finding: str) -> list[dict[str, str]]:
    phrase = readable_finding(finding)
    prompts = []
    for name, template in PROMPT_TEMPLATES:
        prompt = template.format(finding=phrase)
        if prompt.count(phrase) != 1 or not prompt.endswith(PROMPT_SUFFIX):
            raise RuntimeError(f"invalid frozen prompt realization: {name}={prompt!r}")
        prompts.append(
            {
                "name": name,
                "text": prompt,
                "proposition": f"present({finding})",
                "speech_act": "polar_diagnostic_question",
            }
        )
    if len({item["text"] for item in prompts}) != len(prompts):
        raise RuntimeError("frozen prompt templates are not distinct")
    return prompts


def signed_int_sequence_sha256(values: Sequence[int]) -> str:
    encoded = json.dumps([int(value) for value in values], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_audit(tokenizer: Any) -> dict[str, Any]:
    verbalizer_ids: dict[str, int] = {}
    for state, text in VERBALIZERS.items():
        ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        if len(ids) != 1:
            raise ValueError(f"verbalizer must be one token: {text!r} -> {ids}")
        verbalizer_ids[state] = ids[0]
    if len(set(verbalizer_ids.values())) != 3:
        raise ValueError(f"verbalizers do not map to three distinct tokens: {verbalizer_ids}")

    rows = []
    for finding in FROZEN_FINDINGS:
        phrase = readable_finding(finding)
        phrase_ids = [int(value) for value in tokenizer.encode(phrase, add_special_tokens=False)]
        for prompt in prompts_for(finding):
            ids = [
                int(value)
                for value in tokenizer.encode(prompt["text"], add_special_tokens=False)
            ]
            rows.append(
                {
                    "finding": finding,
                    "prompt_name": prompt["name"],
                    "prompt_text": prompt["text"],
                    "token_ids": ids,
                    "token_ids_sha256": signed_int_sequence_sha256(ids),
                    "token_count": len(ids),
                    "finding_phrase": phrase,
                    "finding_phrase_token_ids": phrase_ids,
                    "finding_phrase_token_count": len(phrase_ids),
                }
            )
    payload = {
        "verbalizer_token_ids": verbalizer_ids,
        "rows": rows,
        "contract": (
            "raw question tokenization is frozen and recorded; each answer verbalizer is one "
            "distinct token; model-specific wrapped input tokenization is additionally recorded "
            "inside every scored cell"
        ),
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def full_model_artifact_fingerprint(model_dir: Path) -> dict[str, Any]:
    """Content-hash weights and every local non-weight runtime asset.

    Hulu loads custom model/processor Python files from the checkpoint with
    ``trust_remote_code=True``.  Hashing only ``config.json`` and the weight
    index would therefore leave the executable model path mutable while a run
    retained the same checkpoint identity.
    """

    cheap = model_artifact_fingerprint(model_dir)
    weights = []
    for path in sorted(model_dir.rglob("*.safetensors")):
        weights.append(
            {
                "name": str(path.relative_to(model_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not weights:
        raise FileNotFoundError(f"no safetensor weights found in {model_dir}")
    runtime_assets = [
        {
            "name": str(path.relative_to(model_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and path.suffix != ".safetensors"
    ]
    payload = {
        "metadata_and_inventory": cheap,
        "weight_content_hashes": weights,
        "non_weight_runtime_asset_hashes": runtime_assets,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def python_source_tree_fingerprint(root: Path) -> dict[str, Any]:
    """Freeze an external Python runtime used outside the model directory."""

    files = [path for path in sorted(root.rglob("*.py")) if path.is_file()]
    if not files:
        raise FileNotFoundError(f"no Python runtime sources found in {root}")
    payload = {
        "root": str(root.resolve()),
        "files": [
            {
                "name": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def source_fingerprints() -> dict[str, str]:
    here = Path(__file__)
    root = here.parent
    sources = {
        "factorial_runner": here,
        "dicom_renderer": root / "run_huatuo_dicom_render_pilot_v1.py",
        "huatuo_probe_helpers": root / "run_huatuo_vindr_commitment_probe.py",
        "hulu_probe_helpers": root / "run_hulu_vindr_commitment_probe.py",
    }
    return {name: sha256_file(path) for name, path in sources.items()}


def environment_fingerprint() -> dict[str, str]:
    try:
        import pydicom

        pydicom_version = str(pydicom.__version__)
    except ImportError:
        pydicom_version = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pillow": getattr(Image, "__version__", "unknown"),
        "pydicom": pydicom_version,
    }


def acquisition_view_from_dicom(path: Path) -> dict[str, str | None]:
    """Read ViewPosition without inventing a value when VinDr omits it."""

    import pydicom

    dataset = pydicom.dcmread(
        str(path), stop_before_pixels=True, specific_tags=["ViewPosition"]
    )
    raw_value = getattr(dataset, "ViewPosition", None)
    raw = None if raw_value is None else str(raw_value).strip().upper()
    aliases = {
        "PA": "pa",
        "AP": "ap",
        "AP SUPINE": "ap_supine",
        "LL": "left_lateral",
        "LATERAL": "lateral",
        "LAT": "lateral",
    }
    return {
        "normalized": aliases.get(raw or "", "unknown"),
        "raw_dicom_view_position": raw,
        "source": "DICOM ViewPosition; explicit unknown when absent/unrecognized",
    }


def freeze_config(candidate: dict[str, Any], path: Path, resume: bool) -> dict[str, Any]:
    if not resume:
        if path.exists():
            raise FileExistsError(path)
        immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
        candidate["fingerprint"] = canonical_json_sha256(immutable)
        atomic_json(path, candidate)
        return candidate
    if not path.is_file():
        raise FileNotFoundError("--resume requires the original config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    ignored = {"created_at", "command", "fingerprint"}
    left = {key: value for key, value in existing.items() if key not in ignored}
    right = {key: value for key, value in candidate.items() if key not in ignored}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"refusing resume after config drift: {changed}")
    expected = canonical_json_sha256(
        {key: value for key, value in existing.items() if key not in {"created_at", "command", "fingerprint"}}
    )
    if existing.get("fingerprint") != expected:
        raise ValueError("stored config fingerprint does not match stored immutable config")
    return existing


class FactorialScorer(Protocol):
    tokenizer: Any
    model_family: str

    def score(self, image: Image.Image, prompt: str) -> dict[str, Any]: ...

    def standard_next_token(self, image: Image.Image, prompt: str) -> dict[str, Any]: ...


def fp32_tristate_readout(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    verbalizer_ids: Mapping[str, int],
) -> dict[str, Any]:
    states = tuple(VERBALIZERS)
    ids = torch.tensor([verbalizer_ids[state] for state in states], device=output_weight.device)
    logits_tensor = hidden.float() @ output_weight.index_select(0, ids).float().T
    logits = {state: float(logits_tensor[index].detach().cpu()) for index, state in enumerate(states)}
    values = np.asarray([logits[state] for state in states], dtype=np.float64)
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    tristate_entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    return {
        "logits": logits,
        "polarity": float(logits["supported"] - logits["refuted"]),
        "commitment": float(max(logits["supported"], logits["refuted"]) - logits["undetermined"]),
        "prediction": max(logits, key=logits.get),
        "tristate_entropy": tristate_entropy,
        "tristate_entropy_unit": "nats over FP32 Yes/No/Maybe softmax",
        "readout": "FP32 final hidden @ FP32 Yes/No/Maybe lm-head rows at exact next-token position",
    }


def compare_next_token_readouts(
    direct: Mapping[str, Any],
    standard: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Check answer position against the model's ordinary generation path.

    Centering removes an irrelevant common logit offset.  A small tolerance is
    necessary because the ordinary LM-head path commonly performs its matmul in
    BF16 whereas the scientific readout intentionally promotes both operands to
    FP32.
    """

    states = tuple(VERBALIZERS)
    left = np.asarray([float(direct["logits"][state]) for state in states])
    right = np.asarray([float(standard["logits"][state]) for state in states])
    left_centered, right_centered = left - left.mean(), right - right.mean()
    max_error = float(np.max(np.abs(left_centered - right_centered)))
    direct_choice = states[int(np.argmax(left))]
    standard_choice = states[int(np.argmax(right))]
    passed = bool(max_error <= tolerance and direct_choice == standard_choice)
    return {
        "passed": passed,
        "centered_tristate_max_abs_error": max_error,
        "tolerance": float(tolerance),
        "direct_tristate_choice": direct_choice,
        "standard_tristate_choice": standard_choice,
        "direct_logits": {state: float(direct["logits"][state]) for state in states},
        "standard_generation_logits": {
            state: float(standard["logits"][state]) for state in states
        },
        "standard_generated_token_id": standard.get("generated_token_id"),
        "standard_generated_text": standard.get("generated_text"),
        "contract": (
            "same image and fully wrapped prompt; direct FP32 last-hidden readout versus ordinary "
            "greedy generate(max_new_tokens=1, output_scores=True) at its first generated token"
        ),
    }


class HuatuoScorer:
    model_family = "huatuo"

    def __init__(self, model_dir: Path, huatuo_root: Path, device: str):
        klass = import_huatuo(huatuo_root)
        self.bot = klass(str(model_dir), device=device)
        self.bot.model.eval()
        self.tokenizer = self.bot.tokenizer
        self.verbalizer_ids = tokenizer_audit(self.tokenizer)["verbalizer_token_ids"]

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        bot = self.bot
        tensor = torch.stack(bot.get_image_tensors([image])).to(
            bot.model.device, dtype=torch.bfloat16
        )
        with_image = bot.insert_image_placeholder(prompt, 1)
        input_ids = bot.preprocess(
            bot.get_conv_without_history(with_image), return_tensors="pt"
        ).to(bot.model.device)
        image_positions = torch.where(input_ids < 0)[0]
        if image_positions.numel() != 1:
            raise RuntimeError("Huatuo prompt must contain exactly one image placeholder")
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.full_like(input_ids, -100)
        _, position_ids, attention, _, embeddings, _ = (
            bot.model.prepare_inputs_labels_for_multimodal_new(
                [input_ids], None, [attention], None, [labels], tensor
            )
        )
        if embeddings is None:
            raise RuntimeError("Huatuo multimodal expansion returned no embeddings")
        output = bot.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=position_ids,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[0, -1]
        scores = fp32_tristate_readout(
            hidden, bot.model.get_output_embeddings().weight, self.verbalizer_ids
        )
        raw_ids = [int(value) for value in input_ids.detach().cpu().tolist()]
        scores["wrapped_input_audit"] = {
            "token_count_before_visual_expansion": len(raw_ids),
            "signed_token_ids_sha256": signed_int_sequence_sha256(raw_ids),
            "image_placeholder_count": int(image_positions.numel()),
            "expanded_sequence_length": int(embeddings.shape[1]),
        }
        return scores

    @torch.inference_mode()
    def standard_next_token(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        bot = self.bot
        tensor = torch.stack(bot.get_image_tensors([image])).to(
            bot.model.device, dtype=torch.bfloat16
        )
        with_image = bot.insert_image_placeholder(prompt, 1)
        input_ids = bot.preprocess(
            bot.get_conv_without_history(with_image), return_tensors="pt"
        ).unsqueeze(0).to(bot.model.device)
        generated = bot.model.generate(
            input_ids,
            images=tensor,
            use_cache=True,
            do_sample=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=bot.tokenizer.pad_token_id,
            eos_token_id=bot.tokenizer.eos_token_id,
        )
        if len(generated.scores) != 1:
            raise RuntimeError("Huatuo one-token generation did not return exactly one score tensor")
        logits_tensor = generated.scores[0][0].float()
        logits = {
            state: float(logits_tensor[token_id].detach().cpu())
            for state, token_id in self.verbalizer_ids.items()
        }
        generated_id = int(generated.sequences[0, -1].detach().cpu())
        return {
            "logits": logits,
            "generated_token_id": generated_id,
            "generated_text": bot.tokenizer.decode([generated_id], skip_special_tokens=False),
        }


class HuluScorer:
    model_family = "hulu"

    def __init__(self, model_dir: Path, max_visual_tokens: int):
        from corrected_sgta.run_hulu_vindr_commitment_probe import HuluRuntime

        self.runtime = HuluRuntime(model_dir, max_visual_tokens)
        self.tokenizer = self.runtime.tokenizer
        self.verbalizer_ids = tokenizer_audit(self.tokenizer)["verbalizer_token_ids"]

    @torch.inference_mode()
    def score(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        runtime = self.runtime
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = runtime.processor(
            images=[image],
            conversation=conversation,
            add_system_prompt=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = runtime.model.device
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                if key == "pixel_values":
                    value = value.to(dtype=runtime.model.dtype)
                inputs[key] = value.to(device)
        input_ids = inputs["input_ids"]
        image_mask = input_ids[0].eq(runtime.model.config.image_token_index)
        if int(image_mask.sum()) <= 0:
            raise RuntimeError("Hulu prompt contains no image tokens")
        _, attention, position_ids, _, embeddings, _ = (
            runtime.model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                position_ids=inputs.get("position_ids"),
                pixel_values=inputs.get("pixel_values"),
                grid_sizes=inputs.get("grid_sizes"),
                merge_sizes=inputs.get("merge_sizes"),
                modals=inputs.get("modals"),
            )
        )
        if embeddings is None:
            raise RuntimeError("Hulu multimodal expansion returned no embeddings")
        if attention is None:
            attention = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
        output = runtime.model.model(
            input_ids=None,
            attention_mask=attention,
            position_ids=position_ids,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        scores = fp32_tristate_readout(
            output.last_hidden_state[0, -1],
            runtime.model.get_output_embeddings().weight,
            self.verbalizer_ids,
        )
        raw_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        scores["wrapped_input_audit"] = {
            "token_count_with_visual_tokens": len(raw_ids),
            "signed_token_ids_sha256": signed_int_sequence_sha256(raw_ids),
            "image_token_count": int(image_mask.sum()),
            "expanded_sequence_length": int(embeddings.shape[1]),
        }
        return scores

    @torch.inference_mode()
    def standard_next_token(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        runtime = self.runtime
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = runtime.processor(
            images=[image],
            conversation=conversation,
            add_system_prompt=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                if key == "pixel_values":
                    value = value.to(dtype=runtime.model.dtype)
                inputs[key] = value.to(runtime.model.device)
        generated = runtime.model.generate(
            **inputs,
            use_cache=True,
            do_sample=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=runtime.tokenizer.pad_token_id,
            eos_token_id=runtime.tokenizer.eos_token_id,
        )
        if len(generated.scores) != 1:
            raise RuntimeError("Hulu one-token generation did not return exactly one score tensor")
        logits_tensor = generated.scores[0][0].float()
        logits = {
            state: float(logits_tensor[token_id].detach().cpu())
            for state, token_id in self.verbalizer_ids.items()
        }
        generated_id = int(generated.sequences[0, -1].detach().cpu())
        return {
            "logits": logits,
            "generated_token_id": generated_id,
            "generated_text": runtime.tokenizer.decode(
                [generated_id], skip_special_tokens=False
            ),
        }


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    render_name: str
    prompt_name: str
    prompt_text: str
    role: str
    reference_cell_id: str | None


def cell_specs(finding: str) -> list[CellSpec]:
    prompts = prompts_for(finding)
    output = []
    for render_name in SCIENCE_RENDER_NAMES:
        for prompt in prompts:
            output.append(
                CellSpec(
                    cell_id=f"science__{render_name}__{prompt['name']}",
                    render_name=render_name,
                    prompt_name=prompt["name"],
                    prompt_text=prompt["text"],
                    role="science_factorial",
                    reference_cell_id=None,
                )
            )
    for prompt in prompts:
        output.append(
            CellSpec(
                cell_id=f"control_identity_image__{prompt['name']}",
                render_name=IDENTITY_RENDER_NAME,
                prompt_name=prompt["name"],
                prompt_text=prompt["text"],
                role="identity_image_control",
                reference_cell_id=f"science__{BASELINE_VIEW}__{prompt['name']}",
            )
        )
    duplicate_source = prompts[0]
    output.append(
        CellSpec(
            cell_id=f"control_duplicate_prompt__{DUPLICATE_PROMPT_NAME}",
            render_name=BASELINE_VIEW,
            prompt_name=DUPLICATE_PROMPT_NAME,
            prompt_text=duplicate_source["text"],
            role="exact_duplicate_prompt_control",
            reference_cell_id=f"science__{BASELINE_VIEW}__{duplicate_source['name']}",
        )
    )
    if len(output) != 19 or len({cell.cell_id for cell in output}) != 19:
        raise RuntimeError("CECD cell contract must contain 15 science and 4 control cells")
    return output


def audit_is_admitted(render: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    audit = render["audit"]
    if not bool(audit.get("clinical_guard_pass")):
        reasons.append("per_case_computational_guard_failed")
    if render["name"] not in (*SCIENCE_RENDER_NAMES, IDENTITY_RENDER_NAME):
        reasons.append("render_not_in_frozen_factorial")
    # This flag is intentionally about engineering admission only.  Human
    # clinical admission is external and remains false in the run metadata.
    return not reasons, reasons


def shard_path(root: Path, record_key: str, cell_id: str) -> Path:
    safe_cell = cell_id.replace(os.sep, "_")
    return root / record_key / f"{safe_cell}.json"


def valid_completed_cell(
    path: Path,
    config_fingerprint: str,
    record_key: str,
    spec: CellSpec,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status = row.get("status")
    if status not in {"ok", "missing_invalid_render"}:
        return False
    if any(
        (
            row.get("config_fingerprint") != config_fingerprint,
            row.get("record_key") != record_key,
            row.get("cell_id") != spec.cell_id,
            row.get("render_name") != spec.render_name,
            row.get("prompt_name") != spec.prompt_name,
            row.get("prompt_text_sha256")
            != hashlib.sha256(spec.prompt_text.encode()).hexdigest(),
            row.get("cell_role") != spec.role,
        )
    ):
        return False
    if status == "ok":
        scores = row.get("scores", {})
        logits = scores.get("logits", {})
        return set(logits) == set(VERBALIZERS) and all(
            np.isfinite(float(value)) for value in logits.values()
        ) and np.isfinite(float(scores.get("tristate_entropy", np.nan))) and bool(
            str(row.get("acquisition_view", "")).strip()
        )
    return "scores" not in row and bool(row.get("missing_reasons"))


def valid_render_audit_shard(
    path: Path,
    config_fingerprint: str,
    record_key: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    renders = row.get("renders", [])
    names = {str(render.get("name")) for render in renders}
    required = set(SCIENCE_RENDER_NAMES) | {IDENTITY_RENDER_NAME}
    return bool(
        row.get("status") == "engineering_render_audit_only"
        and row.get("config_fingerprint") == config_fingerprint
        and row.get("record_key") == record_key
        and names == required
        and all(
            len(str(render.get("audit", {}).get("pixel_sha256", ""))) == 64
            and isinstance(render.get("audit", {}).get("clinical_guard_pass"), bool)
            for render in renders
        )
    )


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def pack_factorial_rows(
    output_dir: Path,
    config: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble deterministic analyzer rows from validated atomic cell shards."""

    packed: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    incomplete: dict[tuple[str, str], dict[str, Any]] = {}
    cell_root = output_dir / "cell_shards"
    fingerprint = str(config["fingerprint"])
    model_id = str(config["model"])
    stage_label = str(config.get("stage_label", "dev"))
    manifest_split = str(config.get("manifest_split", "pilot"))
    ordered_rows = sorted(
        selected_rows,
        key=lambda row: (str(row["image_id"]), str(row["finding"])),
    )
    for manifest_row in ordered_rows:
        record_key = safe_record_key(manifest_row)
        for spec in cell_specs(str(manifest_row["finding"])):
            path = shard_path(cell_root, record_key, spec.cell_id)
            if not valid_completed_cell(path, fingerprint, record_key, spec):
                raise ValueError(f"cannot pack absent or invalid cell shard: {path}")
            cell = json.loads(path.read_text(encoding="utf-8"))
            status = str(cell["status"])
            scores = cell.get("scores")
            packed_row = {
                "contract_version": "clinical-equivalence-factorial-v1",
                "config_fingerprint": fingerprint,
                "model": model_id,
                "stage_label": stage_label,
                "source_manifest_split": manifest_split,
                "image_id": str(cell["image_id"]),
                "finding": str(cell["finding"]),
                "reader_votes": int(cell["positive_votes"]),
                "individual_reader_votes": [
                    int(value) for value in cell.get("individual_reader_votes", [])
                ],
                "positive_votes": int(cell["positive_votes"]),
                "reader_support": float(cell["reader_support"]),
                "render_id": str(cell["render_name"]),
                "prompt_id": str(cell["prompt_name"]),
                "cell_id": str(cell["cell_id"]),
                "cell_role": str(cell["cell_role"]),
                "reference_cell_id": cell.get("reference_cell_id"),
                "status": status,
                "signed_score": None if scores is None else float(scores["polarity"]),
                "commitment_score": None
                if scores is None
                else float(scores["commitment"]),
                "tristate_entropy": None
                if scores is None
                else float(scores["tristate_entropy"]),
                "answer_length_tokens": 1,
                "raw_prompt_token_count": int(cell["raw_prompt_token_count"]),
                "acquisition_view": str(cell.get("acquisition_view", "unknown")),
                "tristate_logits": None if scores is None else scores["logits"],
                "render_pixel_sha256": str(cell["render_pixel_sha256"]),
                "prompt_text_sha256": str(cell["prompt_text_sha256"]),
                "missing_reasons": cell.get("missing_reasons", []),
            }
            packed.append(packed_row)
            role_counts[spec.role] = role_counts.get(spec.role, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "ok":
                orbit_key = (str(cell["image_id"]), str(cell["finding"]))
                entry = incomplete.setdefault(
                    orbit_key,
                    {
                        "image_id": orbit_key[0],
                        "finding": orbit_key[1],
                        "invalid_cells": [],
                    },
                )
                entry["invalid_cells"].append(
                    {
                        "cell_id": str(cell["cell_id"]),
                        "reasons": list(cell.get("missing_reasons", [])),
                    }
                )

    expected = len(selected_rows) * 19
    expected_roles = {
        "science_factorial": len(selected_rows) * 15,
        "identity_image_control": len(selected_rows) * 3,
        "exact_duplicate_prompt_control": len(selected_rows),
    }
    if len(packed) != expected or role_counts != expected_roles:
        raise RuntimeError(
            f"packed factorial contract mismatch: n={len(packed)}/{expected}, "
            f"roles={role_counts}/{expected_roles}"
        )
    output_path = output_dir / "factorial_rows.jsonl"
    atomic_jsonl(output_path, packed)
    payload_records = [
        {
            "model": row["model"],
            "image_id": row["image_id"],
            "finding": row["finding"],
            "reader_votes": row["reader_votes"],
            "individual_reader_votes": row["individual_reader_votes"],
            "render_id": row["render_id"],
            "prompt_id": row["prompt_id"],
            "signed_score": row["signed_score"],
            "commitment_score": row["commitment_score"],
            "tristate_entropy": row["tristate_entropy"],
            "tristate_logits": row["tristate_logits"],
            "input_prompt_length_tokens": row["raw_prompt_token_count"],
            "answer_length_tokens": row["answer_length_tokens"],
            "acquisition_view": row["acquisition_view"],
            "valid": row["status"] == "ok",
            "exclusion_reasons": row["missing_reasons"],
        }
        for row in packed
    ]
    payload = {
        "schema_version": "clinical-equivalence-factorial-v1",
        "split": stage_label,
        "source_manifest_split": manifest_split,
        "split_note": (
            "pilot_screen is engineering-only; dev_fit freezes predictors; "
            "confirmation_locked is apply-only and never refits"
        ),
        "frozen_before_outputs": True,
        "score_definition": "fp32_yes_minus_no_logit",
        "primary_renders": list(SCIENCE_RENDER_NAMES),
        "primary_prompts": [name for name, _ in PROMPT_TEMPLATES],
        "baseline_render": BASELINE_VIEW,
        "baseline_prompt": PROMPT_TEMPLATES[0][0],
        "identity_render": IDENTITY_RENDER_NAME,
        "duplicate_prompt": DUPLICATE_PROMPT_NAME,
        "runner_config_fingerprint": fingerprint,
        "clinical_equivalence_admission": config.get(
            "clinical_admission",
            {"status": "pending_blinded_human_review"},
        ),
        "records": payload_records,
    }
    payload_path = output_dir / "factorial_payload.json"
    atomic_json(payload_path, payload)
    summary = {
        "contract_version": "clinical-equivalence-factorial-v1",
        "config_fingerprint": fingerprint,
        "model": model_id,
        "claims": len(selected_rows),
        "rows": len(packed),
        "complete_orbit_count": len(selected_rows) - len(incomplete),
        "incomplete_orbit_count": len(incomplete),
        "incomplete_orbits": [incomplete[key] for key in sorted(incomplete)],
        "role_counts": role_counts,
        "status_counts": status_counts,
        "factorial_rows_sha256": sha256_file(output_path),
        "factorial_payload_sha256": sha256_file(payload_path),
        "factorial_payload": str(payload_path),
        "ordering": "image_id, finding, then frozen 15+3+1 cell order",
    }
    atomic_json(output_dir / "factorial_rows_manifest.json", summary)
    return summary


def load_tokenizer_only(model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )


def selection(
    stage_label: str = "pilot_screen",
    manifest_split: str | None = None,
    per_bin: int | None = None,
) -> list[dict[str, Any]]:
    if stage_label not in STAGE_SPECS:
        raise ValueError(f"unknown CECD stage: {stage_label}")
    spec = STAGE_SPECS[stage_label]
    expected_split = str(spec["manifest_split"])
    expected_per_bin = int(spec["per_bin"])
    manifest_split = expected_split if manifest_split is None else manifest_split
    per_bin = expected_per_bin if per_bin is None else per_bin
    if manifest_split != expected_split or per_bin != expected_per_bin:
        raise ValueError(
            f"{stage_label} requires manifest_split={expected_split!r}, "
            f"per_bin={expected_per_bin}"
        )
    rows = balanced_rows(
        DEFAULT_MANIFEST,
        manifest_split,
        FROZEN_FINDINGS,
        FROZEN_VOTES,
        per_bin,
        FROZEN_SEED,
    )
    expected_claims = int(spec["claims"])
    if len(rows) != expected_claims:
        raise RuntimeError(
            f"frozen CECD {stage_label} must contain {expected_claims} claims, got {len(rows)}"
        )
    return rows


def model_defaults(model_family: str) -> Path:
    return Path(
        "/home/dbw/models/HuatuoGPT-Vision-7B"
        if model_family == "huatuo"
        else "/home/dbw/models/Hulu-Med-4B"
    )


def execution_contract(
    *, admission: dict[str, Any] | None, engineering_render_audit: bool,
    stage_label: str = "pilot_screen",
) -> dict[str, Any]:
    """Resolve the only two legal CECD execution modes.

    Formal scoring is impossible without a validated human-admission payload.
    The sole admission-free mode performs no model scoring or GPU work; its
    render-audit artifacts explicitly carry no scientific authorization.
    """

    if admission is None and not engineering_render_audit:
        raise RuntimeError(
            "formal CECD scoring requires --admission-result; the only "
            "admission-free mode is --engineering-render-audit"
        )
    if engineering_render_audit:
        return {
            "execution_mode": "engineering_render_audit_only",
            "scientific_status": "engineering_render_audit_only_no_scientific_authorization",
            "clinical_equivalence_established": False,
            "cecd_model_scoring_authorized": False,
            "scientific_artifact_authorized": False,
        }
    return {
        "execution_mode": "formal_human_admitted_model_scoring",
        "scientific_status": f"human_admitted_cecd_{stage_label}",
        "clinical_equivalence_established": True,
        "cecd_model_scoring_authorized": True,
        "scientific_artifact_authorized": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=("huatuo", "hulu"), default="huatuo")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--conformance-logit-tolerance", type=float, default=0.1)
    parser.add_argument("--stage-label", choices=tuple(STAGE_SPECS), default="pilot_screen")
    parser.add_argument("--manifest-split", choices=("pilot", "dev", "confirmation"))
    parser.add_argument("--per-bin", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--engineering-render-audit",
        "--render-audit-only",
        dest="render_audit_only",
        action="store_true",
        help=(
            "engineering-only render audit; the sole mode allowed without "
            "--admission-result and never a scientifically authorized artifact"
        ),
    )
    parser.add_argument(
        "--admission-result",
        type=Path,
        help=(
            "passed blinded human-admission analysis; its content hash and "
            "validated provenance are frozen into every formal model run"
        ),
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        help="engineering canary only; formal pilot omits this and always uses all 160 claims",
    )
    args = parser.parse_args()
    # Resolve the fail-closed execution boundary before model paths, tokenizer
    # metadata, DICOMs, output directories, or CUDA can be touched.
    execution = execution_contract(
        admission=None if args.admission_result is None else {"status": "supplied"},
        engineering_render_audit=args.render_audit_only,
        stage_label=args.stage_label,
    )
    stage_spec = STAGE_SPECS[args.stage_label]
    manifest_split = args.manifest_split or str(stage_spec["manifest_split"])
    per_bin = args.per_bin if args.per_bin is not None else int(stage_spec["per_bin"])
    if manifest_split != stage_spec["manifest_split"] or per_bin != stage_spec["per_bin"]:
        raise ValueError("stage-label/manifest-split/per-bin contract mismatch")
    if args.max_claims is not None and (
        args.stage_label != "pilot_screen"
        or args.max_claims <= 0
        or args.max_claims >= int(stage_spec["claims"])
    ):
        raise ValueError(
            "--max-claims is an engineering pilot_screen canary only and must be below 160"
        )
    if not 0.0 < args.conformance_logit_tolerance <= 1.0:
        raise ValueError("--conformance-logit-tolerance must lie in (0,1]")
    admission: dict[str, Any] | None = None
    if args.admission_result is not None:
        from corrected_sgta.cecd_admission_gate import require_cecd_authorization

        authorized = require_cecd_authorization(args.admission_result)
        admission = {
            "status": "passed_hash_bound",
            "analysis_path": str(args.admission_result.resolve()),
            "analysis_sha256": sha256_file(args.admission_result),
            "analysis_version": authorized["version"],
            "cecd_model_scoring_authorized": True,
        }
    execution = execution_contract(
        admission=admission,
        engineering_render_audit=args.render_audit_only,
        stage_label=args.stage_label,
    )
    model_dir = args.model_dir or model_defaults(args.model_family)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    frozen_rows = selection(args.stage_label, manifest_split, per_bin)
    rows = frozen_rows if args.max_claims is None else frozen_rows[: args.max_claims]
    frozen_keys = [safe_record_key(row) for row in frozen_rows]
    active_keys = [safe_record_key(row) for row in rows]
    bbox_rows = load_jsonl(DEFAULT_BBOXES)
    boxes_by_claim = {
        (str(row["image_id"]), str(row["finding"])): list(row.get("boxes", []))
        for row in bbox_rows
    }
    boxes_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in bbox_rows:
        boxes_by_image.setdefault(str(row["image_id"]), []).extend(row.get("boxes", []))

    tokenizer = load_tokenizer_only(model_dir)
    token_audit = tokenizer_audit(tokenizer)
    model_provenance = (
        {
            "mode": "tokenizer_and_metadata_only_for_cpu_audit",
            "cheap_fingerprint": model_artifact_fingerprint(model_dir),
        }
        if args.render_audit_only
        else {
            "mode": "full_content_hash_including_all_weight_shards",
            "full_fingerprint": full_model_artifact_fingerprint(model_dir),
            "external_runtime_source": (
                python_source_tree_fingerprint(args.huatuo_root)
                if args.model_family == "huatuo"
                else None
            ),
        }
    )
    config_candidate: dict[str, Any] = {
        "version": VERSION,
        "measurement_name": MEASUREMENT_NAME,
        "created_at": utc_now(),
        **execution,
        "clinical_admission": admission or {"status": "pending_blinded_human_review"},
        "generic_cross_modal_interaction_novelty_claimed": False,
        "novelty_boundary": (
            "measurement is limited to clinically admitted DICOM-render x speech-act-equivalent "
            "question composition, with later reader-vote calibration; generic vision/text/cross-modal "
            "interventions are prior art including Treble Counterfactual VLMs (arXiv:2503.06169)"
        ),
        "dataset": "vindr-cxr-1.0.0-fixed-three-reader-panel",
        "manifest": str(DEFAULT_MANIFEST),
        "manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "bboxes": str(DEFAULT_BBOXES),
        "bboxes_sha256": sha256_file(DEFAULT_BBOXES),
        "image_root": str(DEFAULT_IMAGE_ROOT),
        "stage_label": args.stage_label,
        "manifest_split": manifest_split,
        "split": manifest_split,
        "findings": list(FROZEN_FINDINGS),
        "votes": list(FROZEN_VOTES),
        "per_finding_vote_bin": per_bin,
        "seed": FROZEN_SEED,
        "frozen_claim_count": len(frozen_rows),
        "frozen_selection_keys_sha256": canonical_json_sha256(frozen_keys),
        "active_claim_count": len(rows),
        "active_selection_keys_sha256": canonical_json_sha256(active_keys),
        "engineering_canary_max_claims": args.max_claims,
        "model_family": args.model_family,
        "model": f"{args.model_family}:{model_dir.name}",
        "model_dir": str(model_dir.resolve()),
        "model_provenance": model_provenance,
        "device": args.device,
        "max_visual_tokens": args.max_visual_tokens if args.model_family == "hulu" else None,
        "render_audit_only": args.render_audit_only,
        "science_render_names": list(SCIENCE_RENDER_NAMES),
        "identity_render_name": IDENTITY_RENDER_NAME,
        "prompt_templates": [
            {"name": name, "template": template} for name, template in PROMPT_TEMPLATES
        ],
        "prompt_contract": {
            "proposition": "present(finding)",
            "speech_act": "polar_diagnostic_question",
            "answer_format": "exactly one of Yes/No/Maybe",
            "clinical_language_admission": (
                "passed_hash_bound" if admission is not None else "pending"
            ),
        },
        "tokenization_audit": token_audit,
        "cells_per_claim": {"science": 15, "identity_image_controls": 3, "duplicate_prompt_controls": 1},
        "missing_cell_policy": "record missing_invalid_render; never substitute baseline pixels or scores",
        "readout": "FP32 Yes/No/Maybe at exact next-token position",
        "next_token_conformance": {
            "required_before_scientific_scoring": not args.render_audit_only,
            "centered_tristate_logit_tolerance": args.conformance_logit_tolerance,
            "choice_must_match": True,
        },
        "source_sha256": source_fingerprints(),
        "environment": environment_fingerprint(),
        "command": " ".join(sys.argv),
    }
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    config = freeze_config(config_candidate, args.output_dir / "config.json", args.resume)
    config_fingerprint = str(config["fingerprint"])

    scorer: FactorialScorer | None = None
    if not args.render_audit_only:
        scorer = (
            HuatuoScorer(model_dir, args.huatuo_root, args.device)
            if args.model_family == "huatuo"
            else HuluScorer(model_dir, args.max_visual_tokens)
        )
        runtime_audit = tokenizer_audit(scorer.tokenizer)
        if runtime_audit["fingerprint"] != token_audit["fingerprint"]:
            raise RuntimeError("runtime tokenizer differs from frozen tokenizer-only audit")

    conformance_path = args.output_dir / "next_token_conformance.json"
    conformance_done = False
    if not args.render_audit_only and args.resume and conformance_path.is_file():
        existing_conformance = json.loads(conformance_path.read_text(encoding="utf-8"))
        conformance_done = bool(
            existing_conformance.get("passed")
            and existing_conformance.get("config_fingerprint") == config_fingerprint
        )

    audit_root = args.output_dir / "render_audit_shards"
    cell_root = args.output_dir / "cell_shards"
    completed_cells = skipped_cells = missing_cells = errors = 0
    started_run = time.perf_counter()
    cached_image_id: str | None = None
    cached: tuple[Any, Path, str, dict[str, str | None]] | None = None
    for claim_index, row in enumerate(rows, start=1):
        record_key = safe_record_key(row)
        image_id, finding = str(row["image_id"]), str(row["finding"])
        audit_target = audit_root / f"{record_key}.json"
        specs = cell_specs(finding)
        if args.render_audit_only and valid_render_audit_shard(
            audit_target, config_fingerprint, record_key
        ):
            print(f"[{claim_index}/{len(rows)}] resume skip audit {record_key}", flush=True)
            continue
        if not args.render_audit_only and all(
            valid_completed_cell(
                shard_path(cell_root, record_key, spec.cell_id),
                config_fingerprint,
                record_key,
                spec,
            )
            for spec in specs
        ):
            skipped_cells += len(specs)
            print(f"[{claim_index}/{len(rows)}] resume skip all cells {record_key}", flush=True)
            continue
        try:
            path = resolve_image(row, DEFAULT_IMAGE_ROOT)
            if not path.is_file():
                raise FileNotFoundError(path)
            if cached_image_id != image_id:
                cached = (
                    read_dicom_pixels(path),
                    path,
                    sha256_file(path),
                    acquisition_view_from_dicom(path),
                )
                cached_image_id = image_id
            assert cached is not None and cached[1] == path
            pixels, _, dicom_sha, acquisition_view = cached
            rendered = build_render_views(
                pixels,
                boxes_by_claim.get((image_id, finding), []),
                boxes_by_image.get(image_id, []),
            )
            by_render = {str(view["name"]): view for view in rendered}
            required = set(SCIENCE_RENDER_NAMES) | {IDENTITY_RENDER_NAME}
            if not required.issubset(by_render):
                raise RuntimeError(f"renderer omitted frozen views: {sorted(required - set(by_render))}")
            audit_payload = {
                "version": VERSION,
                "status": "engineering_render_audit_only",
                "scientific_status": config["scientific_status"],
                "config_fingerprint": config_fingerprint,
                "record_key": record_key,
                "image_id": image_id,
                "finding": finding,
                "positive_votes": int(row["positive_votes"]),
                "dicom_relative_path": row.get("dicom_relpath"),
                "dicom_sha256": dicom_sha,
                "dicom_metadata": pixels.metadata,
                "acquisition_view": acquisition_view,
                "renders": [
                    {
                        key: value
                        for key, value in by_render[name].items()
                        if key != "image"
                    }
                    for name in (*SCIENCE_RENDER_NAMES, IDENTITY_RENDER_NAME)
                ],
            }
            atomic_json(audit_target, audit_payload)
            if args.render_audit_only:
                print(f"[{claim_index}/{len(rows)}] audit {record_key}", flush=True)
                continue

            assert scorer is not None
            if not conformance_done:
                baseline = by_render[BASELINE_VIEW]
                admitted, reasons = audit_is_admitted(baseline)
                if not admitted:
                    raise RuntimeError(
                        f"cannot run next-token conformance on invalid baseline: {reasons}"
                    )
                canary_prompt = prompts_for(finding)[0]["text"]
                direct = scorer.score(baseline["image"], canary_prompt)
                standard = scorer.standard_next_token(baseline["image"], canary_prompt)
                conformance = compare_next_token_readouts(
                    direct,
                    standard,
                    args.conformance_logit_tolerance,
                )
                conformance.update(
                    {
                        "version": VERSION,
                        "config_fingerprint": config_fingerprint,
                        "model": config["model"],
                        "image_id": image_id,
                        "finding": finding,
                        "render_id": BASELINE_VIEW,
                        "prompt_id": "existential",
                    }
                )
                atomic_json(conformance_path, conformance)
                if not conformance["passed"]:
                    raise RuntimeError(
                        "next-token conformance failed: "
                        f"centered_error={conformance['centered_tristate_max_abs_error']}, "
                        f"choices={conformance['direct_tristate_choice']}/"
                        f"{conformance['standard_tristate_choice']}"
                    )
                conformance_done = True
            prompt_token_rows = {
                (item["finding"], item["prompt_name"]): item
                for item in token_audit["rows"]
            }
            for spec in specs:
                target = shard_path(cell_root, record_key, spec.cell_id)
                if valid_completed_cell(target, config_fingerprint, record_key, spec):
                    skipped_cells += 1
                    continue
                render = by_render[spec.render_name]
                admitted, reasons = audit_is_admitted(render)
                prompt_source_name = (
                    "existential" if spec.role == "exact_duplicate_prompt_control" else spec.prompt_name
                )
                prompt_tokens = prompt_token_rows[(finding, prompt_source_name)]
                common = {
                    "version": VERSION,
                    "config_fingerprint": config_fingerprint,
                    "record_key": record_key,
                    "image_id": image_id,
                    "finding": finding,
                    "positive_votes": int(row["positive_votes"]),
                    "individual_reader_votes": individual_reader_votes(row),
                    "reader_support": float(row["reader_support"]),
                    "acquisition_view": acquisition_view["normalized"],
                    "acquisition_view_audit": acquisition_view,
                    "cell_id": spec.cell_id,
                    "cell_role": spec.role,
                    "reference_cell_id": spec.reference_cell_id,
                    "render_name": spec.render_name,
                    "render_pixel_sha256": render["audit"]["pixel_sha256"],
                    "render_audit": render["audit"],
                    "prompt_name": spec.prompt_name,
                    "prompt_text": spec.prompt_text,
                    "prompt_text_sha256": hashlib.sha256(spec.prompt_text.encode()).hexdigest(),
                    "raw_prompt_token_ids_sha256": prompt_tokens["token_ids_sha256"],
                    "raw_prompt_token_count": prompt_tokens["token_count"],
                    "dicom_sha256": dicom_sha,
                }
                if not admitted:
                    atomic_json(
                        target,
                        {**common, "status": "missing_invalid_render", "missing_reasons": reasons},
                    )
                    missing_cells += 1
                    continue
                scores = scorer.score(render["image"], spec.prompt_text)
                atomic_json(target, {**common, "status": "ok", "scores": scores})
                completed_cells += 1
            print(
                f"[{claim_index}/{len(rows)}] score {record_key} "
                f"new={completed_cells} skipped={skipped_cells} missing={missing_cells}",
                flush=True,
            )
        except Exception as error:
            errors += 1
            atomic_json(
                args.output_dir / "claim_errors" / f"{record_key}.json",
                {
                    "version": VERSION,
                    "status": "error",
                    "config_fingerprint": config_fingerprint,
                    "record_key": record_key,
                    "image_id": image_id,
                    "finding": finding,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{claim_index}/{len(rows)}] ERROR {record_key}: {error!r}", file=sys.stderr, flush=True)

    audit_count = len(list(audit_root.glob("*.json")))
    packed_manifest = None
    if not args.render_audit_only and not errors:
        packed_manifest = pack_factorial_rows(args.output_dir, config, rows)
    state = {
        "version": VERSION,
        "measurement_name": MEASUREMENT_NAME,
        "scientific_status": config["scientific_status"],
        "config_fingerprint": config_fingerprint,
        "active_claim_count": len(rows),
        "render_audit_shards_present": audit_count,
        "new_complete_cells": completed_cells,
        "resume_skipped_cells": skipped_cells,
        "new_missing_invalid_render_cells": missing_cells,
        "claim_errors_this_invocation": errors,
        "next_token_conformance_passed": None
        if args.render_audit_only
        else conformance_done,
        "factorial_rows_manifest": packed_manifest,
        "elapsed_seconds": time.perf_counter() - started_run,
        "updated_at": utc_now(),
    }
    atomic_json(args.output_dir / "run_state.json", state)
    print(json.dumps(state, indent=2))
    if errors:
        raise RuntimeError(f"{errors} claims failed; diagnose and rerun with --resume")


if __name__ == "__main__":
    main()
