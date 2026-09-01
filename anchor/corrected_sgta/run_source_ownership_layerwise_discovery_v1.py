#!/usr/bin/env python3
"""Discovery-only layerwise readout for the controlled Source x Polarity probe.

This runner deliberately refuses confirmation manifests.  It reuses the exact
prompt wrapping, DICOM rendering, model runtimes, and tristate verbalizers from
the completed final-layer experiment, but requests all decoder hidden states.
It does not generate text and never reads reader votes or clinical targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from corrected_sgta.run_cecd_factorial_v1 import fp32_tristate_readout
from corrected_sgta.run_target_blind_dicom_tristate_margin_v2 import (
    DEFAULT_GPU_LOCK,
    DEFAULT_IMAGE_ROOT,
    canonical_gpu_lock,
    load_dicom,
    load_rows,
    preflight,
    resolve_dicom,
)
from corrected_sgta.run_target_blind_tristate_margin_v1 import (
    DEFAULT_HUATUO_ROOT,
    DEFAULT_MODELS,
    atomic_json,
    atomic_jsonl,
    build_scorer,
    canonical_hash,
    scoring_prompt,
    sha256_file,
)


VERSION = "source-ownership-layerwise-discovery-v1"


def public_layer_score(raw: Mapping[str, Any]) -> dict[str, Any]:
    logits = raw["logits"]
    return {
        "tristate_logits_fp32": {
            "Yes": float(logits["supported"]),
            "No": float(logits["refuted"]),
            "Maybe": float(logits["undetermined"]),
        },
        "polarity_yes_minus_no": float(raw["polarity"]),
        "commitment_max_yes_no_minus_maybe": float(raw["commitment"]),
        "tristate_entropy_nats": float(raw["tristate_entropy"]),
    }


def readout_hidden_states(
    causal_lm: Any,
    hidden_states: tuple[torch.Tensor, ...],
    verbalizer_ids: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Apply the final norm and LM head consistently at every decoder depth."""

    if not hidden_states:
        raise ValueError("decoder returned no hidden states")
    output_weight = causal_lm.get_output_embeddings().weight
    final = len(hidden_states) - 1
    result: dict[str, dict[str, Any]] = {}
    for layer, state in enumerate(hidden_states):
        answer_state = state[0, -1]
        if layer != final:
            answer_state = causal_lm.model.norm(answer_state.unsqueeze(0))[0]
        result[str(layer)] = public_layer_score(
            fp32_tristate_readout(answer_state, output_weight, verbalizer_ids)
        )
    return result


@torch.inference_mode()
def score_huatuo_layers(scorer: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    bot = scorer.bot
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
    output = bot.model.model(
        input_ids=None,
        attention_mask=attention,
        position_ids=position_ids,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return readout_hidden_states(bot.model, output.hidden_states, scorer.verbalizer_ids)


@torch.inference_mode()
def score_hulu_layers(scorer: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    runtime = scorer.runtime
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
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
    if attention is None:
        attention = torch.ones(input_ids.shape, dtype=torch.bool, device=runtime.model.device)
    output = runtime.model.model(
        input_ids=None,
        attention_mask=attention,
        position_ids=position_ids,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return readout_hidden_states(runtime.model, output.hidden_states, scorer.verbalizer_ids)


def load_final_margins(path: Path | None, expected_qids: set[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_qid = {str(row["qid"]): row for row in rows}
    if len(by_qid) != len(rows) or set(by_qid) != expected_qids:
        raise ValueError("final-margin qids do not exactly match discovery manifest")
    return by_qid


def check_final_identity(
    layers: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Any], tolerance: float
) -> dict[str, Any]:
    final = layers[str(max(map(int, layers)))]
    differences = {
        key: abs(float(final[key]) - float(reference[key]))
        for key in ("polarity_yes_minus_no", "commitment_max_yes_no_minus_maybe")
    }
    passed = max(differences.values()) <= tolerance
    if not passed:
        raise RuntimeError(f"final-layer identity failed: {differences}, tolerance={tolerance}")
    return {"passed": True, "absolute_differences": differences, "tolerance": tolerance}


def run_self_tests() -> None:
    class Norm(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value / value.norm(dim=-1, keepdim=True)

    class FakeLM:
        def __init__(self) -> None:
            self.model = type("Root", (), {"norm": Norm()})()
            self.weight = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])

        def get_output_embeddings(self):
            return type("Head", (), {"weight": self.weight})()

    hidden = (torch.tensor([[[3., 4.]]]), torch.tensor([[[.6, .8]]]))
    scores = readout_hidden_states(
        FakeLM(), hidden, {"supported": 0, "refuted": 1, "undetermined": 2}
    )
    assert set(scores) == {"0", "1"}
    assert math.isclose(scores["0"]["polarity_yes_minus_no"], -.2, abs_tol=1e-6)
    reference = {
        "polarity_yes_minus_no": scores["1"]["polarity_yes_minus_no"],
        "commitment_max_yes_no_minus_maybe": scores["1"]["commitment_max_yes_no_minus_maybe"],
    }
    assert check_final_identity(scores, reference, 1e-7)["passed"]
    with tempfile.TemporaryDirectory() as temporary:
        confirmation = Path(temporary) / "confirmation.json"
        confirmation.write_text(json.dumps([{
            "qid": "x", "pair_id": "x", "arm": "plain", "finding": "x",
            "img_name": "x.dicom", "question": "x", "selection_uses_target_vote": False,
            "experiment_split": "confirmation",
            "controlled_source_injection_not_natural_rag": True,
        }]))
        try:
            load_rows(confirmation)
        except ValueError:
            pass
        else:
            raise AssertionError("confirmation manifest was accepted")
    print(json.dumps({"status": "passed", "tests": 5, "gpu_used": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=False)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-family", choices=("huatuo", "hulu"), default="huatuo")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--wait-for-gpu-lock", action="store_true")
    parser.add_argument("--final-margins", type=Path)
    parser.add_argument("--identity-tolerance", type=float, default=2e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required")
    args.model_dir = args.model_dir or DEFAULT_MODELS[args.model_family]
    rows = load_rows(args.input)  # fail-closed: accepts discovery only
    inspection = preflight(rows, args.input, args.image_root, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        atomic_json(args.output_dir / "preflight.json", inspection)
        print(json.dumps({"status": "passed", "rows": len(rows), "gpu_used": False}, indent=2))
        return

    config_path = args.output_dir / "config.json"
    immutable = {
        "version": VERSION,
        "model_family": args.model_family,
        "model_dir": str(args.model_dir.resolve()),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "image_root": str(args.image_root.resolve()),
        "max_visual_tokens": args.max_visual_tokens,
        "final_margins": str(args.final_margins.resolve()) if args.final_margins else None,
        "final_margins_sha256": sha256_file(args.final_margins) if args.final_margins else None,
        "identity_tolerance": args.identity_tolerance,
        "measurement": "all hidden-state FP32 Yes/No/Maybe logit-lens readouts",
        "generation_called": False,
        "target_or_vote_data_accessed": False,
        "code_sha256": sha256_file(Path(__file__)),
    }
    candidate = {**immutable, "fingerprint": canonical_hash(immutable)}
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != candidate:
            raise ValueError("existing output config differs; use a fresh directory")
    else:
        atomic_json(config_path, candidate)
    raw_path = args.output_dir / "layerwise.jsonl"
    completed = set()
    existing: list[dict[str, Any]] = []
    if raw_path.exists():
        if not args.resume:
            raise FileExistsError(raw_path)
        existing = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        completed = {str(row["qid"]) for row in existing}
    qids = {str(row["qid"]) for row in rows}
    final_margins = load_final_margins(args.final_margins, qids)

    packed = list(existing)
    with canonical_gpu_lock(args.gpu_lock, args.wait_for_gpu_lock):
        scorer = build_scorer(args) if len(completed) < len(rows) else None
        for index, row in enumerate(rows):
            qid = str(row["qid"])
            if qid in completed:
                continue
            image = load_dicom(resolve_dicom(args.image_root, str(row["img_name"])))
            try:
                prompt = scoring_prompt(str(row["question"]))
                layers = (
                    score_huatuo_layers(scorer, image, prompt)
                    if args.model_family == "huatuo"
                    else score_hulu_layers(scorer, image, prompt)
                )
            finally:
                image.close()
            identity = (
                check_final_identity(layers, final_margins[qid], args.identity_tolerance)
                if final_margins else None
            )
            packed.append({
                "status": "complete", "version": VERSION, "qid": qid,
                "pair_id": str(row["pair_id"]), "arm": str(row["arm"]),
                "finding": str(row["finding"]), "img_name": str(row["img_name"]),
                "layers": layers, "final_layer_identity": identity,
                "generation_called": False, "target_or_vote_data_accessed": False,
            })
            atomic_jsonl(raw_path, packed)
            print(f"[{len(packed)}/{len(rows)}] {qid}", flush=True)
    atomic_json(args.output_dir / "summary.json", {
        "status": "complete", "version": VERSION, "rows": len(packed),
        "pairs": len({row["pair_id"] for row in packed}),
        "layers": sorted(map(int, packed[0]["layers"])),
        "final_identity_checked": bool(final_margins),
        "layerwise_sha256": sha256_file(raw_path),
    })


if __name__ == "__main__":
    main()
