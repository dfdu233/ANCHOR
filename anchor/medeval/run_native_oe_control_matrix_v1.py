#!/usr/bin/env python3
"""Run the frozen OE control matrix while loading each VLM only once."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.run_native_oe_vqa import load_resume, load_rows, qid, stable_seed


VERSION = "native-oe-control-matrix-v1"


@dataclass(frozen=True)
class Arm:
    name: str
    decode_mode: str
    max_new_tokens: int
    seed: int
    temperature: float = 1.0
    top_p: float = 1.0

    @property
    def generation(self) -> dict[str, Any]:
        return {
            "do_sample": self.decode_mode == "sample",
            "num_beams": 1,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


def frozen_arms(contract: dict[str, Any]) -> list[Arm]:
    arms: list[Arm] = []
    for item in contract["temperature_length_controls"]["arms"]:
        seed = int(item.get("seed", contract["generation"]["base_seed"]))
        name = str(item["id"])
        if item["decode_mode"] == "sample":
            name = f"{name}_seed{seed}"
        arms.append(
            Arm(
                name=name,
                decode_mode=str(item["decode_mode"]),
                max_new_tokens=int(item["max_new_tokens"]),
                seed=seed,
                temperature=float(item.get("temperature", 1.0)),
                top_p=float(item.get("top_p", 1.0)),
            )
        )
    sc = contract["self_consistency"]
    existing = {arm.name for arm in arms}
    for seed in contract["generation"]["sampling_seed_ledger"]:
        arm = Arm(
            name=f"sample_t07_p09_seed{int(seed)}",
            decode_mode="sample",
            max_new_tokens=int(sc["sampling"]["max_new_tokens"]),
            seed=int(seed),
            temperature=float(sc["sampling"]["temperature"]),
            top_p=float(sc["sampling"]["top_p"]),
        )
        if arm.name not in existing:
            arms.append(arm)
            existing.add(arm.name)
    arms.append(
        Arm(
            name="replay_t07_p09_seed42",
            decode_mode="sample",
            max_new_tokens=int(sc["sampling"]["max_new_tokens"]),
            seed=42,
            temperature=float(sc["sampling"]["temperature"]),
            top_p=float(sc["sampling"]["top_p"]),
        )
    )
    return arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu", "llava"), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    contract = json.loads(args.execution_contract.read_text())
    arms = frozen_arms(contract)
    rows = load_rows(args.manifest, args.limit)
    expected = [qid(row) for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("manifest qids are not unique")

    pending: list[tuple[Arm, Path, list[dict[str, Any]], str]] = []
    for arm in arms:
        output_dir = args.output_root / arm.name
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "protocol": VERSION,
            "model": args.model,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "image_root": str(args.image_root.resolve()),
            "limit": args.limit,
            "max_new_tokens": arm.max_new_tokens,
            "seed": arm.seed,
            "prompt": "exact source question; model-native image placeholder and chat template",
            "generation": arm.generation,
            "arm": arm.name,
            "execution_contract_sha256": sha256_file(args.execution_contract),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "adapter_code_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "corrected_sgta" / "models_oe.py"
            ),
            "model_native_fixed": (
                {"min_new_tokens": 1, "repetition_penalty": 1.2}
                if args.model == "huatuo"
                else {}
            ),
        }
        fingerprint = sha256_json(config)
        config_path = output_dir / "generation_config.json"
        if config_path.exists():
            prior = json.loads(config_path.read_text())
            if prior.get("fingerprint") != fingerprint:
                raise ValueError(f"refusing incompatible resume: {output_dir}")
        else:
            config_path.write_text(json.dumps({**config, "fingerprint": fingerprint}, indent=2) + "\n")
        completed = load_resume(output_dir / "answers.jsonl", expected)
        if len(completed) < len(rows):
            pending.append((arm, output_dir, completed, fingerprint))

    if not pending:
        print("all frozen control arms already complete", flush=True)
        return

    from anchor.corrected_sgta.models_oe import load_oe_adapter

    adapter = load_oe_adapter(args.model, llava_conv_mode="mistral_instruct")
    try:
        for arm_index, (arm, output_dir, completed, fingerprint) in enumerate(pending, 1):
            answers_path = output_dir / "answers.jsonl"
            with answers_path.open("a") as handle:
                for index, row in enumerate(rows[len(completed) :], len(completed)):
                    item_id = expected[index]
                    sample_seed = stable_seed(arm.seed, item_id)
                    with Image.open(args.image_root / str(row["img_name"])) as source:
                        image = source.convert("RGB")
                    result = adapter.generate_control(
                        image=image,
                        prompt=str(row["question"]),
                        do_sample=arm.generation["do_sample"],
                        temperature=arm.temperature,
                        top_p=arm.top_p,
                        num_beams=1,
                        max_new_tokens=arm.max_new_tokens,
                        seed=sample_seed,
                    )
                    if not result.text.strip():
                        raise RuntimeError(f"empty generation for {arm.name}/{item_id}")
                    record = {
                        "question_id": item_id,
                        "text": result.text,
                        "gt_ans": str(row["answer"]),
                        "model_id": args.model,
                        "metadata": {
                            "generated_token_count": result.token_count,
                            "generated_token_ids": list(result.token_ids),
                            "hit_max_new_tokens": result.token_count >= arm.max_new_tokens,
                            "stop_reason": (
                                "length" if result.token_count >= arm.max_new_tokens else "eos_or_template"
                            ),
                            "mean_token_nll": result.uncertainty,
                            "base_seed": arm.seed,
                            "sample_seed": sample_seed,
                            "fingerprint": fingerprint,
                        },
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    print(
                        f"[{arm_index}/{len(pending)} arms][{index + 1}/{len(rows)}] "
                        f"{args.model}/{arm.name}/{item_id}",
                        flush=True,
                    )
            final = load_resume(answers_path, expected)
            if len(final) != len(rows):
                raise RuntimeError(f"incomplete arm {arm.name}: {len(final)}/{len(rows)}")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
