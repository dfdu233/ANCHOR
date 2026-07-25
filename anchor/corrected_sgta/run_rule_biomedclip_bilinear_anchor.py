"""Extract source features, train LODO bilinear anchors, and apply the gate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from corrected_sgta.rule_biomedclip_bilinear_anchor import (
    DOMAIN_NAMES,
    VERSION,
    ResidualBilinearAnchor,
    anchor_texts,
    answer_index,
    assert_disjoint_by_image,
    balanced_bce,
    deterministic_permutation,
    fold_metrics,
    sha256_file,
)


BIOMEDCLIP_ROOT = Path("/root/autodl-tmp/BiomedCLIP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--vlm-dev", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_biomedclip():
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import _MODEL_CONFIGS

    config = json.loads((BIOMEDCLIP_ROOT / "open_clip_config.json").read_text())
    _MODEL_CONFIGS["biomedclip_local"] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        "biomedclip_local",
        pretrained=str(BIOMEDCLIP_ROOT / "open_clip_pytorch_model.bin"),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    return model, preprocess, get_tokenizer("biomedclip_local")


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def _payload_fingerprint(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    payload = {
        "version": VERSION,
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "vlm_dev_sha256": sha256_file(args.vlm_dev),
        "weights_sha256": sha256_file(
            BIOMEDCLIP_ROOT / "open_clip_pytorch_model.bin"
        ),
        "config_sha256": sha256_file(BIOMEDCLIP_ROOT / "open_clip_config.json"),
        "rank": args.rank,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


@torch.inference_mode()
def _extract(
    rows: list[dict[str, Any]],
    *,
    model,
    preprocess,
    tokenizer,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    image_features: list[torch.Tensor] = []
    directions: list[torch.Tensor] = []
    logit_scale = float(model.logit_scale.exp().detach().cpu())
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        texts: list[str] = []
        for row in batch:
            with Image.open(row["image"]) as image:
                images.append(preprocess(image.convert("RGB")))
            texts.extend(anchor_texts(row["conversations"][0]["value"]))
        image = torch.stack(images).to(device)
        tokens = tokenizer(texts, context_length=256).to(device)
        encoded_image = F.normalize(model.encode_image(image), dim=-1)
        encoded_text = F.normalize(model.encode_text(tokens), dim=-1)
        encoded_text = encoded_text.reshape(len(batch), 2, -1)
        image_features.append(encoded_image.cpu())
        directions.append((encoded_text[:, 0] - encoded_text[:, 1]).cpu())
        print(f"feature extraction {min(start + len(batch), len(rows))}/{len(rows)}")
    return {
        "image": torch.cat(image_features),
        "direction": torch.cat(directions),
        "label": torch.tensor(
            [answer_index(row["conversations"][1]["value"]) for row in rows],
            dtype=torch.float32,
        ),
        "domain": torch.tensor(
            [DOMAIN_NAMES.index(row["source_domain"]) for row in rows],
            dtype=torch.long,
        ),
        "ids": [str(row["id"]) for row in rows],
        "logit_scale": logit_scale,
    }


def _load_vlm_margins(path: Path) -> dict[str, float]:
    records = json.loads(path.read_text())["records"]
    result: dict[str, float] = {}
    for domain_records in records.values():
        for row in domain_records:
            values = row["sequence_log_probabilities"]["identity"]
            result[str(row["id"])] = float(values["Yes."] - values["No."])
    return result


def _train_fold(
    *,
    features: dict[str, Any],
    held_domain: int,
    rank: int,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> ResidualBilinearAnchor:
    mask = features["domain"] != held_domain
    image = features["image"][mask].to(device)
    direction = features["direction"][mask].to(device)
    labels = features["label"][mask].to(device)
    model = ResidualBilinearAnchor(image.shape[-1], rank, seed=seed + held_domain)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            image, direction, logit_scale=features["logit_scale"]
        )
        loss = balanced_bce(logits, labels)
        # Keep the learned metric a residual correction around BiomedCLIP.
        loss = loss + 1e-4 * (
            model.left.square().mean() + model.right.square().mean()
        )
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 100 == 0:
            print(
                f"fold={DOMAIN_NAMES[held_domain]} step={step + 1}/{steps} "
                f"loss={float(loss):.6f}"
            )
    return model.eval()


def main() -> None:
    args = parse_args()
    fingerprint, fingerprint_payload = _payload_fingerprint(args)
    train_rows = _rows(args.train)
    dev_rows = _rows(args.dev)
    assert_disjoint_by_image(train_rows, dev_rows)
    if set(row["source_domain"] for row in train_rows + dev_rows) != set(
        DOMAIN_NAMES
    ):
        raise ValueError("Expected exactly the three frozen source domains")

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache_meta = args.cache.with_suffix(args.cache.suffix + ".meta.json")
    cache_ok = (
        args.cache.exists()
        and cache_meta.exists()
        and json.loads(cache_meta.read_text()).get("fingerprint") == fingerprint
    )
    if cache_ok:
        cached = torch.load(args.cache, map_location="cpu", weights_only=True)
        train_features = {
            key.removeprefix("train_"): value
            for key, value in cached.items()
            if key.startswith("train_")
        }
        dev_features = {
            key.removeprefix("dev_"): value
            for key, value in cached.items()
            if key.startswith("dev_")
        }
        train_features["ids"] = json.loads(cache_meta.read_text())["train_ids"]
        dev_features["ids"] = json.loads(cache_meta.read_text())["dev_ids"]
        print(f"reused feature cache {args.cache}")
    else:
        device = torch.device("cuda")
        model, preprocess, tokenizer = _load_biomedclip()
        model = model.to(device).eval()
        train_features = _extract(
            train_rows,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device,
        )
        dev_features = _extract(
            dev_rows,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device=device,
        )
        tensor_payload = {
            **{
                f"train_{key}": value
                for key, value in train_features.items()
                if key != "ids"
            },
            **{
                f"dev_{key}": value
                for key, value in dev_features.items()
                if key != "ids"
            },
        }
        torch.save(tensor_payload, args.cache)
        cache_meta.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                    "train_ids": train_features["ids"],
                    "dev_ids": dev_features["ids"],
                },
                indent=2,
            )
        )
        del model
        torch.cuda.empty_cache()

    vlm_by_id = _load_vlm_margins(args.vlm_dev)
    if set(dev_features["ids"]) != set(vlm_by_id):
        raise ValueError("Frozen dev IDs do not exactly match VLM-margin records")
    vlm_margin = torch.tensor(
        [vlm_by_id[item_id] for item_id in dev_features["ids"]]
    )
    device = torch.device("cuda")
    folds: dict[str, Any] = {}
    predictions = torch.zeros(len(dev_rows), dtype=torch.bool)
    baseline = vlm_margin > 0
    for held_domain, domain_name in enumerate(DOMAIN_NAMES):
        anchor = _train_fold(
            features=train_features,
            held_domain=held_domain,
            rank=args.rank,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        mask = dev_features["domain"] == held_domain
        image = dev_features["image"][mask].to(device)
        direction = dev_features["direction"][mask].to(device)
        labels = dev_features["label"][mask]
        with torch.inference_mode():
            anchor_margin = anchor(
                image,
                direction,
                logit_scale=dev_features["logit_scale"],
            ).cpu()
            permutation = deterministic_permutation(
                len(image), args.seed + 100 + held_domain
            )
            image_shuffle_margin = anchor(
                image[permutation],
                direction,
                logit_scale=dev_features["logit_scale"],
            ).cpu()
            text_shuffle_margin = anchor(
                image,
                direction[permutation],
                logit_scale=dev_features["logit_scale"],
            ).cpu()
        train_mask = train_features["domain"] != held_domain
        train_labels = train_features["label"][train_mask]
        positive = float(train_labels.sum())
        negative = float(len(train_labels) - positive)
        prior = math.log((positive + 0.5) / (negative + 0.5))
        metrics = fold_metrics(
            labels=labels,
            vlm_margin=vlm_margin[mask],
            anchor_margin=anchor_margin,
            image_shuffle_margin=image_shuffle_margin,
            text_shuffle_margin=text_shuffle_margin,
            bias_log_odds=prior,
        )
        predictions[mask] = vlm_margin[mask] + anchor_margin > 0
        folds[domain_name] = dataclasses.asdict(metrics)
        folds[domain_name]["bias_log_odds"] = prior

    labels = dev_features["label"].bool()
    rescues = int((~(baseline == labels) & (predictions == labels)).sum())
    harms = int(((baseline == labels) & ~(predictions == labels)).sum())
    baseline_accuracy = float((baseline == labels).float().mean())
    fused_accuracy = float((predictions == labels).float().mean())
    controls = [
        sum(folds[name][key] * folds[name]["n"] for name in DOMAIN_NAMES)
        / len(labels)
        for key in (
            "image_shuffle_accuracy",
            "text_shuffle_accuracy",
            "bias_only_accuracy",
        )
    ]
    domain_nondecline = sum(
        folds[name]["fused_accuracy"] >= folds[name]["baseline_accuracy"]
        for name in DOMAIN_NAMES
    )
    gate = {
        "net_rescues_at_least_3": rescues - harms >= 3,
        "at_least_two_domains_nondeclining": domain_nondecline >= 2,
        "harms_not_above_rescues": harms <= rescues,
        "beats_all_controls_by_3pp": fused_accuracy >= max(controls) + 0.03,
    }
    result = {
        "version": VERSION,
        "status": "pass" if all(gate.values()) else "fail",
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "method_scope": "external_teacher_fallback",
        "target_labels_read": False,
        "train_n": len(train_rows),
        "dev_n": len(dev_rows),
        "baseline_accuracy": baseline_accuracy,
        "fused_accuracy": fused_accuracy,
        "delta_pp": 100.0 * (fused_accuracy - baseline_accuracy),
        "rescues": rescues,
        "harms": harms,
        "controls_micro_accuracy": {
            "image_shuffle": controls[0],
            "text_shuffle": controls[1],
            "bias_only": controls[2],
        },
        "folds": folds,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

