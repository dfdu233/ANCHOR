#!/usr/bin/env python3
"""Single-image visual-edge constraint probe for VQA-RAD natural pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
import types
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from corrected_sgta.clinical_claims import softmax_states
from corrected_sgta.cecd_positional_prefix_attention_v1 import (
    redistribute_post_softmax_attention,
)
from corrected_sgta.cecd_system_pih_runtime_integration_v1 import (
    EXPECTED_LAYERS,
    MODEL_GEOMETRIES,
    RuntimeIntegrationError,
    _forward_globals,
    _module_geometry,
    _qwen2_forward,
    resolve_decoder_layers,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    VERBALIZERS,
    append_jsonl,
    atomic_json,
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    sha256_file,
)
from corrected_sgta.run_vqa_rad_natural_counterfactual_pilot import analyze as analyze_source_pairs
from corrected_sgta.run_vqa_rad_underidentification_pilot import (
    PROMPT_TEMPLATES,
    auc,
    entropy,
    js_divergence,
    zscore,
)


VERSION = "vqa-rad-visual-edge-constraint-v1"


class LastQueryImageEdgeSession:
    """Remove image-key attention for selected prompt query rows."""

    def __init__(
        self,
        *,
        prefix_length: int,
        image_span: tuple[int, int],
        query_scope: str = "last",
    ) -> None:
        self.prefix_length = int(prefix_length)
        self.image_span = (int(image_span[0]), int(image_span[1]))
        if query_scope not in {"last", "suffix_after_image"}:
            raise RuntimeIntegrationError("unknown visual-edge query scope")
        self.query_scope = query_scope
        self.prefill_seen = False
        self.patched_rows = 0
        self.source_mass_before: list[float] = []
        self.source_mass_after: list[float] = []

    def target_queries(self) -> tuple[int, ...]:
        if self.query_scope == "last":
            return (self.prefix_length - 1,)
        return tuple(range(self.image_span[1], self.prefix_length))

    def apply_chunk(
        self,
        weights: torch.Tensor,
        *,
        query_start: int,
        total_query_length: int,
    ) -> torch.Tensor:
        query_length, key_length = weights.shape[-2:]
        prefix = self.prefix_length
        if total_query_length == prefix and key_length == prefix:
            targets = self.target_queries()
            query_stop = query_start + query_length
            visible_targets = tuple(
                target for target in targets if query_start <= target < query_stop
            )
            if not visible_targets:
                return weights
            if self.prefill_seen:
                raise RuntimeIntegrationError("a sample cannot execute two full prefills")
            start, end = self.image_span
            source = tuple(range(start, end))
            recipients = (tuple(index for index in range(key_length) if index not in source),)
            transformed = weights
            for target in visible_targets:
                transformed, diagnostics = redistribute_post_softmax_attention(
                    transformed,
                    source_keys=source,
                    recipient_groups=recipients,
                    query_index=target - query_start,
                    alpha=0.0,
                    variant="redistribute",
                )
                self.patched_rows += diagnostics.selected_rows
                self.source_mass_before.extend(diagnostics.source_mass_before)
                self.source_mass_after.extend(diagnostics.source_mass_after)
            self.prefill_seen = True
            return transformed
        if total_query_length == 1 and query_length == 1 and key_length > prefix:
            if not self.prefill_seen:
                raise RuntimeIntegrationError("cached decode observed before patched prefill")
            return weights
        raise RuntimeIntegrationError(
            "unsupported attention shape for visual-edge block "
            f"chunk_Q={query_length}, total_Q={total_query_length}, "
            f"start={query_start}, K={key_length}, prefix={prefix}"
        )


class HuatuoVisualEdgeBlockContext(AbstractContextManager["HuatuoVisualEdgeBlockContext"]):
    """Instance-local Huatuo Qwen2 eager attention patch."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        prefix_length: int,
        image_span: tuple[int, int],
        query_scope: str = "last",
        layers: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.prefix_length = int(prefix_length)
        self.image_span = image_span
        self.query_scope = query_scope
        self.layers = tuple(range(EXPECTED_LAYERS["huatuo"])) if layers is None else tuple(int(x) for x in layers)
        if len(set(self.layers)) != len(self.layers) or any(
            not 0 <= layer < EXPECTED_LAYERS["huatuo"] for layer in self.layers
        ):
            raise RuntimeIntegrationError("visual-edge layer selection is duplicated/out of bounds")
        self._restorers: list[Callable[[], None]] = []
        self.sessions: dict[int, LastQueryImageEdgeSession] = {}

    def __enter__(self) -> "HuatuoVisualEdgeBlockContext":
        decoder_layers = resolve_decoder_layers(self.model, "huatuo")
        try:
            for index, layer in enumerate(decoder_layers):
                attention = layer.self_attn
                geometry = _module_geometry(attention, "huatuo")
                original = attention.forward
                had_instance_forward = "forward" in attention.__dict__
                instance_forward = attention.__dict__.get("forward")
                session = None
                if index in self.layers:
                    session = LastQueryImageEdgeSession(
                        prefix_length=self.prefix_length,
                        image_span=self.image_span,
                        query_scope=self.query_scope,
                    )
                    self.sessions[index] = session

                def replacement(
                    module: torch.nn.Module,
                    *args: Any,
                    _original=original,
                    _geometry=geometry,
                    _session=session,
                    **kwargs: Any,
                ):
                    if args:
                        if "hidden_states" in kwargs:
                            raise RuntimeIntegrationError("hidden_states supplied twice")
                        kwargs["hidden_states"] = args[0]
                        if len(args) > 1:
                            raise RuntimeIntegrationError("positional Qwen2 arguments beyond hidden_states are forbidden")
                    return _qwen2_forward(
                        module,
                        original_forward=_original,
                        geometry=_geometry,
                        session=_session,
                        **kwargs,
                    )

                attention.forward = types.MethodType(replacement, attention)

                def restore(target=attention, had=had_instance_forward, prior=instance_forward) -> None:
                    if had:
                        target.forward = prior
                    else:
                        target.__dict__.pop("forward", None)

                self._restorers.append(restore)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()

    def __exit__(self, *_args: Any) -> None:
        self._restore()

    def diagnostics(self) -> dict[str, Any]:
        masses_before = [value for session in self.sessions.values() for value in session.source_mass_before]
        masses_after = [value for session in self.sessions.values() for value in session.source_mass_after]
        return {
            "patched_layers": len(self.sessions),
            "patched_rows": sum(session.patched_rows for session in self.sessions.values()),
            "query_scope": self.query_scope,
            "mean_image_attention_before": float(np.mean(masses_before)) if masses_before else None,
            "max_image_attention_after": float(max(masses_after)) if masses_after else None,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def margin(logits: Mapping[str, float], positive: str, negative: str) -> float:
    return float(logits[positive]) - float(logits[negative])


def selected_margin(logits: Mapping[str, float]) -> tuple[str, str, float]:
    ordered = sorted(logits, key=lambda key: float(logits[key]), reverse=True)
    return ordered[0], ordered[1], margin(logits, ordered[0], ordered[1])


@torch.inference_mode()
def score_prompt(bot: Any, image: Image.Image, prompt: str, query_scope: str) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, image_span = prepared_embeddings(bot, prompt, tensor)
    ids = label_ids(bot)
    full_hidden = hidden_trajectory(bot, embeddings, attention, positions)
    final = len(full_hidden) - 1
    full_logits = layer_logits(bot, full_hidden, [final], ids)[final]
    with HuatuoVisualEdgeBlockContext(
        bot.model,
        prefix_length=int(embeddings.shape[1]),
        image_span=image_span,
        query_scope=query_scope,
    ) as context:
        blocked_hidden = hidden_trajectory(bot, embeddings, attention, positions)
        diagnostics = context.diagnostics()
    blocked_logits = layer_logits(bot, blocked_hidden, [final], ids)[final]
    selected, alternative, full_selected_margin = selected_margin(full_logits)
    blocked_selected_margin = margin(blocked_logits, selected, alternative)
    polarity_full = margin(full_logits, "supported", "refuted")
    polarity_blocked = margin(blocked_logits, "supported", "refuted")
    probabilities = softmax_states(full_logits)
    return {
        "full_logits": full_logits,
        "visual_blocked_logits": blocked_logits,
        "probabilities": probabilities,
        "state": max(probabilities, key=probabilities.get),
        "selected_state": selected,
        "selected_alternative": alternative,
        "selected_margin_full": full_selected_margin,
        "selected_margin_visual_blocked": blocked_selected_margin,
        "selected_visual_support": float(full_selected_margin - blocked_selected_margin),
        "polarity_margin_full": polarity_full,
        "polarity_margin_visual_blocked": polarity_blocked,
        "polarity_visual_delta": float(polarity_full - polarity_blocked),
        "visual_tokens": int(image_span[1] - image_span[0]),
        "attention_diagnostics": diagnostics,
    }


def spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None

    def ranks(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        order = np.argsort(array, kind="mergesort")
        out = np.empty(len(array), dtype=float)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and array[order[stop]] == array[order[start]]:
                stop += 1
            out[order[start:stop]] = (start + 1 + stop) / 2.0
            start = stop
        return out

    rx, ry = ranks(x_values), ranks(y_values)
    if float(rx.std()) <= 1e-12 or float(ry.std()) <= 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze(records: Sequence[Mapping[str, Any]], source_records: Sequence[Mapping[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    source_summary = analyze_source_pairs(source_records, seed, draws=0)
    source_by_pair = {row["pair_id"]: row for row in source_summary["derived_pairs"]}
    source_raw_by_pair = {str(row["pair_id"]): dict(row) for row in source_records}
    rows = [dict(row) for row in records if row.get("status") == "ok"]
    if not rows:
        raise ValueError("no successful records")
    states = ("supported", "refuted", "undetermined")
    images = []
    pairs = []
    for row in rows:
        pair_scores = {}
        for role, expected_state, sign in (
            ("positive", "supported", 1.0),
            ("negative", "refuted", -1.0),
        ):
            canonical = row["scores"][role]["canonical"]
            error = int(canonical["state"] != expected_state)
            pair_scores[role] = {
                "error": error,
                "entropy": entropy([float(canonical["probabilities"][state]) for state in states]),
                "selected_visual_support": float(canonical["selected_visual_support"]),
                "truth_aligned_visual_support": float(sign * canonical["polarity_visual_delta"]),
                "abs_polarity_visual_delta": abs(float(canonical["polarity_visual_delta"])),
                "image_attention_before": canonical["attention_diagnostics"]["mean_image_attention_before"],
            }
            images.append({
                "pair_id": row["pair_id"],
                "role": role,
                "error": error,
                **pair_scores[role],
            })
        positive = pair_scores["positive"]
        negative = pair_scores["negative"]
        source = source_by_pair.get(row["pair_id"], {})
        source_raw = source_raw_by_pair.get(row["pair_id"], {})
        if source_raw:
            positive_vectors = np.asarray([
                [float(source_raw["scores"]["positive"][name]["probabilities"][state]) for state in states]
                for name in PROMPT_TEMPLATES
            ])
            negative_vectors = np.asarray([
                [float(source_raw["scores"]["negative"][name]["probabilities"][state]) for state in states]
                for name in PROMPT_TEMPLATES
            ])
            oracle_between_image_js = js_divergence(
                (positive_vectors.mean(axis=0), negative_vectors.mean(axis=0))
            )
        else:
            oracle_between_image_js = None
        pairs.append({
            "pair_id": row["pair_id"],
            "any_error": int(positive["error"] or negative["error"]),
            "mean_error": float((positive["error"] + negative["error"]) / 2.0),
            "oracle_between_image_js": oracle_between_image_js,
            "source_underidentification_score": source.get("underidentification_score"),
            "source_language_js": source.get("language_js"),
            "source_natural_directional_response": source.get("natural_directional_response"),
            "source_baseline_mean_entropy": source.get("baseline_mean_entropy"),
            "visual_edge_directional_contrast": float(
                row["scores"]["positive"]["canonical"]["polarity_visual_delta"]
                - row["scores"]["negative"]["canonical"]["polarity_visual_delta"]
            ),
            "mean_selected_visual_support": float(
                (positive["selected_visual_support"] + negative["selected_visual_support"]) / 2.0
            ),
            "mean_truth_aligned_visual_support": float(
                (positive["truth_aligned_visual_support"] + negative["truth_aligned_visual_support"]) / 2.0
            ),
            "mean_entropy": float((positive["entropy"] + negative["entropy"]) / 2.0),
        })
    image_errors = [int(row["error"]) for row in images]
    pair_errors = [int(row["any_error"]) for row in pairs]
    visual_pair = np.asarray([row["visual_edge_directional_contrast"] for row in pairs], dtype=float)
    selected_pair = np.asarray([row["mean_selected_visual_support"] for row in pairs], dtype=float)
    truth_pair = np.asarray([row["mean_truth_aligned_visual_support"] for row in pairs], dtype=float)
    oracle_values = [
        float(row["oracle_between_image_js"])
        for row in pairs
        if row.get("oracle_between_image_js") is not None
    ]
    paired_for_oracle = [
        row for row in pairs if row.get("oracle_between_image_js") is not None
    ]
    risk = zscore([-value for value in selected_pair]) + zscore([-value for value in truth_pair])
    for row, value in zip(pairs, risk):
        row["visual_edge_risk"] = float(value)
    metrics = {
        "n_pairs": len(pairs),
        "n_images": len(images),
        "pair_any_error_rate": float(np.mean(pair_errors)),
        "image_error_rate": float(np.mean([row["mean_error"] for row in pairs])),
        "mean_visual_edge_directional_contrast": float(visual_pair.mean()),
        "mean_selected_visual_support": float(selected_pair.mean()),
        "mean_truth_aligned_visual_support": float(truth_pair.mean()),
        "spearman_with_between_image_js_oracle": {
            "negative_directional_contrast": spearman(
                [-float(row["visual_edge_directional_contrast"]) for row in paired_for_oracle],
                oracle_values,
            ),
            "negative_selected_support": spearman(
                [-float(row["mean_selected_visual_support"]) for row in paired_for_oracle],
                oracle_values,
            ),
            "visual_edge_risk": spearman(
                [float(row["visual_edge_risk"]) for row in paired_for_oracle],
                oracle_values,
            ),
        },
        "pair_error_auroc": {
            "negative_visual_edge_directional_contrast": auc(pair_errors, -visual_pair),
            "negative_selected_visual_support": auc(pair_errors, -selected_pair),
            "negative_truth_aligned_visual_support": auc(pair_errors, -truth_pair),
            "visual_edge_risk": auc(pair_errors, [row["visual_edge_risk"] for row in pairs]),
            "mean_entropy": auc(pair_errors, [row["mean_entropy"] for row in pairs]),
        },
        "image_error_auroc": {
            "negative_selected_visual_support": auc(
                image_errors, [-float(row["selected_visual_support"]) for row in images]
            ),
            "negative_truth_aligned_visual_support": auc(
                image_errors, [-float(row["truth_aligned_visual_support"]) for row in images]
            ),
            "entropy": auc(image_errors, [float(row["entropy"]) for row in images]),
        },
    }
    order = np.argsort([row["visual_edge_risk"] for row in pairs])
    quartile = max(1, len(order) // 4)
    metrics["visual_edge_risk_quartiles"] = {
        "quartile_n": quartile,
        "lowest_pair_error_rate": float(np.mean([pair_errors[index] for index in order[:quartile]])),
        "highest_pair_error_rate": float(np.mean([pair_errors[index] for index in order[-quartile:]])),
    }
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in metrics["pair_error_auroc"]}
    for _ in range(draws):
        indices = rng.integers(0, len(pairs), len(pairs))
        sampled_errors = [pair_errors[index] for index in indices]
        for name in boot:
            if name == "negative_visual_edge_directional_contrast":
                scores = [-visual_pair[index] for index in indices]
            elif name == "negative_selected_visual_support":
                scores = [-selected_pair[index] for index in indices]
            elif name == "negative_truth_aligned_visual_support":
                scores = [-truth_pair[index] for index in indices]
            elif name == "visual_edge_risk":
                scores = [pairs[index]["visual_edge_risk"] for index in indices]
            else:
                scores = [pairs[index]["mean_entropy"] for index in indices]
            value = auc(sampled_errors, scores)
            if value is not None:
                boot[name].append(value)
    metrics["pair_error_auroc_bootstrap"] = {
        name: {
            "valid_draws": len(values),
            "ci_low": float(np.quantile(values, 0.025)) if values else None,
            "ci_high": float(np.quantile(values, 0.975)) if values else None,
        }
        for name, values in boot.items()
    }
    return {"metrics": metrics, "derived_pairs": pairs, "derived_images": images}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--seed", type=int, default=260814)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--query-scope", choices=("last", "suffix_after_image"), default="last")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_path} exists; use --resume or a new output directory")
    source_records = [row for row in read_jsonl(args.source_raw) if row.get("status") == "ok"]
    if args.limit_pairs is not None:
        source_records = source_records[: args.limit_pairs]
    completed = {
        str(json.loads(line)["pair_id"])
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("status") == "ok"
    } if args.resume and raw_path.exists() else set()
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_raw": str(args.source_raw.resolve()),
        "source_raw_sha256": sha256_file(args.source_raw),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "image_root": str(args.image_root.resolve()),
        "model": str(args.model_dir.resolve()),
        "method": "white-box inference-only last-prompt-query visual attention edge block",
        "query_scope": args.query_scope,
        "prompt_templates": {"canonical": PROMPT_TEMPLATES["canonical"]},
        "pairs": len(source_records),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "code_sha256": sha256_file(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output_dir / "config.json", config)
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    for index, source in enumerate(source_records):
        pair_id = str(source["pair_id"])
        if pair_id in completed:
            continue
        record: dict[str, Any] = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "pair_id": pair_id,
            "question": source["question"],
            "positive_qid": source["positive_qid"],
            "negative_qid": source["negative_qid"],
            "positive_image": source["positive_image"],
            "negative_image": source["negative_image"],
            "status": "error",
        }
        try:
            scores: dict[str, Any] = {}
            for role in ("positive", "negative"):
                image_path = args.image_root / str(source[f"{role}_image"])
                image = Image.open(image_path).convert("RGB")
                prompt = PROMPT_TEMPLATES["canonical"].format(question=str(source["question"]).strip())
                scores[role] = {"canonical": score_prompt(bot, image, prompt, args.query_scope)}
            record["scores"] = scores
            record["status"] = "ok"
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record["error"] = f"CUDA OOM: {error}"
            record["traceback"] = traceback.format_exc()
        except Exception as error:
            record["error"] = repr(error)
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(json.dumps({"progress": f"{index + 1}/{len(source_records)}", "pair_id": pair_id, "status": record["status"], "error": record.get("error")}), flush=True)
    records = read_jsonl(raw_path)
    summary = analyze(records, source_records, args.seed, args.bootstrap_draws)
    summary["version"] = VERSION
    summary["config"] = config
    summary["runtime_errors"] = sum(row.get("status") != "ok" for row in records)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
