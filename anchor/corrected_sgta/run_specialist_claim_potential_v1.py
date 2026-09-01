#!/usr/bin/env python3
"""Small specialist guidance attached to clinical claims rather than words.

This is a resumable VinDr canary.  A frozen XRV DenseNet is calibrated on the
frozen development claims.  During report generation its evidence is applied
in two ways:

* ``ccd_token``: polarity-blind disease-token bias, matching CCD's ECD primitive;
* ``claim_potential``: apply the evidence once, when a positive or negative
  clinical claim is completed.

The canary is deliberately small and does not claim clinical efficacy.  Its
purpose is to test whether semantic, one-use guidance changes real generations
without the word/polarity ambiguity documented by the companion audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import LogitsProcessor, LogitsProcessorList

from anchor.corrected_sgta.analyze_xrv_specialist_error_geometry_v2 import (
    FINDINGS,
    XRV_LABELS,
    load_claims,
    load_logits,
)
from anchor.corrected_sgta.models_oe import (
    Generation,
    HuatuoOEAdapter,
    HuluOEAdapter,
    _decode_generations,
)

PROTOCOL = "specialist-claim-potential-v1"
ALIASES = {
    "aortic_enlargement": ("aortic enlargement", "enlarged aorta"),
    "cardiomegaly": ("cardiomegaly", "enlarged cardiac silhouette"),
    "lung_opacity": ("lung opacity", "pulmonary opacity", "airspace opacity"),
    "nodule_mass": ("nodule", "mass", "lung lesion"),
    "pleural_effusion": ("pleural effusion", "effusion"),
    "pleural_thickening": ("pleural thickening",),
    "pulmonary_fibrosis": ("pulmonary fibrosis", "fibrosis", "fibrotic change"),
}
NEGATION = re.compile(
    r"\b(no|not|without|absent|absence of|negative for|free of|neither|nor|resolved|clear of)\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dicom_to_pil(path: Path) -> Image.Image:
    dataset = pydicom.dcmread(path, force=True)
    array = dataset.pixel_array.astype(np.float32)
    high = float(2 ** int(dataset.BitsStored) - 1)
    if str(dataset.PhotometricInterpretation).upper() == "MONOCHROME1":
        array = high - array
    array = np.clip(array / max(high, 1.0), 0.0, 1.0)
    return Image.fromarray(np.uint8(np.rint(array * 255.0)), mode="L").convert("RGB")


def encode_phrases(tokenizer: Any) -> dict[str, list[tuple[int, ...]]]:
    output = {}
    for finding, aliases in ALIASES.items():
        sequences = set()
        for alias in aliases:
            for surface in (alias, " " + alias, alias.capitalize(), " " + alias.capitalize()):
                ids = tuple(tokenizer.encode(surface, add_special_tokens=False))
                if ids:
                    sequences.add(ids)
        output[finding] = sorted(sequences, key=lambda value: (len(value), value))
    return output


def mentioned_findings(text: str) -> set[str]:
    lowered = text.lower()
    return {
        finding
        for finding, aliases in ALIASES.items()
        if any(re.search(r"\b" + re.escape(alias) + r"\b", lowered) for alias in aliases)
    }


def surface_claims(text: str) -> dict[str, str]:
    lowered = text.lower()
    output = {}
    for finding, aliases in ALIASES.items():
        matches = []
        for alias in sorted(aliases, key=len, reverse=True):
            matches.extend(re.finditer(r"\b" + re.escape(alias) + r"\b", lowered))
        if not matches:
            continue
        match = min(matches, key=lambda value: value.start())
        context = re.split(r"[.;\n]", lowered[max(0, match.start() - 90) : match.start()])[-1]
        output[finding] = "negative" if NEGATION.search(context) else "positive"
    return output


class SpecialistGuidance(LogitsProcessor):
    """Apply the same specialist odds through one of two interfaces."""

    def __init__(
        self,
        tokenizer: Any,
        probabilities: dict[str, float],
        prompt_length: int,
        mode: str,
        beta: float,
        gamma: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.probabilities = probabilities
        self.prompt_length = prompt_length
        self.mode = mode
        self.beta = beta
        self.maximum = math.log(gamma)
        self.phrases = encode_phrases(tokenizer)
        self.steps = 0
        self.adjusted_candidates = 0

        self.token_bias: dict[int, float] = {}
        if mode == "ccd_token":
            for finding, sequences in self.phrases.items():
                p = min(max(probabilities[finding], 1e-6), 1 - 1e-6)
                value = beta * max(-self.maximum, min(self.maximum, math.log(p / (1 - p))))
                # CCD's default only_first_token=True biases the first token of
                # every word.  Using all word-starts would require tokenizer-
                # specific boundary parsing; the first disease-phrase token is
                # the conservative common subset and is explicitly reported.
                for sequence in sequences:
                    old = self.token_bias.get(sequence[0])
                    if old is None or abs(value) > abs(old):
                        self.token_bias[sequence[0]] = value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.steps += 1
        if input_ids.shape[0] != 1:
            raise ValueError("The canary supports greedy batch size one only")
        if self.mode == "ccd_token":
            for token_id, value in self.token_bias.items():
                scores[0, token_id] += value
            self.adjusted_candidates += len(self.token_bias)
            return scores

        generated = input_ids[0, self.prompt_length :].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        already = mentioned_findings(text)
        for finding, sequences in self.phrases.items():
            if finding in already:
                continue
            p = min(max(self.probabilities[finding], 1e-6), 1 - 1e-6)
            log_odds = max(-self.maximum, min(self.maximum, math.log(p / (1 - p))))
            adjusted_final_tokens: set[int] = set()
            for sequence in sequences:
                prefix, final = sequence[:-1], sequence[-1]
                # Capitalization/leading-space variants can collapse to the
                # same completion token and prefix.  Clinical evidence belongs
                # to the completed claim once, never once per lexical alias.
                if final in adjusted_final_tokens:
                    continue
                if len(prefix) > len(generated) or (prefix and tuple(generated[-len(prefix) :]) != prefix):
                    continue
                before = generated[: len(generated) - len(prefix)] if prefix else generated
                context = self.tokenizer.decode(before[-16:], skip_special_tokens=True)
                polarity_sign = -1.0 if NEGATION.search(context) else 1.0
                scores[0, final] += self.beta * polarity_sign * log_odds
                adjusted_final_tokens.add(final)
                self.adjusted_candidates += 1
        return scores


def fit_specialist_heads(
    development: list[dict[str, Any]], xrv: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    heads = {}
    for finding in FINDINGS:
        rows = [row for row in development if row["finding"] == finding and row["image_id"] in xrv]
        matrix = np.stack([xrv[row["image_id"]] for row in rows])
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        model = LogisticRegression(C=1.0, max_iter=10_000).fit((matrix - mean) / scale, labels)
        heads[finding] = {"model": model, "mean": mean, "scale": scale, "n": len(rows)}
    return heads


def specialist_probabilities(
    image_id: str, heads: dict[str, dict[str, Any]], xrv: dict[str, np.ndarray]
) -> dict[str, float]:
    vector = xrv[image_id]
    return {
        finding: float(
            head["model"].predict_proba(((vector - head["mean"]) / head["scale"])[None])[0, 1]
        )
        for finding, head in heads.items()
    }


def generate_with_processor(
    adapter: Any,
    image: Image.Image,
    prompt: str,
    processor: SpecialistGuidance | None,
    max_new_tokens: int,
) -> Generation:
    common = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_return_sequences": 1,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": adapter.tokenizer.eos_token_id,
    }
    if processor is not None:
        common["logits_processor"] = LogitsProcessorList([processor])
    if isinstance(adapter, HuatuoOEAdapter):
        input_ids, images = adapter._inputs(image, prompt)
        if processor is not None:
            processor.prompt_length = int(input_ids.shape[1])
        output = adapter.model.generate(
            input_ids,
            images=images,
            min_new_tokens=1,
            repetition_penalty=1.2,
            eos_token_id=adapter.tokenizer.eos_token_id,
            **common,
        )
    elif isinstance(adapter, HuluOEAdapter):
        inputs = adapter._inputs(image, prompt)
        if processor is not None:
            processor.prompt_length = int(inputs["input_ids"].shape[1])
        output = adapter.model.generate(**inputs, **common)
    else:
        raise TypeError(type(adapter).__name__)
    return _decode_generations(adapter.tokenizer, output, adapter.model)[0]


def reader_vectors(path: Path) -> dict[str, dict[str, int]]:
    import csv

    name_map = {
        "aortic_enlargement": "Aortic enlargement",
        "cardiomegaly": "Cardiomegaly",
        "lung_opacity": "Lung Opacity",
        "nodule_mass": "Nodule/Mass",
        "pleural_effusion": "Pleural effusion",
        "pleural_thickening": "Pleural thickening",
        "pulmonary_fibrosis": "Pulmonary fibrosis",
    }
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    readers: dict[str, set[str]] = defaultdict(set)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            readers[image_id].add(row["rad_id"])
            for finding, column in name_map.items():
                votes[image_id][finding] += int(row[column])
    return {
        image_id: {finding: int(value) for finding, value in values.items()}
        for image_id, values in votes.items()
        if len(readers[image_id]) == 3
    }


def choose_panel(
    confirmation: list[dict[str, Any]], xrv: dict[str, np.ndarray], readers: dict[str, dict[str, int]], limit: int
) -> list[str]:
    ids = sorted({row["image_id"] for row in confirmation if row["image_id"] in xrv})
    unanimous = [i for i in ids if i in readers and all(readers[i].get(f, -1) in (0, 3) for f in FINDINGS)]
    positive = [i for i in unanimous if any(readers[i][f] == 3 for f in FINDINGS)]
    negative = [i for i in unanimous if all(readers[i][f] == 0 for f in FINDINGS)]
    half = limit // 2
    selected = positive[:half] + negative[: limit - half]
    if len(selected) < limit:
        selected.extend(i for i in unanimous if i not in selected)
        selected = selected[:limit]
    if len(selected) < limit:
        raise RuntimeError(f"Only {len(selected)} all-unanimous panel images are available")
    return selected


def summarize(rows: list[dict[str, Any]], readers: dict[str, dict[str, int]]) -> dict:
    output = {}
    for condition in ("native", "ccd_token", "claim_potential"):
        counts = defaultdict(int)
        for row in rows:
            claims = surface_claims(row[condition]["text"])
            truth = readers[row["image_id"]]
            for finding, state in claims.items():
                present = truth[finding] == 3
                if state == "positive":
                    counts["tp" if present else "fp"] += 1
                else:
                    counts["fn_statement" if present else "tn_statement"] += 1
            for finding in FINDINGS:
                if truth[finding] == 3 and finding not in claims:
                    counts["omitted_positive"] += 1
            counts["tokens"] += row[condition]["token_count"]
        tp, fp = counts["tp"], counts["fp"]
        output[condition] = {
            **dict(counts),
            "positive_claim_precision": tp / (tp + fp) if tp + fp else None,
            "mean_tokens": counts["tokens"] / len(rows) if rows else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu"), required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--xrv-logits", type=Path, required=True)
    parser.add_argument("--reader-labels", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A VLM GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xrv = load_logits(args.xrv_logits)
    development = load_claims(args.development, "development", args.model)
    confirmation = load_claims(args.confirmation, "confirmation", args.model)
    readers = reader_vectors(args.reader_labels)
    heads = fit_specialist_heads(development, xrv)
    panel = choose_panel(confirmation, xrv, readers, args.limit)
    adapter = HuatuoOEAdapter() if args.model == "huatuo" else HuluOEAdapter()
    prompt = "You are a professional radiologist. Generate a concise medical report for the image."
    rows = []
    try:
        for index, image_id in enumerate(panel):
            shard = args.output_dir / f"{index:03d}_{image_id}.json"
            if shard.exists():
                rows.append(json.loads(shard.read_text()))
                continue
            image = dicom_to_pil(args.image_root / f"{image_id}.dicom")
            probabilities = specialist_probabilities(image_id, heads, xrv)
            record: dict[str, Any] = {
                "image_id": image_id,
                "reader_votes": readers[image_id],
                "specialist_probabilities": probabilities,
            }
            for condition in ("native", "ccd_token", "claim_potential"):
                adapter._seed(args.seed)
                processor = None
                if condition != "native":
                    processor = SpecialistGuidance(
                        adapter.tokenizer,
                        probabilities,
                        prompt_length=0,
                        mode=condition,
                        beta=args.beta,
                        gamma=args.gamma,
                    )
                generation = generate_with_processor(
                    adapter, image, prompt, processor, args.max_new_tokens
                )
                record[condition] = {
                    "text": generation.text,
                    "token_ids": list(generation.token_ids),
                    "token_count": generation.token_count,
                    "surface_claims": surface_claims(generation.text),
                    "processor_steps": 0 if processor is None else processor.steps,
                    "adjusted_candidates": 0 if processor is None else processor.adjusted_candidates,
                }
            shard.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            rows.append(record)
    finally:
        adapter.close()

    result = {
        "protocol": PROTOCOL,
        "status": "complete_canary",
        "model": args.model,
        "n": len(rows),
        "config": {
            "development": str(args.development),
            "confirmation": str(args.confirmation),
            "xrv_logits": str(args.xrv_logits),
            "reader_labels": str(args.reader_labels),
            "image_root": str(args.image_root),
            "max_new_tokens": args.max_new_tokens,
            "beta": args.beta,
            "gamma": args.gamma,
            "seed": args.seed,
            "expert_head": "per-finding L2 logistic C=1 over complete 18D XRV state",
            "ccd_token_scope": "conservative first disease-phrase token subset of official only_first_token ECD",
        },
        "summary": summarize(rows, readers),
        "rows": rows,
        "claim_boundary": (
            "Reader-unanimous seven-finding surface audit. Regex claims are not a substitute for clinician review; "
            "this canary can reject the mechanism but cannot establish free-report hallucination efficacy."
        ),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
