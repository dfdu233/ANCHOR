#!/usr/bin/env python3
"""Audit prompt-conditioned template attraction in the VinDr Huatuo pilot.

This is deliberately a surface-form and embedded-claim *diagnostic*, not a
clinical evaluator.  It accepts a run only after reconstructing all 200 x 3
generation pairs and verifying every frozen hash.  The few claim families in
``EMBEDDED_CLAIM_FAMILIES`` are declared in source before inspection and are
reported separately from the template statistics.

The script does not load a model, import torch, reserve a GPU, alter the common
evaluation framework, or authorize a confirmation run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "clinical-template-attractor-diagnostic-v1"
EXPECTED_GENERATION_VERSION = "clinical-presupposition-generation-only-v1"
EXPECTED_CONDITIONS = ("neutral", "existential", "negative_obligation")
EXPECTED_PANEL = ("R8", "R9", "R10")
PREFIX_K = (3, 5, 8, 12)
WORD_BUDGET = 30
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42081

# These are intentionally narrow surface mappings.  They do not cover every
# way a clinician can express a claim and must never be scored as report truth.
# Bare ``effusion`` is admitted because it conventionally denotes pleural
# effusion in a chest-radiograph finding, but that convention is an explicit
# limitation of the diagnostic.
EMBEDDED_CLAIM_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "name": "positive_pleural_effusion",
        "finding": "pleural_effusion",
        "mention": "effusion",
        "polarity": "positive",
        "reference_score": "positive_votes_over_3",
    },
    {
        "name": "negative_pleural_effusion",
        "finding": "pleural_effusion",
        "mention": "effusion",
        "polarity": "negative",
        "reference_score": "negative_votes_over_3",
    },
    {
        "name": "positive_lung_opacity",
        "finding": "lung_opacity",
        "mention": "opacity",
        "polarity": "positive",
        "reference_score": "positive_votes_over_3",
    },
    {
        "name": "uncertain_lung_opacity",
        "finding": "lung_opacity",
        "mention": "opacity",
        "polarity": "uncertain",
        "reference_score": "reader_disagreement",
    },
)

_LEXICAL_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.I)
_WORD = re.compile(r"\b[a-z]+(?:[-'][a-z]+)*\b", re.I)
_NEGATORS = {"no", "without", "absent", "absence", "negative", "free", "neither"}
_UNCERTAIN = {
    "possible", "possibly", "probable", "probably", "may", "might", "could",
    "suggest", "suggests", "suggestive", "suspected", "questionable", "uncertain",
    "indeterminate", "subtle", "cannot", "can't", "exclude", "rule",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _LEXICAL_TOKEN.findall(normalized)


def normalized_template(text: str) -> str:
    """Apply only frozen orthographic normalization, not semantic masking."""

    tokens = ["<num>" if token[0].isdigit() else token for token in lexical_tokens(text)]
    return " ".join(tokens)


def prefix_signature(text: str, k: int) -> str:
    return " ".join(lexical_tokens(text)[:k])


def word_count(text: str) -> int:
    return len(_WORD.findall(unicodedata.normalize("NFKC", text)))


def _mention_polarities(text: str, mention: str) -> set[str]:
    tokens = lexical_tokens(text)
    targets = {
        "opacity": {"opacity", "opacities"},
        "effusion": {"effusion", "effusions"},
    }.get(mention, {mention, mention + "s"})
    output: set[str] = set()
    for index, token in enumerate(tokens):
        if token not in targets:
            continue
        left = tokens[max(0, index - 8):index]
        right = tokens[index + 1:index + 4]
        window = left + right
        joined = " ".join(left)
        # "cannot exclude" and "rule out" are uncertainty, not negation.
        uncertainty = bool(set(window) & _UNCERTAIN) or "cannot exclude" in joined or "rule out" in " ".join(window)
        negation = bool(set(left) & _NEGATORS)
        if uncertainty:
            output.add("uncertain")
        elif negation:
            output.add("negative")
        else:
            output.add("positive")
    return output


def embedded_claim_memberships(text: str) -> dict[str, bool]:
    by_mention = {
        mention: _mention_polarities(text, mention)
        for mention in {row["mention"] for row in EMBEDDED_CLAIM_FAMILIES}
    }
    return {
        row["name"]: row["polarity"] in by_mention[row["mention"]]
        for row in EMBEDDED_CLAIM_FAMILIES
    }


def _record_key(item_id: str, condition: str) -> str:
    return hashlib.sha256(f"{item_id}\0{condition}".encode()).hexdigest()


def validate_generation_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fail closed unless the formal 600-output run is complete and hash-bound."""

    required = (
        "generation_config.json", "generation_conformance.json",
        "selected_manifest.jsonl", "generations.jsonl", "generation_summary.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"formal generation incomplete; missing {missing}")
    config = json.loads((run_dir / "generation_config.json").read_text(encoding="utf-8"))
    if config.get("version") != EXPECTED_GENERATION_VERSION:
        raise ValueError("unexpected generation version")
    if config.get("limit") != 200 or config.get("split") != "pilot":
        raise ValueError("diagnostic is frozen to the 200-image pilot")
    if config.get("formal_clinical_claim_evaluation") is not False:
        raise ValueError("generation run improperly claims a formal clinical evaluation")
    immutable = {key: value for key, value in config.items() if key not in {"created_at", "command", "fingerprint"}}
    if canonical_sha(immutable) != config.get("fingerprint"):
        raise ValueError("generation config fingerprint mismatch")
    for path_key, hash_key in (
        ("labels_csv", "labels_csv_sha256"),
        ("ontology", "ontology_sha256"),
        ("renderer_source", "renderer_source_sha256"),
    ):
        path = Path(str(config[path_key]))
        if not path.is_file() or sha256_file(path) != config.get(hash_key):
            raise ValueError(f"source hash mismatch: {path_key}")
    runner_path = Path(str(config["command"][0]))
    if not runner_path.is_file() or sha256_file(runner_path) != config.get("runner_sha256"):
        raise ValueError("generation runner hash mismatch")
    manifest_path = run_dir / "selected_manifest.jsonl"
    if sha256_file(manifest_path) != config.get("selected_manifest_sha256"):
        raise ValueError("selected manifest hash mismatch")
    manifest = load_jsonl(manifest_path)
    if len(manifest) != 200 or len({str(row.get("item_id")) for row in manifest}) != 200:
        raise ValueError("selected manifest must contain 200 unique images")
    if any(row.get("selection_uses_reader_labels") is not False for row in manifest):
        raise ValueError("pilot selection was not label blind")
    manifest_by_id = {str(row["item_id"]): row for row in manifest}

    conditions = tuple(row.get("name") for row in config.get("prompt_conditions", []))
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError(f"unexpected prompt conditions: {conditions}")
    conformance = json.loads((run_dir / "generation_conformance.json").read_text(encoding="utf-8"))
    if (
        conformance.get("passed") is not True
        or conformance.get("fingerprint") != config["fingerprint"]
        or not conformance.get("direct_text")
        or conformance.get("direct_text") != conformance.get("standard_inference_text")
        or len(conformance.get("direct_generated_token_ids", []))
        != int(conformance.get("direct_generated_token_count", -1))
    ):
        raise ValueError("direct/standard generation conformance failed")

    shard_paths = sorted((run_dir / "shards").glob("*.json"))
    if len(shard_paths) != 600:
        raise ValueError(f"expected exactly 600 shards, found {len(shard_paths)}")
    if list((run_dir / "errors").glob("*.json")):
        raise ValueError("generation error shards remain")
    shard_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in shard_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        item_id, condition = str(row.get("item_id")), str(row.get("prompt_condition"))
        key = (item_id, condition)
        if key in shard_rows:
            raise ValueError(f"duplicate generation pair: {key}")
        if item_id not in manifest_by_id or condition not in EXPECTED_CONDITIONS:
            raise ValueError(f"unexpected generation pair: {key}")
        expected_name = _record_key(item_id, condition) + ".json"
        if path.name != expected_name:
            raise ValueError(f"misnamed generation shard: {path.name}")
        item = manifest_by_id[item_id]
        if (
            row.get("version") != EXPECTED_GENERATION_VERSION
            or row.get("fingerprint") != config["fingerprint"]
            or row.get("image_id") != item.get("image_id")
            or row.get("claim_universe_sha256") != item.get("claim_universe_sha256")
            or row.get("clinical_claim_evaluation_status") != "pending_shared_audit"
            or row.get("automatic_labeler_used") is not False
            or row.get("ground_truth_used_for_generation_or_selection") is not False
            or not str(row.get("text", "")).strip()
        ):
            raise ValueError(f"invalid generation shard: {key}")
        ids = row.get("generated_token_ids")
        if not isinstance(ids, list) or len(ids) != int(row.get("generated_token_count", -1)):
            raise ValueError(f"invalid token accounting: {key}")
        shard_rows[key] = row
    expected_pairs = {(item_id, condition) for item_id in manifest_by_id for condition in EXPECTED_CONDITIONS}
    if set(shard_rows) != expected_pairs:
        raise ValueError("generation pairs do not form the complete 200 x 3 product")

    generations_path = run_dir / "generations.jsonl"
    generations = load_jsonl(generations_path)
    if len(generations) != 600:
        raise ValueError("generations.jsonl must contain 600 rows")
    generation_by_key = {(str(row.get("item_id")), str(row.get("prompt_condition"))): row for row in generations}
    if len(generation_by_key) != 600 or generation_by_key != shard_rows:
        raise ValueError("generations.jsonl is not an exact reconstruction of the shards")
    summary = json.loads((run_dir / "generation_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "generation_complete_clinical_audit_pending"
        or summary.get("items") != 200
        or summary.get("generations") != 600
        or summary.get("fingerprint") != config["fingerprint"]
        or summary.get("generations_sha256") != sha256_file(generations_path)
        or summary.get("clinical_claim_evaluation") != "pending_shared_audit"
    ):
        raise ValueError("generation summary is incomplete or hash-inconsistent")
    ordered = sorted(shard_rows.values(), key=lambda row: (str(row["item_id"]), str(row["prompt_condition"])))
    return config, manifest, ordered, conformance


def load_selected_votes(labels_csv: Path, image_ids: set[str]) -> dict[str, dict[str, int]]:
    source_names = {"pleural_effusion": "Pleural effusion", "lung_opacity": "Lung Opacity"}
    rows: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    with labels_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "rad_id", *source_names.values()}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("VinDr label CSV lacks required embedded-claim columns")
        for line_number, row in enumerate(reader, 2):
            image_id = str(row["image_id"]).strip()
            if image_id not in image_ids:
                continue
            rad_id = str(row["rad_id"]).strip()
            if rad_id in rows[image_id]["_readers"]:
                raise ValueError(f"duplicate reader for selected image on line {line_number}")
            rows[image_id]["_readers"][rad_id] = 1
            for normalized, source in source_names.items():
                raw = str(row[source]).strip()
                if raw not in {"0", "1", "0.0", "1.0"}:
                    raise ValueError(f"non-binary VinDr vote on line {line_number}")
                rows[image_id][normalized][rad_id] = int(float(raw))
    output: dict[str, dict[str, int]] = {}
    for image_id in sorted(image_ids):
        if image_id not in rows:
            raise ValueError(f"selected image absent from label CSV: {image_id}")
        readers = set(rows[image_id]["_readers"])
        if readers != set(EXPECTED_PANEL):
            raise ValueError(f"selected image does not use exact R8/R9/R10 panel: {image_id}")
        output[image_id] = {}
        for finding in source_names:
            if set(rows[image_id][finding]) != set(EXPECTED_PANEL):
                raise ValueError(f"incomplete reader votes: {image_id}/{finding}")
            output[image_id][finding] = sum(rows[image_id][finding].values())
    return output


def _entropy(counts: Iterable[int]) -> float:
    values = [value for value in counts if value > 0]
    total = sum(values)
    return -sum((value / total) * math.log2(value / total) for value in values) if total else 0.0


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score, label in zip(scores, labels):
        grouped[float(score)][int(label)] += 1
    positives = sum(values[1] for values in grouped.values())
    negatives = sum(values[0] for values in grouped.values())
    if not positives or not negatives:
        return None
    wins = 0.0
    negative_below = 0
    for score in sorted(grouped):
        negative_here, positive_here = grouped[score]
        wins += positive_here * negative_below + 0.5 * positive_here * negative_here
        negative_below += negative_here
    return wins / (positives * negatives)


def slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    x_bar, y_bar = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - x_bar) ** 2 for x in xs)
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denominator


def bootstrap_metric(rows: Sequence[Mapping[str, Any]], function, seed: int) -> dict[str, Any]:
    import random

    point = function(rows)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = function(sample)
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    return {
        "point": point,
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "valid_bootstrap_replicates": len(estimates),
        "bootstrap_unit": "image",
    }


def _reference_score(family: Mapping[str, str], votes: int) -> float:
    mode = family["reference_score"]
    if mode == "positive_votes_over_3":
        return votes / 3.0
    if mode == "negative_votes_over_3":
        return (3 - votes) / 3.0
    if mode == "reader_disagreement":
        return 1.0 if votes in {1, 2} else 0.0
    raise ValueError(f"unknown reference score: {mode}")


def claim_family_stats(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family_index, family in enumerate(EMBEDDED_CLAIM_FAMILIES):
        name, finding = family["name"], family["finding"]
        family_out: dict[str, Any] = {
            "mapping": dict(family),
            "scope_warning": "predeclared literal embedded-claim family; not complete clinical truth",
            "conditions": {},
            "paired_prompt_effects": {},
        }
        by_condition: dict[str, list[Mapping[str, Any]]] = {}
        for condition_index, condition in enumerate(EXPECTED_CONDITIONS):
            rows = [row for row in diagnostics if row["prompt_condition"] == condition]
            by_condition[condition] = rows
            labels = [int(row["embedded_claim_memberships"][name]) for row in rows]
            vote_values = [int(row["reader_votes"][finding]) for row in rows]
            rates = {
                f"{vote}/3": {
                    "n": sum(value == vote for value in vote_values),
                    "members": sum(label for label, value in zip(labels, vote_values) if value == vote),
                    "rate": (
                        sum(label for label, value in zip(labels, vote_values) if value == vote)
                        / sum(value == vote for value in vote_values)
                        if sum(value == vote for value in vote_values) else None
                    ),
                }
                for vote in range(4)
            }
            auc_result = bootstrap_metric(
                rows,
                lambda sample, fam=family, key=name, fd=finding: auc(
                    [_reference_score(fam, int(row["reader_votes"][fd])) for row in sample],
                    [int(row["embedded_claim_memberships"][key]) for row in sample],
                ),
                BOOTSTRAP_SEED + family_index * 100 + condition_index * 10,
            )
            support_slope = bootstrap_metric(
                rows,
                lambda sample, fam=family, key=name, fd=finding: slope(
                    [_reference_score(fam, int(row["reader_votes"][fd])) for row in sample],
                    [int(row["embedded_claim_memberships"][key]) for row in sample],
                ),
                BOOTSTRAP_SEED + family_index * 100 + condition_index * 10 + 1,
            )
            family_out["conditions"][condition] = {
                "n": len(rows),
                "members": sum(labels),
                "membership_rate": statistics.fmean(labels),
                "vote_bin_rates": rates,
                "template_membership_auc_from_expected_reference_support": auc_result,
                "expected_support_monotonic_slope": support_slope,
                "monotonicity_note": (
                    "reader-disagreement support is intentionally high at 1/3 and 2/3; raw-vote monotonicity is not expected"
                    if family["reference_score"] == "reader_disagreement"
                    else "positive slope means lexical membership rises with the family's expected reader support"
                ),
            }
        for left_index, left in enumerate(EXPECTED_CONDITIONS):
            for right_index in range(left_index + 1, len(EXPECTED_CONDITIONS)):
                right = EXPECTED_CONDITIONS[right_index]
                paired = []
                right_by_image = {row["image_id"]: row for row in by_condition[right]}
                for row in by_condition[left]:
                    other = right_by_image[row["image_id"]]
                    paired.append({
                        "image_id": row["image_id"],
                        "difference": int(row["embedded_claim_memberships"][name]) - int(other["embedded_claim_memberships"][name]),
                    })
                family_out["paired_prompt_effects"][f"{left}_minus_{right}"] = bootstrap_metric(
                    paired,
                    lambda sample: statistics.fmean(float(row["difference"]) for row in sample),
                    BOOTSTRAP_SEED + family_index * 1000 + left_index * 10 + right_index,
                )
        output[name] = family_out
    return output


def concentration_report(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    distinct_images: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        distinct_images[str(row[field])].add(str(row["image_id"]))
    total = len(rows)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "unique": len(counts),
        "entropy_bits": _entropy(counts.values()),
        "effective_number": 2 ** _entropy(counts.values()),
        "top1_concentration": sum(value for _, value in ordered[:1]) / total,
        "top3_concentration": sum(value for _, value in ordered[:3]) / total,
        "top5_concentration": sum(value for _, value in ordered[:5]) / total,
        "outputs_in_cross_image_repeats": sum(value for key, value in counts.items() if len(distinct_images[key]) >= 2),
        "cross_image_repeat_rate": sum(value for key, value in counts.items() if len(distinct_images[key]) >= 2) / total,
        "maximum_distinct_images_for_one_form": max((len(value) for value in distinct_images.values()), default=0),
        "top_forms": [
            {"form": key, "count": value, "distinct_images": len(distinct_images[key])}
            for key, value in ordered[:20]
        ],
    }


def freeze_pilot_discoveries(
    diagnostics: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    analysis_fingerprint: str,
) -> dict[str, Any]:
    """Freeze reproducible diagnostic families without authorizing a new run."""

    families = []
    for condition in EXPECTED_CONDITIONS:
        rows = [row for row in diagnostics if row["prompt_condition"] == condition]
        for kind, key, k in [("normalized_template", "normalized_template", None)] + [
            ("prefix", f"prefix_{value}", value) for value in PREFIX_K
        ]:
            counts = Counter(str(row[key]) for row in rows)
            images: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                images[str(row[key])].add(str(row["image_id"]))
            for signature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                distinct = len(images[signature])
                share = count / len(rows)
                if distinct < 10 or share < 0.05:
                    continue
                families.append({
                    "condition": condition,
                    "kind": kind,
                    "k": k,
                    "signature": signature,
                    "pilot_count": count,
                    "pilot_distinct_images": distinct,
                    "pilot_share": share,
                    "membership_rule": "exact equality after frozen normalization" if kind == "normalized_template" else "exact equality of first k frozen lexical tokens",
                })
    payload = {
        "version": "pilot-template-family-freeze-v1",
        "created_at": utc_now(),
        "source_generation_fingerprint": source["generation_fingerprint"],
        "source_generations_sha256": source["generations_sha256"],
        "analysis_fingerprint": analysis_fingerprint,
        "discovery_split": "pilot",
        "discovery_is_exploratory": True,
        "normalization": "Unicode NFKC + casefold + frozen lexical tokenization + numeric token masking only",
        "prefix_k": list(PREFIX_K),
        "admission_threshold": {"minimum_distinct_images": 10, "minimum_within_condition_share": 0.05},
        "families": families,
        "embedded_claim_families": [dict(row) for row in EMBEDDED_CLAIM_FAMILIES],
        "confirmation_authorized": False,
        "confirmation_manifest_emitted": False,
        "reason": "Template Collapse and Pensieve create a direct collision; surface attraction alone cannot establish a new mechanism or method.",
        "allowed_next_claim": "diagnostic surface regularity only",
    }
    payload["spec_sha256"] = canonical_sha(payload)
    return payload


def causal_gate_protocol(spec_sha256: str, analysis_fingerprint: str) -> dict[str, Any]:
    return {
        "version": "clinical-autoregressive-lock-in-causal-gate-v1",
        "frozen_template_spec_sha256": spec_sha256,
        "analysis_fingerprint": analysis_fingerprint,
        "status": "not_run",
        "confirmation_generation_authorized": False,
        "hypothesis": "A prompt-induced prefix can lock generation onto an image-insensitive clinical continuation despite decodable pre-generation reader polarity.",
        "required_sequence": [
            {
                "gate": "pre_generation_reader_polarity",
                "test": "Decode VinDr reader polarity from image-conditioned state before the first response token; compare real image with same-support shuffled-image and text-only controls.",
                "pass": "image-cluster bootstrap AUROC lower 95% bound > 0.5 and real-minus-shuffled AUROC lower bound > 0",
            },
            {
                "gate": "prefix_indexed_image_causal_collapse",
                "test": "Teacher-force the identical frozen prefix and continuation under correct versus deterministic view/vote-matched swapped images; estimate swapped-minus-correct continuation NLL at successive prefix lengths and decoder layers.",
                "pass": "pre-registered prefix-length trend toward zero has upper 95% bound < 0, the crossing token/layer is reproducible, and a length/condition/vote-matched non-attractor control does not show the same collapse",
            },
            {
                "gate": "selective_activation_patch",
                "test": "At the frozen crossing token/layer, patch only the correct-image component into the swapped-image trajectory and compare clinical next-token polarity margins.",
                "pass": "patch restores the correct-image margin with lower 95% bound > 0 while random-direction, norm-matched, text-only, pre-prefix, and matched non-attractor controls remain null; supported clear claims change <=1pp",
            },
            {
                "gate": "audited_clinical_link",
                "test": "Use the separately audited claim evaluator or physician labels; lexical membership is prohibited as truth.",
                "pass": "lock-in membership predicts audited error at matched length/claim count and the selective patch reduces error without omission or blanket hedging",
            },
        ],
        "stop_rule": "Failure of any gate stops this direction; do not tune a template-subtraction mitigation on the pilot.",
        "alternative_explanations": [
            "The three prompts define different pragmatic tasks and answer spaces, so condition differences may be legitimate task compliance.",
            "Longer answers mechanically increase phrase and claim-family opportunities.",
            "VinDr prevalence and the exact R8/R9/R10 panel can induce apparent lexical calibration.",
            "Greedy decoding can amplify a shallow initial-token preference without a layerwise causal collapse.",
            "A generic radiology-report prior can repeat across images while still using image evidence for claim-bearing tokens.",
            "The narrow embedded-claim matcher misses synonyms and can mishandle long-range negation or mixed-polarity clauses.",
            "View position, acquisition quality, and renderer artifacts can alter both reader votes and response form.",
            "Template Collapse and Pensieve already cover broad template-insensitivity and cross-image subtraction claims.",
        ],
    }


def analyze(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix with existing outputs: {output_dir}")
    config, manifest, generations, conformance = validate_generation_run(run_dir)
    votes = load_selected_votes(Path(str(config["labels_csv"])), {str(row["image_id"]) for row in manifest})
    analyzer_path = Path(__file__).resolve()
    source = {
        "run_dir": str(run_dir.resolve()),
        "generation_fingerprint": config["fingerprint"],
        "generations_sha256": sha256_file(run_dir / "generations.jsonl"),
        "labels_csv_sha256": config["labels_csv_sha256"],
        "selected_manifest_sha256": config["selected_manifest_sha256"],
        "conformance_passed": conformance["passed"],
    }
    analysis_contract = {
        "analyzer_source": str(analyzer_path),
        "analyzer_source_sha256": sha256_file(analyzer_path),
        "exact_command": list(sys.argv),
        "bootstrap_unit": "image",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prefix_k": list(PREFIX_K),
        "word_budget": WORD_BUDGET,
        "normalization": "Unicode NFKC + casefold + frozen lexical tokenization + numeric token masking only",
        "embedded_claim_families": [dict(row) for row in EMBEDDED_CLAIM_FAMILIES],
        "output_dir": str(output_dir.resolve()),
    }
    analysis_fingerprint = canonical_sha({"source": source, "analysis_contract": analysis_contract})
    diagnostics = []
    for row in generations:
        text = str(row["text"])
        item_id = str(row["item_id"])
        diagnostic = {
            "version": VERSION,
            "analysis_fingerprint": analysis_fingerprint,
            "image_id": item_id,
            "prompt_condition": str(row["prompt_condition"]),
            "text": text,
            "exact_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "normalized_template": normalized_template(text),
            "word_count": word_count(text),
            "word_budget": WORD_BUDGET,
            "word_budget_violation": word_count(text) > WORD_BUDGET,
            "generated_token_count": int(row["generated_token_count"]),
            "visible_answer_token_count": int(row["visible_answer_token_count"]),
            "max_new_tokens": int(row["max_new_tokens"]),
            "generation_cap_hit": bool(row["hit_max_new_tokens"]),
            # Huatuo's generation-only sequence may include a leading/trailing
            # template token.  The visible retokenized answer is the comparable
            # new-token budget; keep raw-sequence excess as a separate audit.
            "visible_token_budget_violation": int(row["visible_answer_token_count"]) > int(row["max_new_tokens"]),
            "raw_sequence_count_exceeds_requested_cap": int(row["generated_token_count"]) > int(row["max_new_tokens"]),
            "embedded_claim_memberships": embedded_claim_memberships(text),
            "reader_votes": votes[item_id],
            "clinical_truth_status": "not_assigned_by_this_diagnostic",
        }
        for k in PREFIX_K:
            diagnostic[f"prefix_{k}"] = prefix_signature(text, k)
        diagnostics.append(diagnostic)

    condition_reports: dict[str, Any] = {}
    for condition in EXPECTED_CONDITIONS:
        rows = [row for row in diagnostics if row["prompt_condition"] == condition]
        words = [int(row["word_count"]) for row in rows]
        generated = [int(row["generated_token_count"]) for row in rows]
        visible = [int(row["visible_answer_token_count"]) for row in rows]
        condition_reports[condition] = {
            "n": len(rows),
            "exact_text": concentration_report(rows, "text"),
            "normalized_template": concentration_report(rows, "normalized_template"),
            "prefix": {str(k): concentration_report(rows, f"prefix_{k}") for k in PREFIX_K},
            "length": {
                "word_count_mean": statistics.fmean(words),
                "word_count_median": statistics.median(words),
                "word_count_min": min(words),
                "word_count_max": max(words),
                "word_budget_violations": sum(bool(row["word_budget_violation"]) for row in rows),
                "word_budget_violation_rate": statistics.fmean(bool(row["word_budget_violation"]) for row in rows),
                "generated_token_count_mean": statistics.fmean(generated),
                "visible_answer_token_count_mean": statistics.fmean(visible),
                "generation_cap_hits": sum(bool(row["generation_cap_hit"]) for row in rows),
                "visible_token_budget_violations": sum(bool(row["visible_token_budget_violation"]) for row in rows),
                "raw_sequence_count_exceeds_requested_cap": sum(bool(row["raw_sequence_count_exceeds_requested_cap"]) for row in rows),
                "raw_sequence_note": "Huatuo generation-only sequences can include one template boundary token; visible-answer tokens define the comparable cap",
            },
        }
    cross_condition_identical = 0
    by_image: dict[str, dict[str, str]] = defaultdict(dict)
    for row in diagnostics:
        by_image[str(row["image_id"])][str(row["prompt_condition"])] = str(row["text"])
    for values in by_image.values():
        if len(set(values.values())) < len(EXPECTED_CONDITIONS):
            cross_condition_identical += 1
    frozen = freeze_pilot_discoveries(diagnostics, source, analysis_fingerprint)
    gate = causal_gate_protocol(frozen["spec_sha256"], analysis_fingerprint)
    summary = {
        "version": VERSION,
        "created_at": utc_now(),
        "status": "complete_exploratory_diagnostic_only",
        "source": source,
        "analysis_contract": analysis_contract,
        "analysis_fingerprint": analysis_fingerprint,
        "integrity": {
            "images": len(manifest),
            "conditions": len(EXPECTED_CONDITIONS),
            "generations": len(generations),
            "complete_cartesian_product": True,
            "shards_config_conformance_hashes_verified": True,
        },
        "scope": {
            "clinical_evaluator": False,
            "automatic_labeler": False,
            "embedded_claim_scope": "four predeclared literal families only",
            "interpretation": "exploratory prompt-conditioned surface behavior; no hallucination or mechanism claim",
            "confirmation_authorized": False,
        },
        "conditions": condition_reports,
        "images_with_any_exact_cross_condition_duplicate": cross_condition_identical,
        "embedded_claim_families": claim_family_stats(diagnostics),
        "frozen_template_spec_sha256": frozen["spec_sha256"],
        "causal_gate_status": "not_run",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in diagnostics)
    atomic_text(output_dir / "text_diagnostics.jsonl", diagnostics_text)
    atomic_json(output_dir / "frozen_pilot_template_spec.json", frozen)
    atomic_json(output_dir / "causal_lock_in_gate.json", gate)
    atomic_json(output_dir / "summary.json", summary)
    complete = {
        "version": VERSION,
        "status": "complete",
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "diagnostics_sha256": sha256_file(output_dir / "text_diagnostics.jsonl"),
        "template_spec_sha256": sha256_file(output_dir / "frozen_pilot_template_spec.json"),
        "causal_gate_sha256": sha256_file(output_dir / "causal_lock_in_gate.json"),
        "source_generation_fingerprint": config["fingerprint"],
        "analysis_fingerprint": analysis_fingerprint,
        "analyzer_source_sha256": analysis_contract["analyzer_source_sha256"],
    }
    atomic_json(output_dir / "COMPLETE.json", complete)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.run_dir, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
