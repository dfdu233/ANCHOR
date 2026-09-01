#!/usr/bin/env python3
"""Cross-model fatal gate for frame-covariant laterality compilation.

The method is deliberately minimal: generate in an explicitly typed screen
frame, then compile screen coordinates into patient coordinates using the
known radiological display involution.  The compiler cannot insert, remove,
rerank, or rescore a finding.

Two arms are collected on the same image-disjoint VinDr cases:

* ``named``: the prompt supplies two reader-supported findings on opposite
  patient sides.  This isolates attribute binding from finding discovery.
* ``natural``: the prompt asks for abnormalities and locations without naming
  findings.  The screen answer is compiled token-deterministically; a separate
  native patient-frame generation is only the paired baseline.  Their complete
  ontology mention sets are audited so prompt-induced omission cannot be
  mistaken for a laterality gain.

This script is resumable and model-agnostic across the repository's audited OE
adapters.  It does not launch automatically and therefore does not disturb the
baseline GPU queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.models_oe import load_oe_adapter
from corrected_sgta.run_huatuo_binding_conservation_probe_v1 import (
    FOCAL_FINDINGS,
    append_jsonl,
    atomic_json,
    build_cases,
    normalize_finding,
    parse_binding,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil


VERSION = "frame-covariant-cross-model-v3"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_orientation_certificates(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text())
    if payload.get("mapping") != (
        "screen_left_is_patient_right; screen_right_is_patient_left"
    ):
        raise ValueError("orientation certificate has an unsupported mapping")
    output: dict[str, dict[str, str]] = {}
    valid = {
        ("R", "left"),
        ("L", "right"),
        ("L+R", "right+left"),
    }
    for row in payload.get("certificates", []):
        image_id = str(row["image_id"])
        marker = str(row["marker"])
        marker_side = str(row["marker_screen_side"])
        if (marker, marker_side) not in valid:
            raise ValueError(
                f"orientation marker conflicts with radiological display: {row}"
            )
        if image_id in output:
            raise ValueError(f"duplicate orientation certificate: {image_id}")
        output[image_id] = {
            "marker": marker,
            "marker_screen_side": marker_side,
        }
    if not output:
        raise ValueError("orientation certificate contains no admitted images")
    return output


def swap_laterality(text: str) -> str:
    """Legacy Z2 word swap retained only for the diagnostic native arm."""

    def replacement(match: re.Match[str]) -> str:
        word = match.group(0)
        target = "right" if word.lower() == "left" else "left"
        if word.isupper():
            return target.upper()
        if word[:1].isupper():
            return target.capitalize()
        return target

    return re.sub(r"\b(?:left|right)\b", replacement, text, flags=re.IGNORECASE)


SCREEN_COORDINATE_RE = re.compile(
    r"\bscreen\s*[- ]\s*(left|right)\b", re.IGNORECASE
)
PATIENT_COORDINATE_RE = re.compile(
    r"\b(?:the\s+)?patient(?:'s|s)?\s+(left|right)\b", re.IGNORECASE
)


def screen_coordinate_contract(text: str) -> bool:
    """Require every lateral word to be explicitly typed as screen coordinates."""

    if PATIENT_COORDINATE_RE.search(text):
        return False
    spans = [match.span(1) for match in SCREEN_COORDINATE_RE.finditer(text)]
    lateral = [match.span() for match in re.finditer(r"\b(?:left|right)\b", text, re.I)]
    return bool(spans) and all(any(begin <= a and b <= end for begin, end in spans) for a, b in lateral)


def compile_screen_to_patient(text: str) -> str:
    """Compile typed display coordinates into anatomical patient coordinates.

    The radiological convention puts patient-right on screen-left and
    patient-left on screen-right.  Untyped lateral words are rejected rather
    than silently reinterpreted; this prevents a patient-frame answer from
    being double-flipped.
    """

    if not screen_coordinate_contract(text):
        raise ValueError("answer violates explicit screen-coordinate contract")

    def replacement(match: re.Match[str]) -> str:
        side = match.group(1).lower()
        return "patient's right" if side == "left" else "patient's left"

    return SCREEN_COORDINATE_RE.sub(replacement, text)


def erase_frame_words(text: str) -> str:
    output = SCREEN_COORDINATE_RE.sub("<side>", text.lower())
    output = PATIENT_COORDINATE_RE.sub("<side>", output)
    output = re.sub(r"\b(?:left|right)\b", "<side>", output)
    return re.sub(r"\s+", " ", output).strip()


def mentioned_findings(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized)
    output: set[str] = set()
    for finding in FOCAL_FINDINGS:
        canonical = normalize_finding(finding)
        aliases = {canonical}
        if finding == "Nodule/Mass":
            aliases.update({"nodule", "mass"})
        if finding == "Lung Opacity":
            aliases.add("opacity")
        if finding == "Clavicle fracture":
            aliases.add("clavicular fracture")
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            output.add(finding)
    return output


def named_prompt(left_finding: str, right_finding: str) -> str:
    return (
        f"This chest X-ray contains {normalize_finding(left_finding)} and "
        f"{normalize_finding(right_finding)}. State which finding is on the patient's "
        "left and which is on the patient's right. Answer in one short sentence."
    )


def screen_prompt(left_finding: str, right_finding: str) -> str:
    return (
        f"This displayed chest X-ray contains {normalize_finding(left_finding)} and "
        f"{normalize_finding(right_finding)}. Ignore patient-anatomical left and right. "
        "Using only screen coordinates, state where each finding appears. Use exactly the "
        "labels screen-left and screen-right; never use patient-anatomical left or right. "
        "Answer in one short sentence."
    )


def natural_prompt() -> str:
    return (
        "Describe the abnormalities visible on this chest X-ray and state their anatomical "
        "locations. Answer in one concise sentence."
    )


def natural_screen_prompt() -> str:
    return (
        "Describe the abnormalities visible on this displayed chest X-ray and locate them "
        "using screen coordinates only. Use exactly the labels screen-left, screen-right, "
        "or both-sides for every location; never use patient-anatomical left or right. "
        "Answer in one concise sentence."
    )


def parse_finding_side(text: str, finding: str, radius: int = 60) -> str:
    """Return left/right/ambiguous/unparsed for one named finding."""

    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized)
    name = normalize_finding(finding)
    occurrences = [match.span() for match in re.finditer(re.escape(name), normalized)]
    if not occurrences:
        return "unparsed"
    candidates: list[tuple[int, str]] = []
    for begin, end in occurrences:
        lower = max(0, begin - radius)
        upper = min(len(normalized), end + radius)
        window = normalized[lower:upper]
        for match in re.finditer(r"\b(left|right)\b", window):
            side_begin = lower + match.start()
            side_end = lower + match.end()
            # Distance between non-overlapping spans.  Nearest attachment is
            # less likely to steal the side belonging to a neighboring claim.
            distance = min(abs(side_end - begin), abs(side_begin - end))
            candidates.append((distance, match.group(1)))
    if not candidates:
        return "unparsed"
    candidates.sort()
    best_distance = candidates[0][0]
    best_sides = {side for distance, side in candidates if distance == best_distance}
    return next(iter(best_sides)) if len(best_sides) == 1 else "ambiguous"


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_bootstrap(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "estimate": None, "ci95": [None, None]}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "n": len(values),
        "estimate": float(array.mean()),
        "ci95": percentile_interval(means),
    }


def analyze(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]

    named_rows = [
        row
        for row in ok
        if row.get("named_screen_contract", False)
        if row.get("named_direct_parse") != "unparsed"
        and row.get("named_screen_parse") != "unparsed"
        and row.get("named_compiled_parse") != "unparsed"
    ]
    named_delta = [
        float(row["named_compiled_parse"] == "correct")
        - float(row["named_direct_parse"] == "correct")
        for row in named_rows
    ]
    named_bootstrap = paired_bootstrap(named_delta, draws, seed)
    named_direct_accuracy = (
        float(np.mean([row["named_direct_parse"] == "correct" for row in named_rows]))
        if named_rows
        else None
    )
    named_screen_accuracy = (
        float(np.mean([row["named_screen_parse"] == "correct" for row in named_rows]))
        if named_rows
        else None
    )
    named_compiled_accuracy = (
        float(np.mean([row["named_compiled_parse"] == "correct" for row in named_rows]))
        if named_rows
        else None
    )
    named_content = (
        float(np.mean([row["named_content_preserved"] for row in named_rows]))
        if named_rows
        else None
    )

    natural_claims: list[dict[str, Any]] = []
    native_target_mentions = 0
    method_target_mentions = 0
    target_opportunities = 0
    exact_target_mention_sets: list[bool] = []
    exact_ontology_mention_sets: list[bool] = []
    natural_contract_rows = 0
    for row in ok:
        if not row.get("natural_screen_contract", False):
            continue
        natural_contract_rows += 1
        method_answer = row.get(
            "natural_screen_compiled_answer", row.get("natural_compiled_answer", "")
        )
        native_mentioned: set[str] = set()
        method_mentioned: set[str] = set()
        for truth_key, finding_key in (
            ("left", "left_finding"),
            ("right", "right_finding"),
        ):
            finding = row[finding_key]
            native_side = parse_finding_side(row["natural_answer"], finding)
            compiled_side = parse_finding_side(method_answer, finding)
            target_opportunities += 1
            if native_side in {"left", "right", "ambiguous"}:
                native_target_mentions += 1
                native_mentioned.add(finding)
            if compiled_side in {"left", "right", "ambiguous"}:
                method_target_mentions += 1
                method_mentioned.add(finding)
            if native_side not in {"left", "right"} or compiled_side not in {"left", "right"}:
                continue
            natural_claims.append(
                {
                    "case_key": row["case_key"],
                    "finding": finding,
                    "truth_side": truth_key,
                    "native_side": native_side,
                    "compiled_side": compiled_side,
                    "native_correct": native_side == truth_key,
                    "compiled_correct": compiled_side == truth_key,
                }
            )
        exact_target_mention_sets.append(native_mentioned == method_mentioned)
        exact_ontology_mention_sets.append(
            mentioned_findings(row["natural_answer"])
            == mentioned_findings(method_answer)
        )
    natural_delta = [
        float(item["compiled_correct"]) - float(item["native_correct"])
        for item in natural_claims
    ]
    natural_bootstrap = paired_bootstrap(natural_delta, draws, seed + 1)
    natural_native_errors = sum(not item["native_correct"] for item in natural_claims)
    natural_compiled_errors = sum(not item["compiled_correct"] for item in natural_claims)
    corrected = sum(
        (not item["native_correct"]) and item["compiled_correct"]
        for item in natural_claims
    )
    harmed = sum(
        item["native_correct"] and (not item["compiled_correct"])
        for item in natural_claims
    )
    relative_error_reduction = (
        (natural_native_errors - natural_compiled_errors) / natural_native_errors
        if natural_native_errors
        else None
    )
    clear_harm_rate = (
        harmed / sum(item["native_correct"] for item in natural_claims)
        if any(item["native_correct"] for item in natural_claims)
        else 0.0
    )
    natural_content = float(
        np.mean(
            [
                row.get(
                    "natural_screen_content_preserved",
                    row.get("natural_content_preserved", False),
                )
                for row in ok
            ]
        )
    ) if ok else None
    native_target_recall = (
        native_target_mentions / target_opportunities if target_opportunities else None
    )
    method_target_recall = (
        method_target_mentions / target_opportunities if target_opportunities else None
    )
    exact_target_mention_set_rate = (
        float(np.mean(exact_target_mention_sets)) if exact_target_mention_sets else None
    )
    exact_ontology_mention_set_rate = (
        float(np.mean(exact_ontology_mention_sets))
        if exact_ontology_mention_sets
        else None
    )

    named_ci_low = named_bootstrap["ci95"][0]
    named_gate = bool(
        len(named_rows) >= 16
        and named_bootstrap["estimate"] is not None
        and named_bootstrap["estimate"] >= 0.20
        and named_ci_low is not None
        and named_ci_low > 0
        and named_content == 1.0
    )
    natural_ci_low = natural_bootstrap["ci95"][0]
    natural_gate = bool(
        len(natural_claims) >= 20
        and relative_error_reduction is not None
        and relative_error_reduction >= 0.20
        and natural_bootstrap["estimate"] is not None
        and natural_bootstrap["estimate"] > 0
        and natural_ci_low is not None
        and natural_ci_low > 0
        and clear_harm_rate <= 0.01
        and natural_content == 1.0
        and native_target_recall is not None
        and method_target_recall is not None
        and method_target_recall >= native_target_recall - 0.01
        and exact_ontology_mention_set_rate == 1.0
    )
    return {
        "version": VERSION,
        "status": (
            "GO_CROSS_MODEL_AND_NATURAL"
            if named_gate and natural_gate
            else "GO_NAMED_ONLY"
            if named_gate
            else "NO_GO_FRAME_COMPILATION"
        ),
        "named_gate_passed": named_gate,
        "natural_gate_passed": natural_gate,
        "named": {
            "n_jointly_parseable": len(named_rows),
            "direct_patient_accuracy": named_direct_accuracy,
            "screen_coordinate_accuracy": named_screen_accuracy,
            "compiled_patient_accuracy": named_compiled_accuracy,
            "compiled_minus_direct": named_bootstrap,
            "content_preservation_rate": named_content,
        },
        "natural": {
            "n_parseable_lateralized_claims": len(natural_claims),
            "native_errors": natural_native_errors,
            "compiled_errors": natural_compiled_errors,
            "corrected_native_errors": corrected,
            "harmed_native_correct": harmed,
            "relative_error_reduction": relative_error_reduction,
            "clear_case_harm_rate": clear_harm_rate,
            "paired_accuracy_delta": natural_bootstrap,
            "content_preservation_rate": natural_content,
            "native_target_finding_recall": native_target_recall,
            "method_target_finding_recall": method_target_recall,
            "exact_target_mention_set_rate": exact_target_mention_set_rate,
            "exact_ontology_mention_set_rate": exact_ontology_mention_set_rate,
            "screen_contract_rows": natural_contract_rows,
            "claims": natural_claims,
        },
        "gate": {
            "named": ">=16 parseable; +20pp; paired CI low>0; content 100%",
            "natural": (
                ">=20 parseable claims; relative error -20%; paired CI low>0; "
                "native-correct harm <=1%; screen->patient compilation content 100%; "
                "target finding recall no more than 1pp below native; complete VinDr "
                "ontology mention sets exactly matched"
            ),
        },
        "boundary": (
            "A named-only pass is an attribute-binding mechanism result, not an open-generation "
            "mitigation. The method advances only if the exact same-answer natural compilation passes."
        ),
    }


def generate(adapter: Any, image: Any, prompt: str, max_new_tokens: int, seed: int) -> str:
    value = adapter.generate_control(
        image,
        prompt,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return value.text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu", "llava"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--orientation-certificate", type=Path)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pool_size = max(args.limit, args.candidate_pool_size)
    cases = build_cases(args.csv, args.image_root, pool_size, args.seed)
    orientation_certificate_sha256 = None
    if args.orientation_certificate is not None:
        certificates = load_orientation_certificates(args.orientation_certificate)
        cases = [
            {**case, "orientation_certificate": certificates[case["image_id"]]}
            for case in cases
            if case["image_id"] in certificates
        ]
        if len(cases) < args.limit:
            raise ValueError(
                f"only {len(cases)} orientation-certified cases in a pool of {pool_size}; "
                f"requested {args.limit}"
            )
        cases = cases[: args.limit]
        orientation_certificate_sha256 = sha256_file(args.orientation_certificate)
    else:
        cases = cases[: args.limit]
    config = {
        "version": VERSION,
        "created_at": now(),
        "model": args.model,
        "csv": str(args.csv.resolve()),
        "csv_sha256": sha256_file(args.csv),
        "image_root": str(args.image_root.resolve()),
        "limit": args.limit,
        "candidate_pool_size": pool_size,
        "orientation_certificate": (
            str(args.orientation_certificate.resolve())
            if args.orientation_certificate is not None
            else None
        ),
        "orientation_certificate_sha256": orientation_certificate_sha256,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "cases": cases,
        "method": (
            "patient-frame native control plus screen-frame generation; deterministic "
            "left/right Z2 compilation"
        ),
        "content_contract": (
            "within the method arm only whole-word left/right may change during compilation; "
            "native versus screen-arm finding recall is audited separately"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume")
        old = json.loads(config_path.read_text())
        for key in (
            "version",
            "model",
            "csv",
            "csv_sha256",
            "image_root",
            "limit",
            "candidate_pool_size",
            "orientation_certificate",
            "orientation_certificate_sha256",
            "seed",
            "max_new_tokens",
            "cases",
            "method",
            "content_contract",
        ):
            if old[key] != config[key]:
                raise RuntimeError(f"resume config drift: {key}")
    else:
        atomic_json(config_path, config)

    raw_path = args.output_dir / "raw.jsonl"
    completed: set[str] = set()
    if raw_path.exists() and args.resume:
        completed = {
            json.loads(line)["case_key"]
            for line in raw_path.read_text().splitlines()
            if line.strip()
        }

    adapter = load_oe_adapter(args.model)
    try:
        for index, case in enumerate(cases):
            if case["case_key"] in completed:
                continue
            row: dict[str, Any] = {**case, "version": VERSION, "status": "error"}
            try:
                image = dicom_to_pil(Path(case["image"]))
                direct = generate(
                    adapter,
                    image,
                    named_prompt(case["left_finding"], case["right_finding"]),
                    args.max_new_tokens,
                    args.seed + index * 4,
                )
                screen = generate(
                    adapter,
                    image,
                    screen_prompt(case["left_finding"], case["right_finding"]),
                    args.max_new_tokens,
                    args.seed + index * 4 + 1,
                )
                named_screen_contract = screen_coordinate_contract(screen)
                screen_compiled = (
                    compile_screen_to_patient(screen)
                    if named_screen_contract
                    else ""
                )
                natural = generate(
                    adapter,
                    image,
                    natural_prompt(),
                    args.max_new_tokens,
                    args.seed + index * 4 + 2,
                )
                natural_compiled = swap_laterality(natural)
                natural_screen = generate(
                    adapter,
                    image,
                    natural_screen_prompt(),
                    args.max_new_tokens,
                    args.seed + index * 4 + 3,
                )
                natural_screen_contract = screen_coordinate_contract(natural_screen)
                natural_screen_compiled = (
                    compile_screen_to_patient(natural_screen)
                    if natural_screen_contract
                    else ""
                )
                row.update(
                    {
                        "status": "ok",
                        "named_direct_answer": direct,
                        "named_direct_parse": parse_binding(
                            direct, case["left_finding"], case["right_finding"]
                        ),
                        "named_screen_answer": screen,
                        "named_screen_contract": named_screen_contract,
                        # Patient-right is displayed on screen-left.
                        "named_screen_parse": parse_binding(
                            screen, case["right_finding"], case["left_finding"]
                        ),
                        "named_compiled_answer": screen_compiled,
                        "named_compiled_parse": parse_binding(
                            screen_compiled, case["left_finding"], case["right_finding"]
                        ),
                        "named_content_preserved": erase_frame_words(screen)
                        == erase_frame_words(screen_compiled),
                        "natural_answer": natural,
                        "natural_compiled_answer": natural_compiled,
                        "natural_content_preserved": erase_frame_words(natural)
                        == erase_frame_words(natural_compiled),
                        "natural_screen_answer": natural_screen,
                        "natural_screen_contract": natural_screen_contract,
                        "natural_screen_compiled_answer": natural_screen_compiled,
                        "natural_screen_content_preserved": erase_frame_words(natural_screen)
                        == erase_frame_words(natural_screen_compiled),
                        "completed_at": now(),
                    }
                )
            except Exception as error:
                row.update(
                    {
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "completed_at": now(),
                    }
                )
            append_jsonl(raw_path, row)
            completed.add(case["case_key"])
            print(
                f"[{len(completed)}/{len(cases)}] {case['case_key']} "
                f"status={row['status']} direct={row.get('named_direct_parse')} "
                f"screen={row.get('named_screen_parse')} compiled={row.get('named_compiled_parse')}",
                flush=True,
            )
    finally:
        adapter.close()

    rows = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
    ]
    atomic_json(
        args.output_dir / "analysis.json",
        analyze(rows, args.bootstrap_draws, args.seed),
    )


if __name__ == "__main__":
    main()
