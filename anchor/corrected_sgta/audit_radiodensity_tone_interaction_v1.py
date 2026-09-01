#!/usr/bin/env python3
"""CPU-only screen for a tone-nuisance by radiodensity-sign interaction.

The audit reuses frozen Huatuo CE logits collected before this hypothesis.  It
does not rerun a model, relabel clinical truth, or infer a source domain.  The
candidate mechanism predicts that global brightening is spuriously treated as
positive evidence for opacity-family claims relative to lucency-family claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_ID = "radiodensity-signed-tone-substitution-screen-v1"
BRIGHT_ARM = "gamma_0.9"
DARK_ARM = "gamma_1.1"
BASELINE_ARM = "original"
SEED = 7319
PERMUTATIONS = 10_000
BOOTSTRAPS = 10_000
MIN_FAMILY_N = 20
MIN_TRUTH_CELL_N = 10
MIN_LOGIT_INTERACTION = 0.05

OPACITY = re.compile(
    r"\b(opacit\w*|consolidat\w*|pleural effusion\w*|pulmonary edema|"
    r"interstitial edema|vascular congestion|vascular engorgement|"
    r"atelecta\w*|pneumonia|aspirat\w*|increased interstitial markings?)\b",
    re.IGNORECASE,
)
LUCENCY = re.compile(
    r"\b(pneumothora\w*|emphysema|hyperinflation|subcutaneous (?:gas|emphysema))\b",
    re.IGNORECASE,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_claim(question: str) -> str | None:
    opacity = bool(OPACITY.search(question))
    lucency = bool(LUCENCY.search(question))
    if opacity == lucency:
        return None
    return "opacity" if opacity else "lucency"


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requested for empty cell")
    return statistics.fmean(values)


def _truth_stratified_interaction(rows: list[dict[str, Any]]) -> float:
    effects = []
    for truth in ("yes", "no"):
        opacity = [
            row["bright_minus_dark"]
            for row in rows
            if row["family"] == "opacity" and row["ground_truth"] == truth
        ]
        lucency = [
            row["bright_minus_dark"]
            for row in rows
            if row["family"] == "lucency" and row["ground_truth"] == truth
        ]
        effects.append(_mean(opacity) - _mean(lucency))
    return statistics.fmean(effects)


def _permutation_p(rows: list[dict[str, Any]], observed: float) -> float:
    rng = random.Random(SEED)
    by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_counts = Counter()
    for row in rows:
        by_truth[row["ground_truth"]].append(row)
        family_counts[(row["ground_truth"], row["family"])] += 1
    exceed = 0
    for _ in range(PERMUTATIONS):
        permuted = []
        for truth in ("yes", "no"):
            group = by_truth[truth]
            values = [row["bright_minus_dark"] for row in group]
            rng.shuffle(values)
            n_opacity = family_counts[(truth, "opacity")]
            permuted.extend(
                {"ground_truth": truth, "family": "opacity", "bright_minus_dark": value}
                for value in values[:n_opacity]
            )
            permuted.extend(
                {"ground_truth": truth, "family": "lucency", "bright_minus_dark": value}
                for value in values[n_opacity:]
            )
        if abs(_truth_stratified_interaction(permuted)) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (PERMUTATIONS + 1)


def _bootstrap_ci(rows: list[dict[str, Any]]) -> list[float]:
    rng = random.Random(SEED + 1)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["family"], row["ground_truth"])].append(row)
    estimates = []
    for _ in range(BOOTSTRAPS):
        sample = []
        for cell in sorted(cells):
            source = cells[cell]
            sample.extend(source[rng.randrange(len(source))] for _ in source)
        estimates.append(_truth_stratified_interaction(sample))
    estimates.sort()
    return [
        estimates[int(0.025 * BOOTSTRAPS)],
        estimates[int(0.975 * BOOTSTRAPS) - 1],
    ]


def audit(raw_path: Path) -> dict[str, Any]:
    source = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    eligible = []
    exclusions = Counter()
    for row in source:
        family = classify_claim(str(row["question"]))
        if family is None:
            exclusions["not_one_unambiguous_radiodensity_family"] += 1
            continue
        scores = row.get("scores", {})
        guards = row.get("style_guards", {})
        if not all(arm in scores for arm in (BASELINE_ARM, BRIGHT_ARM, DARK_ARM)):
            exclusions["missing_frozen_score_arm"] += 1
            continue
        if guards.get(BRIGHT_ARM, {}).get("passed") is not True or guards.get(
            DARK_ARM, {}
        ).get("passed") is not True:
            exclusions["style_guard_failed"] += 1
            continue
        truth = str(row["ground_truth"]).casefold()
        if truth not in {"yes", "no"}:
            exclusions["nonbinary_ground_truth"] += 1
            continue
        baseline = float(scores[BASELINE_ARM]["yes_minus_no"])
        bright = float(scores[BRIGHT_ARM]["yes_minus_no"])
        dark = float(scores[DARK_ARM]["yes_minus_no"])
        eligible.append(
            {
                "question_id": str(row["question_id"]),
                "patient_id": str(row["patient_id"]),
                "family": family,
                "ground_truth": truth,
                "baseline_margin": baseline,
                "bright_minus_dark": bright - dark,
            }
        )

    cell_counts = Counter((row["family"], row["ground_truth"]) for row in eligible)
    family_counts = Counter(row["family"] for row in eligible)
    observed = _truth_stratified_interaction(eligible)
    ci = _bootstrap_ci(eligible)
    p_value = _permutation_p(eligible, observed)
    raw_difference = _mean(
        [row["bright_minus_dark"] for row in eligible if row["family"] == "opacity"]
    ) - _mean(
        [row["bright_minus_dark"] for row in eligible if row["family"] == "lucency"]
    )
    prevalence_gate = all(family_counts[family] >= MIN_FAMILY_N for family in ("opacity", "lucency"))
    truth_cell_gate = all(
        cell_counts[(family, truth)] >= MIN_TRUTH_CELL_N
        for family in ("opacity", "lucency")
        for truth in ("yes", "no")
    )
    direction_gate = observed >= MIN_LOGIT_INTERACTION and ci[0] > 0.0 and p_value <= 0.05
    passed = prevalence_gate and truth_cell_gate and direction_gate
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "candidate_passed_cpu_screen" if passed else "candidate_rejected_cpu_screen",
        "candidate_mechanism": "radiodensity-signed tone substitution",
        "claim_boundary": (
            "Tests a frozen CE logit interaction only; it does not establish clinical "
            "semantic preservation, a source domain, OE hallucination, or causality."
        ),
        "input": str(raw_path.resolve()),
        "input_sha256": _sha(raw_path),
        "source_sha256": _sha(Path(__file__).resolve()),
        "model": "HuatuoGPT-Vision-7B",
        "dataset": "RULE/MIMIC-CXR balanced CE cache",
        "arms": {
            "bright": BRIGHT_ARM,
            "dark": DARK_ARM,
            "baseline": BASELINE_ARM,
            "contrast": "yes_minus_no(bright) - yes_minus_no(dark)",
        },
        "n_source": len(source),
        "n_eligible": len(eligible),
        "family_counts": dict(sorted(family_counts.items())),
        "truth_cell_counts": {
            f"{family}_{truth}": cell_counts[(family, truth)]
            for family in ("opacity", "lucency")
            for truth in ("yes", "no")
        },
        "exclusions": dict(sorted(exclusions.items())),
        "estimates": {
            "raw_opacity_minus_lucency": raw_difference,
            "truth_stratified_opacity_minus_lucency": observed,
            "bootstrap_95ci": ci,
            "two_sided_within_truth_permutation_p": p_value,
        },
        "frozen_gates": {
            "minimum_family_n": MIN_FAMILY_N,
            "minimum_each_family_truth_cell_n": MIN_TRUTH_CELL_N,
            "minimum_truth_stratified_logit_interaction": MIN_LOGIT_INTERACTION,
            "bootstrap_lower_gt_zero": ci[0] > 0.0,
            "permutation_p_le_0.05": p_value <= 0.05,
            "prevalence_gate": prevalence_gate,
            "truth_cell_gate": truth_cell_gate,
            "direction_and_effect_gate": direction_gate,
            "all_passed": passed,
        },
        "decision": (
            "Do not spend GPU or reinterpret generic style sensitivity as a "
            "radiodensity mechanism."
            if not passed
            else "Authorize only a pre-registered VinDr replication, not a method claim."
        ),
        "rows": eligible,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw.resolve())
    result["command"] = [shlex.join([str(Path(__file__)), *os.sys.argv[1:]])]
    _atomic_write(
        args.output.resolve(),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
