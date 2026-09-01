"""Fail-closed CECD pilot/dev/locked-confirmation verifier.

Legacy pilot-as-dev artifacts may remain on disk but can never satisfy this
contract or authorize the dual-semantics transition.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 import (
    CONFIRMATION_VERSION,
    DEV_FIT_VERSION,
    apply_confirmation_stage,
    fit_dev_stage,
    load_inputs,
)
from corrected_sgta.cecd_admission_gate import (
    EXPECTED_VERSION as ADMISSION_ANALYSIS_VERSION,
    require_cecd_authorization,
)
from corrected_sgta.run_cecd_factorial_v1 import (
    FROZEN_FINDINGS,
    STAGE_SPECS,
    canonical_json_sha256,
    cell_specs,
    sha256_file,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json
from corrected_sgta.verify_cecd_two_model_stage1_v2 import (
    _verify_model_provenance,
    _verify_next_token_conformance,
)


VERSION = "cecd-three-stage-input-gate-v4-independent-recomputation"
ANALYZER_SOURCE = Path(__file__).with_name(
    "analyze_clinical_equivalence_composition_defect_v1.py"
)
EXPECTED_SELECTION_HASHES = {
    "pilot_screen": "276bac3ffe3f06e47e6377f3dcc2b5877959a9ad372cd1c5801629719051a24a",
    "dev_fit": "2e9b0b0c427068e017a5ce1fbc098dbe7028bfacfa3604d395aa782807e57420",
    "confirmation_locked": "39195d0f606da9acfa1b2b2de413176496efa6e4e235c09913c417a95c6bd1e9",
}
COMMON_SCIENTIFIC_KEYS = (
    "version", "measurement_name", "dataset", "manifest_sha256", "bboxes_sha256",
    "stage_label", "manifest_split", "split", "findings", "votes",
    "per_finding_vote_bin", "seed", "frozen_claim_count",
    "frozen_selection_keys_sha256", "science_render_names", "identity_render_name",
    "prompt_templates", "prompt_contract", "cells_per_claim", "missing_cell_policy",
    "readout", "next_token_conformance", "source_sha256",
)


def canonical_input_hashes(values: Mapping[str, Any], root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw_path, digest in values.items():
        path = Path(str(raw_path))
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        key = str(resolved)
        if key in output:
            raise RuntimeError("analysis aliases one input path")
        output[key] = str(digest)
    return output


def verify_stage_run(
    *, family: str, stage: str, run_dir: Path, admission: Path
) -> dict[str, Any]:
    if stage not in STAGE_SPECS:
        raise RuntimeError(f"unknown stage: {stage}")
    config_path = run_dir / "config.json"
    rows_path = run_dir / "factorial_rows.jsonl"
    manifest_path = run_dir / "factorial_rows_manifest.json"
    for path in (config_path, rows_path, manifest_path):
        if not path.is_file():
            raise RuntimeError(f"{family}/{stage}: missing {path.name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = STAGE_SPECS[stage]
    claims = int(spec["claims"])
    expected_rows = claims * 19
    expected_admission = {
        "status": "passed_hash_bound",
        "analysis_path": str(admission.resolve()),
        "analysis_sha256": sha256_file(admission),
        "analysis_version": ADMISSION_ANALYSIS_VERSION,
        "cecd_model_scoring_authorized": True,
    }
    immutable = {
        key: value for key, value in config.items()
        if key not in {"created_at", "command", "fingerprint"}
    }
    missing_scientific = [key for key in COMMON_SCIENTIFIC_KEYS if key not in config]
    if missing_scientific:
        raise RuntimeError(
            f"{family}/{stage}: scientific contract lacks {missing_scientific}"
        )
    scientific_contract_sha256 = canonical_json_sha256(
        {key: config[key] for key in COMMON_SCIENTIFIC_KEYS}
    )
    required = bool(
        config.get("model_family") == family
        and config.get("stage_label") == stage
        and config.get("manifest_split") == spec["manifest_split"]
        and config.get("split") == spec["manifest_split"]
        and config.get("per_finding_vote_bin") == spec["per_bin"]
        and config.get("frozen_claim_count") == claims
        and config.get("active_claim_count") == claims
        and config.get("engineering_canary_max_claims") is None
        and config.get("frozen_selection_keys_sha256") == EXPECTED_SELECTION_HASHES[stage]
        and config.get("active_selection_keys_sha256") == EXPECTED_SELECTION_HASHES[stage]
        and config.get("scientific_status") == f"human_admitted_cecd_{stage}"
        and config.get("execution_mode") == "formal_human_admitted_model_scoring"
        and config.get("clinical_equivalence_established") is True
        and config.get("cecd_model_scoring_authorized") is True
        and config.get("clinical_admission") == expected_admission
        and config.get("fingerprint") == canonical_json_sha256(immutable)
        and manifest.get("config_fingerprint") == config.get("fingerprint")
        and manifest.get("claims") == claims
        and manifest.get("rows") == expected_rows
        and manifest.get("complete_orbit_count") == claims
        and manifest.get("incomplete_orbit_count") == 0
        and manifest.get("factorial_rows_sha256") == sha256_file(rows_path)
    )
    if not required:
        raise RuntimeError(f"{family}/{stage}: config/manifest contract mismatch")
    orbits: dict[tuple[str, str], set[str]] = defaultdict(set)
    finding_counts: Counter[str] = Counter()
    for line_number, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image = str(row.get("image_id", ""))
        finding = str(row.get("finding", ""))
        cell_id = str(row.get("cell_id", ""))
        if (
            row.get("model") != config["model"]
            or row.get("config_fingerprint") != config["fingerprint"]
            or row.get("stage_label") != stage
            or row.get("source_manifest_split") != spec["manifest_split"]
            or finding not in FROZEN_FINDINGS
            or not image or not cell_id
        ):
            raise RuntimeError(f"{family}/{stage}: row drift at line {line_number}")
        key = (image, finding)
        if cell_id in orbits[key]:
            raise RuntimeError(f"{family}/{stage}: duplicate cell")
        orbits[key].add(cell_id)
    if len(orbits) != claims:
        raise RuntimeError(f"{family}/{stage}: orbit count mismatch")
    for (image, finding), cells in orbits.items():
        if cells != {item.cell_id for item in cell_specs(finding)}:
            raise RuntimeError(f"{family}/{stage}: incomplete orbit {image}/{finding}")
        finding_counts[finding] += 1
    expected_per_finding = int(spec["per_bin"]) * 4
    if finding_counts != Counter({finding: expected_per_finding for finding in FROZEN_FINDINGS}):
        raise RuntimeError(f"{family}/{stage}: finding balance mismatch")
    model_provenance_sha256 = _verify_model_provenance(
        family=family, config=config
    )
    conformance = _verify_next_token_conformance(
        family=family, run_dir=run_dir, config=config
    )
    return {
        "family": family,
        "stage": stage,
        "run_dir": str(run_dir.resolve()),
        "model": config["model"],
        "claims": claims,
        "rows": expected_rows,
        "factorial_rows": str(rows_path.resolve()),
        "factorial_rows_sha256": sha256_file(rows_path),
        "selection_keys_sha256": EXPECTED_SELECTION_HASHES[stage],
        "image_ids": sorted({image for image, _ in orbits}),
        "config_sha256": sha256_file(config_path),
        "admission_sha256": sha256_file(admission),
        "model_provenance_sha256": model_provenance_sha256,
        "next_token_conformance": conformance,
        "scientific_contract_sha256": scientific_contract_sha256,
    }


def _verify_analysis_inputs(
    *, artifact: Mapping[str, Any], expected_runs: list[Mapping[str, Any]], root: Path,
    expected_mode: str,
) -> None:
    provenance = artifact.get("provenance", {})
    inputs = provenance.get("input_sha256")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("analysis input provenance missing")
    if (
        provenance.get("code_sha256") != sha256_file(ANALYZER_SOURCE)
        or provenance.get("mode") != expected_mode
        or provenance.get("seed") != 42
        or provenance.get("folds") != 5
        or provenance.get("bootstrap_draws") != 5000
    ):
        raise RuntimeError("analysis code or frozen execution contract mismatch")
    expected = {
        str(Path(str(run["factorial_rows"])).resolve()): str(run["factorial_rows_sha256"])
        for run in expected_runs
    }
    if canonical_input_hashes(inputs, root) != expected:
        raise RuntimeError("analysis input hashes do not match verified runs")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"locked confirmation lacks finite {label}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"locked confirmation lacks finite {label}")
    return result


def _ci(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"locked confirmation lacks {label} interval")
    lower, upper = (_finite(item, label) for item in value)
    if lower > upper:
        raise RuntimeError(f"locked confirmation has reversed {label} interval")
    return lower, upper


def _recompute_model_gate(model: str, result: Mapping[str, Any]) -> bool:
    """Independently reconstruct the frozen confirmation conjunction.

    This deliberately duplicates the threshold logic instead of trusting the
    analyzer's booleans.  The analyzer artifact is subsequently compared to a
    fresh raw-input recomputation as a second, stronger check.
    """

    pooled = result.get("pooled_four_finding_delta_auc")
    harmful = result.get("pooled_harmful_alignment")
    rms = result.get("interaction_rms_reader_equivalents")
    identity = result.get("identity_controls")
    slopes = result.get("reader_slope_cluster_bootstrap")
    per_finding = result.get("per_finding")
    components = result.get("gate_components")
    if not all(
        isinstance(value, Mapping)
        for value in (pooled, harmful, rms, identity, slopes, per_finding, components)
    ):
        raise RuntimeError(f"{model}: locked confirmation model metrics are incomplete")

    pooled_ci = _ci(
        pooled.get("image_cluster_bootstrap", {}).get("delta_auc_ci95"),
        f"{model} pooled delta-AUROC",
    )
    pooled_pass = bool(
        _finite(pooled.get("delta_auc"), f"{model} pooled delta-AUROC") >= 0.03
        and pooled_ci[0] > 0
    )
    harmful_pass = _ci(harmful.get("ci95"), f"{model} harmful alignment")[0] > 0
    rms_point = _finite(rms.get("point"), f"{model} interaction RMS")
    rms_pass = bool(
        rms_point >= 0.25
        and _ci(rms.get("ci95"), f"{model} interaction RMS")[0] > 0
    )
    maximum_identity = _finite(
        identity.get("maximum_rms_re"), f"{model} maximum identity RMS"
    )
    identity_pass = maximum_identity <= 0.1 * rms_point

    if len(slopes) != 4:
        raise RuntimeError(f"{model}: confirmation must contain four reader slopes")
    reader_pass = all(
        _ci(row.get("ci95"), f"{model}/{finding} reader slope")[0] > 0
        for finding, row in slopes.items()
        if isinstance(row, Mapping)
    )
    if not reader_pass or any(not isinstance(row, Mapping) for row in slopes.values()):
        reader_pass = False

    if len(per_finding) != 4 or any(
        not isinstance(row, Mapping) for row in per_finding.values()
    ):
        raise RuntimeError(f"{model}: confirmation must contain four finding metrics")
    delta_positive = 0
    alignment_positive = 0
    no_opposite_mcid = True
    no_significant_opposite = True
    for finding, row in per_finding.items():
        delta = _finite(row.get("delta_auc"), f"{model}/{finding} delta-AUROC")
        alignment = row.get("harmful_alignment")
        point = _finite(
            alignment.get("point") if isinstance(alignment, Mapping) else None,
            f"{model}/{finding} harmful alignment",
        )
        delta_ci = _ci(
            row.get("image_cluster_bootstrap", {}).get("delta_auc_ci95"),
            f"{model}/{finding} delta-AUROC",
        )
        delta_positive += int(delta > 0)
        alignment_positive += int(point > 0)
        no_opposite_mcid = bool(no_opposite_mcid and delta > -0.03)
        no_significant_opposite = bool(
            no_significant_opposite and delta_ci[1] >= 0
        )
    heterogeneity_pass = bool(
        delta_positive >= 3
        and alignment_positive >= 3
        and no_opposite_mcid
        and no_significant_opposite
    )
    expected_components = {
        "pooled_delta_auc_point_at_least_0p03_and_ci_above_zero": pooled_pass,
        "pooled_harmful_alignment_ci_above_zero": harmful_pass,
        "interaction_rms_at_least_0p25_re_and_ci_above_zero": rms_pass,
        "identity_below_one_tenth": identity_pass,
        "all_reader_slopes_ci_above_zero": reader_pass,
        "heterogeneity_guard": heterogeneity_pass,
    }
    if dict(components) != expected_components:
        raise RuntimeError(f"{model}: asserted gate components disagree with metrics")
    asserted_heterogeneity = result.get("heterogeneity_guard")
    if not isinstance(asserted_heterogeneity, Mapping) or dict(asserted_heterogeneity) != {
        "delta_positive_findings": delta_positive,
        "harmful_alignment_positive_findings": alignment_positive,
        "no_finding_delta_at_or_below_minus_0p03": no_opposite_mcid,
        "no_finding_ci_strictly_below_zero": no_significant_opposite,
        "passed": heterogeneity_pass,
    }:
        raise RuntimeError(f"{model}: asserted heterogeneity guard disagrees with metrics")
    expected_pass = bool(
        pooled_pass
        and harmful_pass
        and rms_pass
        and identity_pass
        and reader_pass
        and heterogeneity_pass
    )
    if result.get("model_confirmation_pass") is not expected_pass:
        raise RuntimeError(f"{model}: asserted model pass disagrees with metrics")
    if identity.get("below_one_tenth") is not identity_pass:
        raise RuntimeError(f"{model}: asserted identity control disagrees with metrics")
    return expected_pass


def _validate_recomputed_confirmation_gate(
    confirmation: Mapping[str, Any], expected_models: set[str]
) -> dict[str, Any]:
    models = confirmation.get("models")
    if not isinstance(models, Mapping) or set(models) != expected_models:
        raise RuntimeError("recomputed confirmation model set mismatch")
    passing = sorted(
        model
        for model, result in models.items()
        if _recompute_model_gate(str(model), result)
    )
    gate = confirmation.get("gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("locked confirmation gate is missing")
    authorized = len(passing) == len(expected_models) == 2
    if (
        gate.get("name") != "behavioral_confirmation_locked_v1"
        or gate.get("confirmation_passing_models") != passing
        or gate.get("both_models_pass") is not authorized
        or gate.get("authorized_for_method_level_treble_adapter_run") is not authorized
        or gate.get("authorized_for_hidden_state_stage") is not False
        or gate.get("behavioral_phenomenon_confirmed_on_locked_test") is not authorized
    ):
        raise RuntimeError("locked confirmation gate disagrees with recomputed model gates")
    return {"passing_models": passing, "authorized": authorized}


def _analysis_core(artifact: Mapping[str, Any], ignored: set[str]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key not in ignored}


def _require_exact_recomputation(
    *, label: str, artifact: Mapping[str, Any], recomputed: Mapping[str, Any],
    ignored: set[str],
) -> None:
    observed = canonical_json_sha256(_analysis_core(artifact, ignored))
    expected = canonical_json_sha256(_analysis_core(recomputed, ignored))
    if observed != expected:
        raise RuntimeError(
            f"{label} artifact disagrees with independent raw-input recomputation"
        )


def _recompute_three_stage_analysis(
    *, runs: Mapping[str, list[Mapping[str, Any]]],
    dev_fit: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    dev_payload = load_inputs(
        [Path(str(row["factorial_rows"])) for row in runs["dev_fit"]]
    )
    recomputed_dev = fit_dev_stage(dev_payload, folds=5, draws=5000, seed=42)
    _require_exact_recomputation(
        label="dev-fit", artifact=dev_fit, recomputed=recomputed_dev,
        ignored={"provenance"},
    )
    confirmation_payload = load_inputs(
        [Path(str(row["factorial_rows"])) for row in runs["confirmation_locked"]]
    )
    recomputed_confirmation = apply_confirmation_stage(
        confirmation_payload, recomputed_dev, draws=5000, seed=42
    )
    _require_exact_recomputation(
        label="locked-confirmation", artifact=confirmation,
        recomputed=recomputed_confirmation,
        ignored={"provenance", "dev_fit_binding"},
    )
    return recomputed_confirmation


def verify_three_stage(
    *, admission: Path, run_dirs: Mapping[str, Mapping[str, Path]],
    dev_fit_path: Path, confirmation_path: Path, root: Path,
) -> dict[str, Any]:
    authorization = require_cecd_authorization(admission)
    runs: dict[str, list[dict[str, Any]]] = {}
    for stage in STAGE_SPECS:
        runs[stage] = [
            verify_stage_run(
                family=family, stage=stage, run_dir=run_dirs[family][stage],
                admission=admission,
            )
            for family in ("huatuo", "hulu")
        ]
        if len({row["model"] for row in runs[stage]}) != 2:
            raise RuntimeError(f"{stage}: models are not distinct")
        if len({row["scientific_contract_sha256"] for row in runs[stage]}) != 1:
            raise RuntimeError(f"{stage}: models do not share one scientific contract")
        if len({tuple(row["image_ids"]) for row in runs[stage]}) != 1:
            raise RuntimeError(f"{stage}: model runs use different images")
    stage_images = {
        stage: set(runs[stage][0]["image_ids"]) for stage in STAGE_SPECS
    }
    for family in ("huatuo", "hulu"):
        family_runs = [
            row for stage in STAGE_SPECS
            for row in runs[stage] if row["family"] == family
        ]
        identities = {row["model"] for row in family_runs}
        if len(identities) != 1:
            raise RuntimeError(f"{family}: model identity changes across stages")
        provenances = {row["model_provenance_sha256"] for row in family_runs}
        if len(provenances) != 1:
            raise RuntimeError(f"{family}: model weights change across stages")
    stages = list(STAGE_SPECS)
    overlap = {
        f"{left}__{right}": len(stage_images[left] & stage_images[right])
        for index, left in enumerate(stages) for right in stages[index + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"whole-image stage leakage: {overlap}")
    dev_fit = json.loads(dev_fit_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if (
        dev_fit.get("version") != DEV_FIT_VERSION
        or dev_fit.get("status") != "dev_fit_complete_confirmation_not_opened"
        or dev_fit.get("stage_label") != "dev_fit"
        or dev_fit.get("source_manifest_split") != "dev"
        or dev_fit.get("gate", {}).get("authorized_for_method_level_treble_adapter_run") is not False
        or set(dev_fit.get("models", {})) != {row["model"] for row in runs["dev_fit"]}
    ):
        raise RuntimeError("dev-fit artifact contract mismatch")
    _verify_analysis_inputs(
        artifact=dev_fit, expected_runs=runs["dev_fit"], root=root,
        expected_mode="dev_fit",
    )
    if (
        confirmation.get("version") != CONFIRMATION_VERSION
        or confirmation.get("status") != "complete"
        or confirmation.get("stage_label") != "confirmation_locked"
        or confirmation.get("source_manifest_split") != "confirmation"
        or confirmation.get("dev_fit_binding", {}).get("path") != str(dev_fit_path.resolve())
        or confirmation.get("dev_fit_binding", {}).get("sha256") != sha256_file(dev_fit_path)
        or confirmation.get("gate", {}).get("name") != "behavioral_confirmation_locked_v1"
        or confirmation.get("gate", {}).get("authorized_for_hidden_state_stage") is not False
        or set(confirmation.get("models", {}))
        != {row["model"] for row in runs["confirmation_locked"]}
    ):
        raise RuntimeError("locked-confirmation artifact contract mismatch")
    _verify_analysis_inputs(
        artifact=confirmation, expected_runs=runs["confirmation_locked"], root=root,
        expected_mode="confirmation_locked",
    )
    expected_models = {row["model"] for row in runs["confirmation_locked"]}
    recomputed_confirmation = _recompute_three_stage_analysis(
        runs=runs, dev_fit=dev_fit, confirmation=confirmation
    )
    recomputed_decision = _validate_recomputed_confirmation_gate(
        recomputed_confirmation, expected_models
    )
    public_runs = {
        stage: [
            {key: value for key, value in row.items() if key != "image_ids"}
            for row in stage_runs
        ]
        for stage, stage_runs in runs.items()
    }
    return {
        "version": VERSION,
        "status": "passed",
        "passed": True,
        "admission": {
            "path": str(admission.resolve()), "sha256": sha256_file(admission),
            "version": authorization["version"],
        },
        "runs": public_runs,
        "whole_image_overlap_counts": overlap,
        "dev_fit": {"path": str(dev_fit_path.resolve()), "sha256": sha256_file(dev_fit_path)},
        "confirmation_locked": {
            "path": str(confirmation_path.resolve()),
            "sha256": sha256_file(confirmation_path),
            "behavioral_gate_passed": bool(recomputed_decision["authorized"]),
            "scientific_gate_authority": "independent_raw_input_recomputation",
        },
        "legacy_pilot_as_dev_authorized": False,
        "legacy_v1_v3_artifacts_authorized": False,
        "authorized_for_method_level_treble_adapter_run": bool(
            recomputed_decision["authorized"]
        ),
        "hidden_state_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    for family in ("huatuo", "hulu"):
        for stage in STAGE_SPECS:
            parser.add_argument(f"--{family}-{stage.replace('_', '-')}-dir", type=Path, required=True)
    parser.add_argument("--dev-fit", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = {
        family: {
            stage: getattr(args, f"{family}_{stage}_dir")
            for stage in STAGE_SPECS
        }
        for family in ("huatuo", "hulu")
    }
    result = verify_three_stage(
        admission=args.admission, run_dirs=run_dirs,
        dev_fit_path=args.dev_fit, confirmation_path=args.confirmation,
        root=Path.cwd(),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
