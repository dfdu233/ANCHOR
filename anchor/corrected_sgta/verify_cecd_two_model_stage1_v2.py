"""Fail-closed input gate before combining CECD Huatuo and Hulu Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from corrected_sgta.cecd_admission_gate import require_cecd_authorization
from corrected_sgta.cecd_admission_gate import EXPECTED_VERSION as ADMISSION_ANALYSIS_VERSION
from corrected_sgta.run_cecd_factorial_v1 import (
    FROZEN_FINDINGS,
    FROZEN_PER_BIN,
    FROZEN_SEED,
    FROZEN_VOTES,
    IDENTITY_RENDER_NAME,
    MEASUREMENT_NAME,
    PROMPT_TEMPLATES,
    SCIENCE_RENDER_NAMES,
    VERSION as FACTORIAL_VERSION,
    canonical_json_sha256,
    cell_specs,
    full_model_artifact_fingerprint,
    python_source_tree_fingerprint,
    sha256_file,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json


VERSION = "cecd-two-model-stage1-input-gate-v2"
ROW_CONTRACT_VERSION = "clinical-equivalence-factorial-v1"


SCIENTIFIC_CONTRACT_KEYS = (
    "version",
    "measurement_name",
    "dataset",
    "manifest_sha256",
    "bboxes_sha256",
    "split",
    "findings",
    "votes",
    "per_finding_vote_bin",
    "seed",
    "frozen_claim_count",
    "frozen_selection_keys_sha256",
    "active_claim_count",
    "active_selection_keys_sha256",
    "engineering_canary_max_claims",
    "science_render_names",
    "identity_render_name",
    "prompt_templates",
    "prompt_contract",
    "cells_per_claim",
    "missing_cell_policy",
    "readout",
    "next_token_conformance",
    "source_sha256",
)


def _scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in SCIENTIFIC_CONTRACT_KEYS if key not in config]
    if missing:
        raise RuntimeError(f"formal factorial config lacks scientific contract keys: {missing}")
    contract = {key: config[key] for key in SCIENTIFIC_CONTRACT_KEYS}
    expected_constants = {
        "version": FACTORIAL_VERSION,
        "measurement_name": MEASUREMENT_NAME,
        "split": "pilot",
        "findings": list(FROZEN_FINDINGS),
        "votes": list(FROZEN_VOTES),
        "per_finding_vote_bin": FROZEN_PER_BIN,
        "seed": FROZEN_SEED,
        "frozen_claim_count": 160,
        "active_claim_count": 160,
        "engineering_canary_max_claims": None,
        "science_render_names": list(SCIENCE_RENDER_NAMES),
        "identity_render_name": IDENTITY_RENDER_NAME,
        "prompt_templates": [
            {"name": name, "template": template} for name, template in PROMPT_TEMPLATES
        ],
        "cells_per_claim": {
            "science": 15,
            "identity_image_controls": 3,
            "duplicate_prompt_controls": 1,
        },
    }
    for key, expected in expected_constants.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"formal factorial scientific contract drift: {key}")
    return contract


def _verify_cell_closure(
    *, family: str, rows_path: Path, config: dict[str, Any]
) -> tuple[int, int]:
    """Independently re-derive every 19-cell orbit from the frozen contract."""

    orbits: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    finding_counts: Counter[str] = Counter()
    count = 0
    for line_number, line in enumerate(
        rows_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("config_fingerprint") != config["fingerprint"]:
            raise RuntimeError(f"{family}: row config mismatch at line {line_number}")
        if row.get("model") != config["model"]:
            raise RuntimeError(f"{family}: row model mismatch at line {line_number}")
        image_id = str(row.get("image_id", "")).strip()
        finding = str(row.get("finding", "")).strip()
        cell_id = str(row.get("cell_id", "")).strip()
        if not image_id or finding not in FROZEN_FINDINGS or not cell_id:
            raise RuntimeError(f"{family}: invalid orbit identity at line {line_number}")
        key = (image_id, finding)
        if cell_id in orbits[key]:
            raise RuntimeError(f"{family}: duplicate cell {key!r}/{cell_id}")
        orbits[key][cell_id] = row
        count += 1
    if count != 3040 or len(orbits) != 160:
        raise RuntimeError(
            f"{family}: unexpected row/orbit count rows={count}, orbits={len(orbits)}"
        )

    for (image_id, finding), cells in orbits.items():
        specs = {spec.cell_id: spec for spec in cell_specs(finding)}
        if set(cells) != set(specs):
            missing = sorted(set(specs) - set(cells))
            extra = sorted(set(cells) - set(specs))
            raise RuntimeError(
                f"{family}: incomplete 19-cell orbit {(image_id, finding)!r}; "
                f"missing={missing}, extra={extra}"
            )
        for cell_id, spec in specs.items():
            row = cells[cell_id]
            expected = {
                "contract_version": ROW_CONTRACT_VERSION,
                "render_id": spec.render_name,
                "prompt_id": spec.prompt_name,
                "cell_role": spec.role,
                "reference_cell_id": spec.reference_cell_id,
                "prompt_text_sha256": hashlib.sha256(
                    spec.prompt_text.encode()
                ).hexdigest(),
                "status": "ok",
            }
            drift = [key for key, value in expected.items() if row.get(key) != value]
            if drift:
                raise RuntimeError(
                    f"{family}: cell contract drift {(image_id, finding, cell_id)!r}: {drift}"
                )
        finding_counts[finding] += 1
    expected_findings = Counter({finding: 40 for finding in FROZEN_FINDINGS})
    if finding_counts != expected_findings:
        raise RuntimeError(
            f"{family}: frozen finding-orbit balance drift: "
            f"{dict(finding_counts)} != {dict(expected_findings)}"
        )
    return count, len(orbits)


def _verify_next_token_conformance(
    *, family: str, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Require the real generation-path check claimed by the frozen config."""

    path = run_dir / "next_token_conformance.json"
    if not path.is_file():
        raise RuntimeError(f"{family}: next-token conformance artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = config.get("next_token_conformance")
    tolerance = (
        contract.get("centered_tristate_logit_tolerance")
        if isinstance(contract, dict)
        else None
    )
    direct = payload.get("direct_logits")
    standard = payload.get("standard_generation_logits")
    states = {"supported", "refuted", "undetermined"}
    if (
        not isinstance(contract, dict)
        or contract.get("required_before_scientific_scoring") is not True
        or contract.get("choice_must_match") is not True
        or not isinstance(tolerance, (int, float))
        or not 0.0 < float(tolerance) <= 1.0
        or payload.get("version") != FACTORIAL_VERSION
        or payload.get("config_fingerprint") != config.get("fingerprint")
        or payload.get("model") != config.get("model")
        or payload.get("render_id") != "baseline_percentile"
        or payload.get("prompt_id") != "existential"
        or payload.get("passed") is not True
        or payload.get("direct_tristate_choice")
        != payload.get("standard_tristate_choice")
        or not isinstance(payload.get("centered_tristate_max_abs_error"), (int, float))
        or not math.isfinite(float(payload["centered_tristate_max_abs_error"]))
        or float(payload["centered_tristate_max_abs_error"]) > float(tolerance)
        or float(payload.get("tolerance", math.nan)) != float(tolerance)
        or not isinstance(direct, dict)
        or set(direct) != states
        or not isinstance(standard, dict)
        or set(standard) != states
        or not all(math.isfinite(float(value)) for value in (*direct.values(), *standard.values()))
    ):
        raise RuntimeError(f"{family}: next-token conformance contract mismatch")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _verify_model_provenance(*, family: str, config: dict[str, Any]) -> str:
    """Re-hash executable checkpoint assets at the Stage-1 join boundary."""

    provenance = config.get("model_provenance")
    model_dir = Path(str(config.get("model_dir", "")))
    if (
        not isinstance(provenance, dict)
        or provenance.get("mode") != "full_content_hash_including_all_weight_shards"
        or not model_dir.is_dir()
    ):
        raise RuntimeError(f"{family}: executable model provenance is incomplete")
    current_model = full_model_artifact_fingerprint(model_dir)
    if provenance.get("full_fingerprint") != current_model:
        raise RuntimeError(f"{family}: checkpoint or local runtime asset hash drift")
    external = provenance.get("external_runtime_source")
    if family == "huatuo":
        if not isinstance(external, dict) or not external.get("root"):
            raise RuntimeError("huatuo: external runtime source provenance is missing")
        current_external = python_source_tree_fingerprint(Path(str(external["root"])))
        if external != current_external:
            raise RuntimeError("huatuo: external runtime source hash drift")
    elif external is not None:
        raise RuntimeError("hulu: unexpected external runtime source provenance")
    return canonical_json_sha256(provenance)


def verify_run(*, family: str, run_dir: Path, admission: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    rows_path = run_dir / "factorial_rows.jsonl"
    manifest_path = run_dir / "factorial_rows_manifest.json"
    for path in (config_path, rows_path, manifest_path):
        if not path.is_file():
            raise RuntimeError(f"{family}: missing Stage 1 artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("model_family") != family:
        raise RuntimeError(f"{family}: model family mismatch")
    if config.get("scientific_status") != "human_admitted_behavioral_dev_screen":
        raise RuntimeError(f"{family}: run is not human-admitted")
    if config.get("clinical_equivalence_established") is not True:
        raise RuntimeError(f"{family}: clinical equivalence was not established")
    if (
        config.get("execution_mode") != "formal_human_admitted_model_scoring"
        or config.get("cecd_model_scoring_authorized") is not True
        or config.get("scientific_artifact_authorized") is not True
    ):
        raise RuntimeError(f"{family}: formal execution authorization mismatch")
    if config.get("engineering_canary_max_claims") is not None:
        raise RuntimeError(f"{family}: canary cannot enter the two-model analysis")
    if config.get("active_claim_count") != 160 or config.get("findings") != list(FROZEN_FINDINGS):
        raise RuntimeError(f"{family}: frozen 160-claim substrate drift")
    admission_record = config.get("clinical_admission")
    expected_admission = {
        "status": "passed_hash_bound",
        "analysis_path": str(admission.resolve()),
        "analysis_sha256": sha256_file(admission),
        "analysis_version": ADMISSION_ANALYSIS_VERSION,
        "cecd_model_scoring_authorized": True,
    }
    if admission_record != expected_admission:
        raise RuntimeError(f"{family}: admission binding mismatch")
    immutable = {
        key: value
        for key, value in config.items()
        if key not in {"created_at", "command", "fingerprint"}
    }
    if config.get("fingerprint") != canonical_json_sha256(immutable):
        raise RuntimeError(f"{family}: config fingerprint mismatch")
    if manifest.get("config_fingerprint") != config["fingerprint"]:
        raise RuntimeError(f"{family}: factorial manifest config mismatch")
    if manifest.get("factorial_rows_sha256") != sha256_file(rows_path):
        raise RuntimeError(f"{family}: factorial rows hash mismatch")
    if (
        manifest.get("claims") != 160
        or manifest.get("rows") != 3040
        or manifest.get("complete_orbit_count") != 160
        or manifest.get("incomplete_orbit_count") != 0
    ):
        raise RuntimeError(f"{family}: incomplete formal factorial")
    scientific_contract = _scientific_contract(config)
    model_provenance_sha256 = _verify_model_provenance(
        family=family, config=config
    )
    conformance = _verify_next_token_conformance(
        family=family, run_dir=run_dir, config=config
    )
    count, orbit_count = _verify_cell_closure(
        family=family, rows_path=rows_path, config=config
    )
    return {
        "family": family,
        "run_dir": str(run_dir.resolve()),
        "config_sha256": sha256_file(config_path),
        "config_fingerprint": config["fingerprint"],
        "factorial_rows_sha256": sha256_file(rows_path),
        "model": config["model"],
        "claims": 160,
        "rows": count,
        "complete_orbit_count": orbit_count,
        "scientific_contract_sha256": canonical_json_sha256(scientific_contract),
        "admission_sha256": expected_admission["analysis_sha256"],
        "next_token_conformance": conformance,
        "model_provenance_sha256": model_provenance_sha256,
    }


def verify_inputs(*, admission: Path, huatuo_dir: Path, hulu_dir: Path) -> dict[str, Any]:
    authorization = require_cecd_authorization(admission)
    runs = [
        verify_run(family="huatuo", run_dir=huatuo_dir, admission=admission),
        verify_run(family="hulu", run_dir=hulu_dir, admission=admission),
    ]
    if len({row["admission_sha256"] for row in runs}) != 1:
        raise RuntimeError("two-model runs bind different human admissions")
    if len({row["model"] for row in runs}) != 2:
        raise RuntimeError("two-model analysis requires two distinct model identities")
    if len({row["scientific_contract_sha256"] for row in runs}) != 1:
        raise RuntimeError("two-model runs do not share one frozen transform/prompt contract")
    return {
        "version": VERSION,
        "status": "passed",
        "passed": True,
        "admission": {
            "path": str(admission.resolve()),
            "sha256": sha256_file(admission),
            "version": authorization["version"],
        },
        "runs": runs,
        "hidden_state_authorized": False,
        "interpretation": "input eligibility only; CECD behavioral gate remains downstream",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--huatuo-dir", type=Path, required=True)
    parser.add_argument("--hulu-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_inputs(
        admission=args.admission,
        huatuo_dir=args.huatuo_dir,
        hulu_dir=args.hulu_dir,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
