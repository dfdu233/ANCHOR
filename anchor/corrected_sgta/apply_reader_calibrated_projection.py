#!/usr/bin/env python3
"""Apply reader-calibrated commitment projection to structured claim rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from corrected_sgta.authorize_reader_grounded_projection import (
    authorization_fingerprint,
)
from corrected_sgta.clinical_claims import (
    evidence_bounded_commitment_projection,
    normalize_term,
    reader_calibrated_state_distribution,
)


VERSION = "reader-calibrated-commitment-projection-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_calibration_provenance(
    row: Mapping[str, object],
    expected_hashes: Mapping[str, str] | None = None,
) -> None:
    provenance = row.get("calibration_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("formal projection requires calibration_provenance")
    required = {
        "formal_reference": True,
        "reference_source": "vindr_reader_votes",
        "calibration_split": "dev",
        "image_disjoint_from_target": True,
    }
    mismatches = {
        key: {"required": expected, "observed": provenance.get(key)}
        for key, expected in required.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"inadmissible calibration provenance: {mismatches}")
    hash_fields = (
        "support_calibrator_sha256",
        "clarity_calibrator_sha256",
        "calibration_manifest_sha256",
        "ontology_sha256",
    )
    invalid_hashes = {
        key: provenance.get(key)
        for key in hash_fields
        if not isinstance(provenance.get(key), str)
        or len(str(provenance.get(key))) != 64
        or any(character not in "0123456789abcdef" for character in str(provenance.get(key)))
    }
    if invalid_hashes:
        raise ValueError(f"missing or invalid calibration hashes: {invalid_hashes}")
    if expected_hashes is not None:
        mismatched_hashes = {
            key: {"expected": expected, "observed": provenance.get(key)}
            for key, expected in expected_hashes.items()
            if provenance.get(key) != expected
        }
        if mismatched_hashes:
            raise ValueError(
                f"calibration provenance hashes do not match files: {mismatched_hashes}"
            )


def validate_mechanism_authorization(
    authorization: Mapping[str, object],
    clarity_gate_sha256: str,
    support_calibrator_sha256: str | None = None,
) -> str:
    if authorization.get("reader_grounded_projection_authorized") is not True:
        raise ValueError("reader-grounded projection is not authorized")
    fingerprint = authorization.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != authorization_fingerprint(
        authorization
    ):
        raise ValueError("invalid or modified mechanism authorization fingerprint")
    model_id = authorization.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("mechanism authorization requires a non-empty model_id")
    artifacts = authorization.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("mechanism authorization is missing artifact hashes")
    if artifacts.get("clarity_gate") != clarity_gate_sha256:
        raise ValueError(
            "mechanism authorization clarity gate does not match the calibrator"
        )
    if (
        support_calibrator_sha256 is not None
        and artifacts.get("support_calibrator") != support_calibrator_sha256
    ):
        raise ValueError(
            "mechanism authorization support calibrator does not match the input"
        )
    return model_id


def project_row(
    row: Mapping[str, object],
    commitment_slack: float,
    polarity_margin: float,
    formal: bool,
    expected_hashes: Mapping[str, str] | None = None,
    expected_model_id: str | None = None,
) -> dict[str, object]:
    if formal:
        validate_calibration_provenance(row, expected_hashes=expected_hashes)
    if "image_id" not in row or "finding" not in row:
        raise ValueError("projection rows require image_id and finding")
    if expected_model_id is not None and row.get("model_id") != expected_model_id:
        raise ValueError(
            f"projection row model_id does not match authorization: {row.get('model_id')!r}"
        )
    decoder = row.get("decoder_probabilities")
    if not isinstance(decoder, Mapping):
        raise ValueError("projection rows require decoder_probabilities")
    if "calibrated_support_probability" not in row:
        raise ValueError("missing calibrated_support_probability")
    if "calibrated_clarity_probability" not in row:
        raise ValueError("missing calibrated_clarity_probability")

    evidence = reader_calibrated_state_distribution(
        float(row["calibrated_support_probability"]),
        float(row["calibrated_clarity_probability"]),
    )
    projected, audit = evidence_bounded_commitment_projection(
        decoder,
        evidence,
        commitment_slack=commitment_slack,
        polarity_margin=polarity_margin,
    )
    output = dict(row)
    output.update(
        {
            "projection_version": VERSION,
            "finding": normalize_term(str(row["finding"])),
            "evidence_probabilities": evidence,
            "projected_probabilities": projected,
            "prediction_state": audit["projected_state"],
            "projection_audit": audit,
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commitment-slack", type=float, default=0.0)
    parser.add_argument("--polarity-margin", type=float, default=0.0)
    parser.add_argument("--support-calibrator", type=Path)
    parser.add_argument("--clarity-calibrator", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument(
        "--mechanism-authorization",
        type=Path,
        help=(
            "passed directional/erasure/clarity authorization; required in formal mode"
        ),
    )
    parser.add_argument(
        "--plumbing-only",
        action="store_true",
        help="allow non-formal calibration provenance; output is not paper evidence",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError("projection input is empty")
    formal = not args.plumbing_only
    artifact_paths = {
        "support_calibrator_sha256": args.support_calibrator,
        "clarity_calibrator_sha256": args.clarity_calibrator,
        "calibration_manifest_sha256": args.calibration_manifest,
        "ontology_sha256": args.ontology,
    }
    if formal:
        missing_artifacts = [
            name for name, path in artifact_paths.items() if path is None
        ]
        if missing_artifacts:
            raise ValueError(
                "formal projection requires real calibration artifacts: "
                + ", ".join(missing_artifacts)
            )
        nonexistent = [
            str(path) for path in artifact_paths.values() if not path.is_file()
        ]
        if nonexistent:
            raise ValueError(f"calibration artifacts do not exist: {nonexistent}")
        expected_hashes = {
            name: sha256_file(path) for name, path in artifact_paths.items()
        }
        if (
            args.mechanism_authorization is None
            or not args.mechanism_authorization.is_file()
        ):
            raise ValueError(
                "formal projection requires --mechanism-authorization"
            )
        mechanism_authorization = json.loads(
            args.mechanism_authorization.read_text(encoding="utf-8")
        )
        if not isinstance(mechanism_authorization, Mapping):
            raise ValueError("mechanism authorization must be a JSON object")
        expected_model_id = validate_mechanism_authorization(
            mechanism_authorization,
            expected_hashes["clarity_calibrator_sha256"],
            expected_hashes["support_calibrator_sha256"],
        )
    else:
        expected_hashes = None
        mechanism_authorization = None
        expected_model_id = None
    projected = [
        project_row(
            row,
            commitment_slack=args.commitment_slack,
            polarity_margin=args.polarity_margin,
            formal=formal,
            expected_hashes=expected_hashes,
            expected_model_id=expected_model_id,
        )
        for row in rows
    ]
    identifiers = [
        (str(row["image_id"]), normalize_term(str(row["finding"]))) for row in rows
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate image/finding projection rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in projected)
    args.output.write_text(payload, encoding="utf-8")
    code_path = Path(__file__).resolve()
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_calibration_provenance": formal,
        "reader_grounded_projection_authorized": formal,
        "model_id": expected_model_id,
        "mechanism_authorization": (
            {
                "path": str(args.mechanism_authorization.resolve()),
                "sha256": sha256_file(args.mechanism_authorization),
                "fingerprint": mechanism_authorization["fingerprint"],
            }
            if formal
            else None
        ),
        "calibration_artifacts": (
            {
                name: {"path": str(path.resolve()), "sha256": expected_hashes[name]}
                for name, path in artifact_paths.items()
            }
            if formal
            else None
        ),
        "paper_evidence_admissible": False,
        "claim_ceiling": (
            "Directional admission, observational erasure, and reader-adjusted "
            "clarity measurement passed before evaluation. Paper evidence still "
            "requires causal-control, efficacy, matched-coverage, omission, and "
            "second-model gates."
        ),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "code_sha256": sha256_file(code_path),
        "n_claim_rows": len(projected),
        "commitment_slack": args.commitment_slack,
        "polarity_margin": args.polarity_margin,
        "command": shlex.join([str(code_path), *sys.argv[1:]]),
    }
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
