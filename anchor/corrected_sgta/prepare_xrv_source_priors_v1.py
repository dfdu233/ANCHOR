#!/usr/bin/env python3
"""Compute source-domain pathology prevalence using XRV's default dataset policy.

The policy follows the public TorchXRayVision dataset classes: PA view only,
one first study per patient, uncertain CheXpert labels excluded, and the same
PadChest string mappings.  These source prevalences are metadata for testing a
Bayes-factor collaboration hypothesis; they are never estimated on VinDr test
labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Callable


TARGETS = (
    "cardiomegaly",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def first_per_patient(rows: list[dict[str, str]], key: Callable[[dict[str, str]], str]):
    output, seen = [], set()
    for row in rows:
        patient = key(row)
        if patient in seen:
            continue
        seen.add(patient)
        output.append(row)
    return output


def summarize(binary: dict[str, list[int | None]], source: str) -> dict[str, Any]:
    result = {"source": source, "n_rows": len(next(iter(binary.values()))), "targets": {}}
    for target, values in binary.items():
        observed = [value for value in values if value is not None]
        positives = int(sum(observed))
        result["targets"][target] = {
            "positive": positives,
            "observed": len(observed),
            "prevalence": positives / len(observed) if observed else None,
        }
    return result


def nih(path: Path) -> dict[str, Any]:
    rows = [row for row in read_gzip_csv(path) if row["View Position"] == "PA"]
    rows = first_per_patient(rows, lambda row: row["Patient ID"])
    labels = [set(row["Finding Labels"].split("|")) for row in rows]
    return summarize(
        {
            "cardiomegaly": [int("Cardiomegaly" in value) for value in labels],
            "nodule_mass": [int(bool({"Nodule", "Mass"} & value)) for value in labels],
            "pleural_effusion": [int("Effusion" in value) for value in labels],
            "pleural_thickening": [int("Pleural_Thickening" in value) for value in labels],
            "pulmonary_fibrosis": [int("Fibrosis" in value) for value in labels],
        },
        "NIH ChestX-ray14",
    )


def padchest(path: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_gzip_csv(path)
        if row["Projection"] == "PA"
        and row["Labels"].strip()
        and float(row["PatientBirth"] or 2019) < 2009
    ]
    rows = first_per_patient(rows, lambda row: row["PatientID"])
    labels = [row["Labels"].lower() for row in rows]

    def contains(value: str, terms: tuple[str, ...]) -> int:
        return int(any(term in value for term in terms))

    return summarize(
        {
            "cardiomegaly": [contains(value, ("cardiomegaly",)) for value in labels],
            "nodule_mass": [contains(value, ("nodule", "mass")) for value in labels],
            "pleural_effusion": [contains(value, ("effusion",)) for value in labels],
            "pleural_thickening": [contains(value, ("pleural thickening",)) for value in labels],
            "pulmonary_fibrosis": [contains(value, ("fibrosis",)) for value in labels],
        },
        "PadChest",
    )


def chexpert(path: Path) -> dict[str, Any]:
    rows = [row for row in read_gzip_csv(path) if row["Frontal/Lateral"] == "Frontal" and row["AP/PA"] == "PA"]

    def patient(row: dict[str, str]) -> str:
        for component in row["Path"].split("/"):
            if component.startswith("patient"):
                return component
        raise ValueError(row["Path"])

    rows = first_per_patient(rows, patient)

    def value(row: dict[str, str], column: str) -> int | None:
        raw = row[column].strip()
        if row["No Finding"].strip() == "1.0":
            return 0
        if raw in ("", "-1.0"):
            return None
        return int(float(raw) == 1.0)

    return summarize(
        {
            "cardiomegaly": [value(row, "Cardiomegaly") for row in rows],
            "nodule_mass": [value(row, "Lung Lesion") for row in rows],
            "pleural_effusion": [value(row, "Pleural Effusion") for row in rows],
            # These findings are not annotated by the CheXpert expert.
            "pleural_thickening": [None for _ in rows],
            "pulmonary_fibrosis": [None for _ in rows],
        },
        "CheXpert",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nih", type=Path, required=True)
    parser.add_argument("--padchest", type=Path, required=True)
    parser.add_argument("--chexpert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "protocol": "xrv-source-priors-v1",
        "policy": "TorchXRayVision defaults: PA only, first row per patient; CheXpert uncertain excluded",
        "command": shlex.join(sys.argv),
        "input_sha256": {
            "nih": sha256_file(args.nih),
            "pc": sha256_file(args.padchest),
            "chex": sha256_file(args.chexpert),
        },
        "domains": {
            "nih": nih(args.nih),
            "pc": padchest(args.padchest),
            "chex": chexpert(args.chexpert),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
