#!/usr/bin/env python3
"""Prepare relevance and image-identity controls for admitted common RAG arms."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image

from .audit_retrieval_split import read_rows
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "common-rag-causal-controls-v2"
CONTEXT_START = "Retrieved reports:\n"
CONTEXT_ENDS = ("\nQuestion:\n", "\nQuestion: ")


def context_end(prompt: str) -> str:
    matches = [delimiter for delimiter in CONTEXT_ENDS if delimiter in prompt]
    if len(matches) != 1:
        raise ValueError("prompt does not follow the frozen common-RAG schema")
    return matches[0]


def qid(row: dict, index: int) -> str:
    return str(row.get("question_id", row.get("qid", row.get("id", index))))


def context_text(prompt: str) -> str:
    if CONTEXT_START not in prompt:
        raise ValueError("prompt does not follow the frozen common-RAG schema")
    delimiter = context_end(prompt)
    return prompt.split(CONTEXT_START, 1)[1].split(delimiter, 1)[0]


def replace_context(prompt: str, donor_prompt: str) -> str:
    prefix, suffix = prompt.split(CONTEXT_START, 1)
    delimiter = context_end(prompt)
    _, question = suffix.split(delimiter, 1)
    return prefix + CONTEXT_START + context_text(donor_prompt) + delimiter + question


def image_identity_group(path: str) -> tuple[str, str]:
    """Return the strongest auditable subject identity encoded in the path.

    MIMIC-CXR paths contain ``pXX/pXXXXXXXX/sXXXXXXXX`` and therefore expose a
    pseudonymous patient identifier.  IU-Xray exposes only a study directory;
    it must be described as a different-study, not a different-patient,
    intervention.
    """

    parts = Path(path).parts
    if len(parts) >= 4 and parts[0].startswith("p") and parts[1].startswith("p"):
        return "patient", parts[1]
    if not parts:
        raise ValueError("empty image path")
    return "study", parts[0]


def minimum_cost_derangement(rows: list[dict], allowed, cost) -> list[int]:
    """Find the deterministic globally minimum-cost valid donor matching."""

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n = len(rows)
    matrix = np.full((n, n), np.inf, dtype=np.float64)
    for target in range(n):
        for donor in range(n):
            if allowed(target, donor):
                # The tiny index term gives stable tie-breaking without
                # changing the substantive cost ordering.
                matrix[target, donor] = float(cost(target, donor)) + donor * 1e-12
    if np.any(~np.isfinite(matrix).any(axis=1)):
        raise ValueError("control derangement has an unmatched row")
    finite = matrix[np.isfinite(matrix)]
    forbidden = max(1e6, float(finite.max() + 1.0) * (n + 1))
    solved = np.where(np.isfinite(matrix), matrix, forbidden)
    target_indices, donor_indices = linear_sum_assignment(solved)
    donors = [-1] * n
    for target, donor in zip(target_indices.tolist(), donor_indices.tolist()):
        if not allowed(target, donor):
            raise ValueError("no perfect control derangement exists")
        donors[target] = donor
    if any(value < 0 for value in donors):
        raise RuntimeError("incomplete control assignment")
    return donors


def shuffled_context(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    lengths = [len(context_text(str(row["question"]))) for row in rows]
    docs = [set(map(str, row.get("retrieved_doc_ids", []))) for row in rows]
    donors = minimum_cost_derangement(
        rows,
        allowed=lambda i, j: i != j and docs[i].isdisjoint(docs[j]),
        cost=lambda i, j: abs(lengths[i] - lengths[j]),
    )
    output, assignments = [], []
    for index, donor_index in enumerate(donors):
        row, donor = dict(rows[index]), rows[donor_index]
        row["question"] = replace_context(str(row["question"]), str(donor["question"]))
        row["context_condition"] = "length_matched_shuffled_retrieval"
        row["retrieved_doc_ids"] = list(donor.get("retrieved_doc_ids", []))
        row["context_donor_qid"] = qid(donor, donor_index)
        output.append(row)
        assignments.append(
            {
                "qid": qid(row, index),
                "donor_qid": qid(donor, donor_index),
                "target_context_characters": lengths[index],
                "donor_context_characters": lengths[donor_index],
                "absolute_character_delta": abs(lengths[index] - lengths[donor_index]),
                "document_overlap": sorted(docs[index] & docs[donor_index]),
            }
        )
    return output, assignments


def swapped_images(rows: list[dict], image_root: Path) -> tuple[list[dict], list[dict]]:
    dimensions = []
    identity_groups = []
    for row in rows:
        with Image.open(image_root / str(row["img_name"])) as image:
            dimensions.append(image.size)
        identity_groups.append(image_identity_group(str(row["img_name"])))

    def image_cost(i: int, j: int) -> float:
        wi, hi = dimensions[i]
        wj, hj = dimensions[j]
        return abs(math.log((wi / hi) / (wj / hj))) + 0.25 * abs(
            math.log((wi * hi) / (wj * hj))
        )

    donors = minimum_cost_derangement(
        rows,
        allowed=lambda i, j: identity_groups[i] != identity_groups[j],
        cost=image_cost,
    )
    output, assignments = [], []
    for index, donor_index in enumerate(donors):
        row, donor = dict(rows[index]), rows[donor_index]
        original_image = str(row["img_name"])
        row["img_name"] = str(donor["img_name"])
        identity_level = identity_groups[index][0]
        row["image_condition"] = f"different_{identity_level}_image"
        row["image_donor_qid"] = qid(donor, donor_index)
        output.append(row)
        assignments.append(
            {
                "qid": qid(row, index),
                "donor_qid": qid(donor, donor_index),
                "target_image": original_image,
                "donor_image": str(donor["img_name"]),
                "identity_level": identity_level,
                "target_identity_group": identity_groups[index][1],
                "donor_identity_group": identity_groups[donor_index][1],
                "target_dimensions": list(dimensions[index]),
                "donor_dimensions": list(dimensions[donor_index]),
                "dimension_cost": image_cost(index, donor_index),
            }
        )
    return output, assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--exclude-most-common-docs",
        type=int,
        default=0,
        help=(
            "Outcome-blind diagnostic eligibility rule: exclude rows containing "
            "the N most frequent retrieved document IDs before applying --limit."
        ),
    )
    args = parser.parse_args()
    rows = read_rows(args.rag_manifest)
    source_rows = len(rows)
    document_frequency = Counter(
        str(doc_id) for row in rows for doc_id in row.get("retrieved_doc_ids", [])
    )
    excluded_docs = [
        doc_id
        for doc_id, _ in document_frequency.most_common(args.exclude_most_common_docs)
    ]
    if excluded_docs:
        excluded = set(excluded_docs)
        rows = [
            row
            for row in rows
            if set(map(str, row.get("retrieved_doc_ids", []))).isdisjoint(excluded)
        ]
    eligible_rows = len(rows)
    if args.limit:
        rows = rows[: args.limit]
    shuffled, context_assignments = shuffled_context(rows)
    image_swap, image_assignments = swapped_images(rows, args.image_root)
    target_context_lengths = sorted(
        row["target_context_characters"] for row in context_assignments
    )
    donor_context_lengths = sorted(
        row["donor_context_characters"] for row in context_assignments
    )
    context_length_multiset_exact = target_context_lengths == donor_context_lengths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, values in (
        ("selected_rag", rows),
        ("shuffled_context", shuffled),
        ("image_swap", image_swap),
    ):
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
        outputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "n": len(values)}
    result = {
        "protocol_version": VERSION,
        "rag_manifest": str(args.rag_manifest.resolve()),
        "rag_manifest_sha256": sha256_file(args.rag_manifest),
        "image_root": str(args.image_root.resolve()),
        "selection": {
            "outcome_blind": True,
            "source_rows": source_rows,
            "eligible_rows": eligible_rows,
            "rule": "exclude most frequent retrieved document IDs, then retain manifest order",
            "exclude_most_common_docs": args.exclude_most_common_docs,
            "excluded_doc_ids": excluded_docs,
            "limit": args.limit,
            "selected_qids": [qid(row, index) for index, row in enumerate(rows)],
        },
        "n": len(rows),
        "outputs": outputs,
        "context_assignment": {
            "same_qid": sum(row["qid"] == row["donor_qid"] for row in context_assignments),
            "document_overlap": sum(bool(row["document_overlap"]) for row in context_assignments),
            "global_context_length_multiset_exact": context_length_multiset_exact,
            "mean_absolute_character_delta": sum(row["absolute_character_delta"] for row in context_assignments) / len(context_assignments),
            "maximum_absolute_character_delta": max(row["absolute_character_delta"] for row in context_assignments),
            "records": context_assignments,
        },
        "image_assignment": {
            "same_image": sum(row["target_image"] == row["donor_image"] for row in image_assignments),
            "same_identity_group": sum(
                row["target_identity_group"] == row["donor_identity_group"]
                for row in image_assignments
            ),
            "mean_dimension_cost": sum(row["dimension_cost"] for row in image_assignments) / len(image_assignments),
            "maximum_dimension_cost": max(row["dimension_cost"] for row in image_assignments),
            "records": image_assignments,
        },
        "passed": bool(
            rows
            and context_length_multiset_exact
            and not any(row["document_overlap"] for row in context_assignments)
            and not any(
                row["target_identity_group"] == row["donor_identity_group"]
                for row in image_assignments
            )
        ),
    }
    atomic_write_json(args.output_dir / "manifest.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"context_assignment", "image_assignment"}}, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
