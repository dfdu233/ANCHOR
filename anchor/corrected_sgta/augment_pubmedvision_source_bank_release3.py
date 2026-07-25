"""High-precision description-derived PubMedVision X-ray bank builder."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import ijson
from tqdm import tqdm

from corrected_sgta import augment_pubmedvision_source_bank as base
from corrected_sgta import augment_pubmedvision_source_bank_release2 as implementation


POSITIVE = (
    "x-ray",
    "xray",
    "radiograph",
    "radiographic",
    "plain film",
    "chest film",
    " cxr",
)
NEGATIVE = (
    "computed tomography",
    " ct ",
    "ct scan",
    "magnetic resonance",
    " mri",
    "ultrasound",
    "sonograph",
    "microscop",
    "histolog",
    "patholog",
)


def strict_metadata_candidates(metadata, members, seed):
    candidates = {}
    records = 0
    positive_records = 0
    excluded_multi_image = 0
    excluded_mixed_modality = 0
    with Path(metadata).open("rb") as handle:
        for item in tqdm(ijson.items(handle, "item"), desc="scan strict PubMedVision X-ray"):
            records += 1
            conversations = item.get("conversations") or []
            text = " ".join(str(part.get("value", "")) for part in conversations).lower()
            if not any(keyword in text for keyword in POSITIVE):
                continue
            positive_records += 1
            if any(keyword in text for keyword in NEGATIVE):
                excluded_mixed_modality += 1
                continue
            images = item.get("image") or []
            if isinstance(images, str):
                images = [images]
            if len(images) != 1:
                excluded_multi_image += 1
                continue
            resolved = base.resolve_member(str(images[0]), members)
            if resolved is None:
                continue
            canonical_name, (archive, member) = resolved
            key = (str(archive.resolve()), member)
            rank = hashlib.sha256(
                f"{seed}:{item.get('id')}:{canonical_name}:{member}".encode()
            ).hexdigest()
            candidates[key] = (rank, str(item.get("id")), archive, member)
    ranked = sorted(candidates.values(), key=lambda value: value[0])
    return ranked, {
        "metadata_records": records,
        "description_xray_records": positive_records,
        "excluded_mixed_modality_records": excluded_mixed_modality,
        "excluded_multi_image_records": excluded_multi_image,
        "local_unique_xray_candidates": len(ranked),
        "selection_rule": (
            "single-image record; description contains an X-ray/radiograph/CXR keyword; "
            "description contains no CT/MRI/ultrasound/microscopy/pathology keyword"
        ),
    }


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    base.metadata_candidates = strict_metadata_candidates
    implementation.main()
    manifest_path = Path(argument("--output-dir")) / "source_bank.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest.get("entries", []):
        if entry.get("source_id") == "pubmedvision_xray_formal":
            entry["modality_selection"] = (
                "high-precision rule over official PubMedVision GPT description metadata; "
                "single-image records only; mixed-modality descriptions excluded"
            )
    manifest["provenance_version"] = "source-image-content-v4-pubmed-description"
    manifest.setdefault("notes", {})["pubmedvision_metadata_quality"] = (
        "The dataset modality field is inconsistent and was not used. Selection is "
        "deterministic from the pinned description metadata with an explicit rule."
    )
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
