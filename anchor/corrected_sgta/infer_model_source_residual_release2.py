"""V5 inference entry with exact evaluation-image identity in its fingerprint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from corrected_sgta import infer_model_source_residual as implementation
from corrected_sgta.model_source_residual_provenance_release2 import model_source_residual_identity_release2
from corrected_sgta.protocol_v2 import ProtocolError, ground_truth_index, labels_for_sample, resolve_image, task_kind
from corrected_sgta.source_bank_v2 import sha256_file


_base_protocol_fingerprint = implementation.protocol_fingerprint


def canonical_rgb_identity(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    return {
        "resolved_path": str(path.resolve()), "file_sha256": sha256_file(path),
        "canonical_rgb_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "width": int(array.shape[1]), "height": int(array.shape[0]), "mode": "RGB",
    }


def protocol_fingerprint(config: dict) -> str:
    rows = json.loads(Path(config["dataset"]).read_text()); eligible = []
    for sample in rows:
        try:
            if task_kind(sample) == "open": continue
            labels_for_sample(sample); ground_truth_index(sample)
            if resolve_image(sample.get("img_name", "")) is not None: eligible.append(sample)
        except ProtocolError:
            continue
    eligible.sort(key=lambda row: hashlib.sha256(f"{config['seed']}:{row['qid']}".encode()).hexdigest())
    eligible = eligible[: int(config["max_samples"])]
    identities = []
    for sample in eligible:
        path = resolve_image(sample.get("img_name", "")); assert path is not None
        identities.append({"qid": str(sample["qid"]), "img_name": sample.get("img_name", ""), **canonical_rgb_identity(path)})
    if len(identities) != int(config["max_samples"]):
        raise RuntimeError("fixed subset has fewer eligible rows than requested")
    config["evaluation_input_identity"] = identities
    config["evaluation_input_identity_policy"] = "file and canonical RGB SHA256 before cache fingerprint"
    return _base_protocol_fingerprint(config)


def main() -> None:
    implementation.CACHE_VERSION = "sgta-model-source-visual-residual-release2-v1"
    implementation.model_source_residual_identity = model_source_residual_identity_release2
    implementation.protocol_fingerprint = protocol_fingerprint
    implementation.main()


if __name__ == "__main__": main()

