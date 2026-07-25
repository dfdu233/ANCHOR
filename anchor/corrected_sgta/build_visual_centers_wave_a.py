"""Final content-verifying visual-center builder for Wave A."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import build_visual_centers_v2 as implementation
from corrected_sgta.provenance_wave_a import code_identity, model_identity
from corrected_sgta.source_bank_v3 import load_descriptor_image, verify_source_artifacts


def argument(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def main() -> None:
    implementation.code_identity = code_identity
    implementation.model_identity = model_identity
    implementation.load_descriptor_image = load_descriptor_image
    implementation.verify_source_artifacts = verify_source_artifacts
    implementation.main()
    output = Path(argument("--output"))
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["version"] = "sgta-visual-source-centers-wave-a-v1"
    metadata["batch_size"] = int(argument("--batch-size"))
    metadata["code_identity"] = code_identity(Path(__file__).resolve().parents[1])
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(metadata_path)


if __name__ == "__main__":
    main()
