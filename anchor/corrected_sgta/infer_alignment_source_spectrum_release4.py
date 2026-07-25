"""Exact V4 selection: closest paired-safe alpha, without a closure gate."""

from __future__ import annotations

from corrected_sgta import infer_alignment_source_spectrum_release2 as implementation
from corrected_sgta.provenance_source_spectrum_release4 import inference_code_identity


_reviewed_protocol_fingerprint = implementation.protocol_fingerprint


def no_closure_paired_selection(*args, **kwargs):
    _ignored, original, candidates = implementation._base_select_alignment_views(*args, **kwargs)
    parsed_args = args[5] if len(args) > 5 else kwargs["args"]
    selected = []
    for source_id in sorted({item["source_id"] for item in candidates}):
        safe = [
            item for item in candidates
            if item["source_id"] == source_id and item["safe"]
            and item.get("wrong_image") is not None and item.get("wrong_safe")
        ]
        if safe:
            selected.append(min(safe, key=lambda item: (
                item["distance_after"], item["low_frequency_ratio"]
            )))
    selected.sort(key=lambda item: (-item["relative_closure"], item["source_id"]))
    return selected[: parsed_args.max_views], original, candidates


def protocol_fingerprint(config: dict) -> str:
    config["closure_gate"] = "none; choose closest paired-structure-safe alpha"
    return _reviewed_protocol_fingerprint(config)


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = "sgta-source-spectrum-preregistered-release4-v1"
    implementation.inference_code_identity = inference_code_identity
    implementation.paired_safe_select_alignment_views = no_closure_paired_selection
    implementation.protocol_fingerprint = protocol_fingerprint
    implementation.main()


if __name__ == "__main__":
    main()

