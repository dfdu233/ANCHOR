#!/usr/bin/env python3
"""AAAI baseline/method registry for aligned result collection.

The registry keeps method comparison protocol-aware:

* ``ce_logits`` is the main cross-model protocol. It reuses the same finite
  label cache for Baseline, FedDG/TTA/SGTA, LAME/LATA, and SCA-T.
* ``official_generative`` is a separate LLaVA-Med mitigation protocol for
  architecture-specific decoding baselines. Its numbers must not be plotted as
  visual peers of CE-logit results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/Hulu-Med/MedUniEval")

CE_TASKS = ("context", "cxr_vishal", "knowledge_ce", "mm_vishal")
MODELS = ("hulu", "llava")

CE_METHODS = {
    "Baseline": {"source": "summary", "key": "baseline", "scope": "all finite-label rows"},
    "FedDG": {"source": "summary", "key": "feddg_center", "scope": "all finite-label rows"},
    "Gamma-TTA": {"source": "summary", "key": "tta_entropy", "scope": "all finite-label rows"},
    "SGTA": {"source": "summary", "key": "sgta", "scope": "all finite-label rows"},
    "LAME": {"source": "summary", "key": "lame", "scope": "Yes/No rows only"},
    "LATA": {"source": "summary", "key": "lata", "scope": "Yes/No rows only"},
    "SCA-T TIM": {"source": "scat", "key": "scat_tim", "scope": "Yes/No transductive"},
    "SCA-T TIM-KL": {"source": "scat", "key": "scat_tim_kl", "scope": "Yes/No transductive"},
}

OFFICIAL_GENERATIVE_METHODS = (
    "greedy",
    "DoLa",
    "PAI",
    "opera",
    "avisc",
    "m3id",
    "VCD",
    "damro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "corrected_runs/aaai_aligned_baseline_manifest_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "version": "aaai-aligned-baseline-v1",
        "principle": "Compare methods only within the same protocol, model set, prompt/interface, sample set, and budget.",
        "protocols": {
            "ce_logits": {
                "role": "main_paper_protocol",
                "models": list(MODELS),
                "tasks": list(CE_TASKS),
                "cache_dir": str(ROOT / "corrected_runs/full_v52"),
                "runner": "bash corrected_sgta/run_full_ce.sh && bash corrected_sgta/run_scat.sh",
                "methods": CE_METHODS,
                "reuse_allowed": True,
            },
            "official_generative": {
                "role": "supplementary_architecture_specific_decoding_protocol",
                "models": ["llava"],
                "tasks": ["context", "knowledge_ce", "cxr_vishal", "mm_vishal"],
                "runner": "python corrected_runs/aaai_medheval_mitigation_full_v1/run_llava_medheval_official_queue.py",
                "methods": list(OFFICIAL_GENERATIVE_METHODS),
                "important_caveat": "Do not compare directly with ce_logits baseline; report parse rate and invalid-as-error accuracy.",
                "reuse_allowed": False,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
