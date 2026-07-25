"""Surface-primary frozen analysis for the V5 residual."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from corrected_sgta import analyze_alignment_v2 as implementation
from corrected_sgta.cache import iter_successes


def argument(name: str) -> str: return sys.argv[sys.argv.index(name) + 1]
def decoded_state(value): return ("invalid",) if value is None else ("label", int(value))


def main() -> None:
    implementation.main(); output = Path(argument("--output")); cache = Path(argument("--cache"))
    report = json.loads(output.read_text()); metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    rows = list(iter_successes(cache, metadata["fingerprint"])); disagreements=[]; base_correct=[]; oracle_correct=[]
    for row in rows:
        decoded=row.get("style_decoded_prediction"); matched=[i for i,r in enumerate(row.get("style_roles",[])) if r=="matched"]
        gt=int(row["gt_index"])
        if decoded is None or not matched:
            disagreements.append(False); base_correct.append(False); oracle_correct.append(False); continue
        original=decoded[0]; views=[decoded[i] for i in matched]
        disagreements.append(any(decoded_state(v)!=decoded_state(original) for v in views))
        base_correct.append(original is not None and int(original)==gt)
        oracle_correct.append((original is not None and int(original)==gt) or any(v is not None and int(v)==gt for v in views))
    decoded_base=float(np.mean(base_correct)); decoded_oracle=float(np.mean(oracle_correct))
    surface_disagreement=report["domain_diagnostics"]["matched_cross_view_prediction_disagreement_rate"]
    surface_headroom=report["matched_style_oracle_headroom_diagnostic_only"]
    fixed="matched_laplacian_anchor_l1"; flips=report["flips_vs_original"][fixed]
    checks={
        "surface_disagreement_and_oracle_headroom_ge_2pp": surface_disagreement>0 and surface_headroom>=0.02,
        "fixed_laplacian_l1_rescues_ge_harmful": flips["rescues"]>=flips["harmful"],
    }
    report["version"]="sgta-model-source-residual-analysis-release2-v1"
    report["decoded_diagnostic_only"]={
        "original_accuracy_invalid_as_error":decoded_base,
        "source_style_oracle_accuracy_invalid_as_error":decoded_oracle,
        "source_style_oracle_headroom":decoded_oracle-decoded_base,
        "disagreement_rate":float(np.mean(disagreements)),
    }
    report["legacy_accuracy_selected_gate_ignored"]=report.pop("gate")
    report["preregistered_prediction_gate"]={"fixed_method":fixed,"primary_channel":"surface_logits","checks":checks,"pass":all(checks.values())}
    report["method_note"]="Progression uses only the surface-logit channel that defines fixed Laplacian lambda=1; decoded output is diagnostic."
    temporary=output.with_name(output.name+".tmp"); temporary.write_text(json.dumps(report,indent=2)); temporary.replace(output)


if __name__ == "__main__": main()

