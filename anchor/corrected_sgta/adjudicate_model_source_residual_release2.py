"""Final V5 release-2 adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--analysis",required=True,type=Path); parser.add_argument("--audit",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); args=parser.parse_args()
    analysis=json.loads(args.analysis.read_text()); audit=json.loads(args.audit.read_text())
    if analysis["fingerprint"]!=audit["fingerprint"]: raise RuntimeError("analysis/audit fingerprint mismatch")
    diagnostics=analysis["domain_diagnostics"]
    checks={**analysis["preregistered_prediction_gate"]["checks"],"pubmed_closure_gt_equal_beta_target_domain_control":diagnostics["matched_relative_closure_median"]>diagnostics["wrong_control_relative_closure_median"],"strict_cache_and_pixel_identity_audit":audit["formal_matched_structure_pass"] and audit["matched"]["n"]==analysis["n"]}
    passed=all(checks.values()); report={"version":"sgta-model-source-residual-final-release2-v1","fingerprint":analysis["fingerprint"],"n":analysis["n"],"checks":checks,"pass":passed,"decision":"allow_256" if passed else "stop_mean_center_alignment","analysis":str(args.analysis.resolve()),"audit":str(args.audit.resolve())}
    args.output.parent.mkdir(parents=True,exist_ok=True); temporary=args.output.with_name(args.output.name+".tmp"); temporary.write_text(json.dumps(report,indent=2)); temporary.replace(args.output); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()

