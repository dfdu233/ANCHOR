"""Fail-closed source-dev decision gate for source preference barycenters."""

from __future__ import annotations

from typing import Any


def source_dev_gate(
    summary: dict[str, Any],
    *,
    matched_vs_shuffled_delta_pp: float | None = None,
    identity_greedy_parse_rate: float | None = None,
    lodo_greedy_parse_rate: float | None = None,
) -> dict[str, Any]:
    """Use LODO as primary evidence and fail closed on missing controls."""
    primary = summary.get("barycenter_lodo")
    if not isinstance(primary, dict):
        return {
            "status": "failed",
            "method_selection_allowed": False,
            "primary_variant": "barycenter_lodo",
            "reason": "barycenter_lodo results are absent",
        }
    micro = primary["micro"]
    per_domain = primary["per_domain"]
    net_rescues = int(micro["rescues"]) - int(micro["harms"])
    nondeclining = sum(
        float(values["delta_pp"]) >= 0.0 for values in per_domain.values()
    )
    checks: dict[str, dict[str, Any]] = {
        "complete_source_dev": {
            "value": int(micro["n"]),
            "required": 85,
            "passed": int(micro["n"]) == 85,
        },
        "net_rescues": {
            "value": net_rescues,
            "required": ">=5/85",
            "passed": net_rescues >= 5,
        },
        "nondeclining_domains": {
            "value": nondeclining,
            "domains": len(per_domain),
            "required": ">=2/3",
            "passed": len(per_domain) == 3 and nondeclining >= 2,
        },
        "matched_vs_shuffled_delta_pp": {
            "value": matched_vs_shuffled_delta_pp,
            "required": ">=3pp",
            "passed": (
                None
                if matched_vs_shuffled_delta_pp is None
                else matched_vs_shuffled_delta_pp >= 3.0
            ),
        },
        "greedy_parse_noninferiority": {
            "identity_parse_rate": identity_greedy_parse_rate,
            "lodo_parse_rate": lodo_greedy_parse_rate,
            "required": "LODO >= identity",
            "passed": (
                None
                if identity_greedy_parse_rate is None
                or lodo_greedy_parse_rate is None
                else lodo_greedy_parse_rate >= identity_greedy_parse_rate
            ),
        },
    }
    missing = [name for name, item in checks.items() if item["passed"] is None]
    known_failure = any(item["passed"] is False for item in checks.values())
    status = "failed" if known_failure else "incomplete" if missing else "passed"
    return {
        "status": status,
        "method_selection_allowed": status == "passed",
        "primary_variant": "barycenter_lodo",
        "checks": checks,
        "missing_external_evidence": missing,
        "selection_note": (
            "Pooled and full-barycenter results are controls; only the locked "
            "LODO gate may select the source-barycenter method."
        ),
    }
