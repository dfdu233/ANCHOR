"""Unified claim semantics for the Missing Third State experiments.

This module intentionally evaluates structured claims.  A lexical parser or an
LLM judge may propose the structure, but neither is allowed to silently define
the reference state used by the metrics below.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Iterable, Mapping, Sequence


VERSION = "missing-third-state-claims-v8"
STATES = ("supported", "refuted", "undetermined", "unobservable")
REFERENCE_RELEVANCE = ("required", "optional", "out_of_scope")
FORMAL_REFERENCE_SOURCES = {
    "vindr_reader_votes",
    "physician_review",
    "structured_dataset",
}


@dataclass(frozen=True)
class ClinicalClaim:
    finding: str
    polarity: str = "present"
    uncertainty: str = "definite"
    anatomy: str | None = None
    attributes: tuple[str, ...] = ()
    provenance: str = "image_grounded"

    def __post_init__(self) -> None:
        if self.polarity not in {"present", "absent"}:
            raise ValueError(f"invalid polarity: {self.polarity}")
        if self.uncertainty not in {"definite", "uncertain", "unobservable"}:
            raise ValueError(f"invalid uncertainty: {self.uncertainty}")
        if self.provenance not in {"image_grounded", "knowledge", "context"}:
            raise ValueError(f"invalid provenance: {self.provenance}")
        object.__setattr__(self, "finding", normalize_term(self.finding))
        if self.anatomy:
            object.__setattr__(self, "anatomy", normalize_term(self.anatomy))
        object.__setattr__(
            self,
            "attributes",
            tuple(sorted({normalize_term(value) for value in self.attributes if value})),
        )

    @property
    def key(self) -> tuple[str, str | None, tuple[str, ...]]:
        return self.finding, self.anatomy, self.attributes

    @property
    def state(self) -> str:
        if self.provenance != "image_grounded" or self.uncertainty == "unobservable":
            return "unobservable"
        if self.uncertainty == "uncertain":
            return "undetermined"
        return "supported" if self.polarity == "present" else "refuted"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["attributes"] = list(self.attributes)
        value["state"] = self.state
        return value


def normalize_term(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("/", " ").split())


def reader_state(positive_votes: int, reader_count: int) -> str:
    """Map votes to the deliberately conservative three-state reference.

    Only unanimous endpoints are definite.  Any internal vote fraction is the
    clinically important disagreement band and therefore ``undetermined``.
    """

    if reader_count <= 0 or not 0 <= positive_votes <= reader_count:
        raise ValueError("votes must satisfy 0 <= positive_votes <= reader_count")
    if positive_votes == reader_count:
        return "supported"
    if positive_votes == 0:
        return "refuted"
    return "undetermined"


def state_probability(state: str, epsilon: float = 0.05) -> float:
    if state == "supported":
        return 1.0 - epsilon
    if state == "refuted":
        return epsilon
    if state == "undetermined":
        return 0.5
    raise ValueError(f"state has no image-grounded probability: {state}")


def signed_commitment(state: str) -> float:
    return {"supported": 1.0, "refuted": -1.0, "undetermined": 0.0}[state]


def sigmoid(value: float) -> float:
    """Numerically stable conversion from a claim logit to support probability."""

    if not math.isfinite(value):
        raise ValueError("claim score must be finite")
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def support_commitment_decomposition(
    reader_support: float,
    predicted_support: float,
    prediction_state: str,
) -> dict[str, float]:
    """Exactly split total claim mismatch into visual and language terms.

    With signed reader support ``R``, signed predicted visual support
    ``R_hat``, and expressed language commitment ``K``:
    ``K - R = (K - R_hat) + (R_hat - R)``.
    """

    if not 0.0 <= reader_support <= 1.0:
        raise ValueError("reader_support must lie in [0,1]")
    if not 0.0 <= predicted_support <= 1.0:
        raise ValueError("predicted_support must lie in [0,1]")
    if prediction_state not in {"supported", "refuted", "undetermined"}:
        raise ValueError("decomposition requires an image-grounded three-state claim")
    reference = 2.0 * reader_support - 1.0
    predicted = 2.0 * predicted_support - 1.0
    commitment = signed_commitment(prediction_state)
    transfer = commitment - predicted
    visual = predicted - reference
    total = commitment - reference
    return {
        "signed_total_gap": total,
        "signed_language_transfer_gap": transfer,
        "signed_visual_support_gap": visual,
        "decomposition_residual": total - transfer - visual,
    }


def tristate_logits(evidence: float, tau: float) -> dict[str, float]:
    """Legacy one-dimensional three-state construction.

    This is retained for reproducibility.  It constrains the two-dimensional
    three-state simplex to ``commitment = abs(polarity) - tau`` and therefore
    must not be described as a general representation of visual evidence.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    return {
        "supported": float(evidence),
        "refuted": float(-evidence),
        "undetermined": float(tau - abs(evidence)),
    }


def epistemic_coordinates(
    logits: Mapping[str, float],
    null_logits: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return the two independent log-contrast coordinates of a claim.

    ``polarity`` measures support versus refutation.  ``commitment`` measures
    definite (support or refute) evidence versus an undetermined answer.  A
    visual coordinate is obtained by subtracting the same coordinate under a
    locked null.  The transform is invariant to a shared additive logit shift.
    """

    missing = set(STATES[:3]) - set(logits)
    if missing:
        raise ValueError(f"missing three-state logits: {sorted(missing)}")

    def coordinates(values: Mapping[str, float]) -> tuple[float, float]:
        support = float(values["supported"])
        refute = float(values["refuted"])
        unknown = float(values["undetermined"])
        polarity = 0.5 * (support - refute)
        commitment = 0.5 * (support + refute) - unknown
        return polarity, commitment

    polarity, commitment = coordinates(logits)
    if null_logits is not None:
        null_missing = set(STATES[:3]) - set(null_logits)
        if null_missing:
            raise ValueError(f"missing null three-state logits: {sorted(null_missing)}")
        null_polarity, null_commitment = coordinates(null_logits)
        polarity -= null_polarity
        commitment -= null_commitment
    return {"polarity": polarity, "commitment": commitment}


def simplex_logits(polarity: float, commitment: float) -> dict[str, float]:
    """Invert :func:`epistemic_coordinates` up to a shared logit constant."""

    return {
        "supported": float(commitment + polarity),
        "refuted": float(commitment - polarity),
        "undetermined": 0.0,
    }


def paired_clinical_selectivity(
    anchor_polarity: float,
    same_state_polarity: float,
    opposite_state_polarity: float,
    anchor_reader_support: float,
    opposite_reader_support: float,
) -> dict[str, float]:
    """Separate clinically relevant sensitivity from visual nuisance drift.

    The same-state image changes patient/image-specific appearance while
    preserving the reference claim state.  The opposite-state image changes
    that state.  A positive gap means claim polarity reacts in the clinically
    correct direction more than it reacts to a state-preserving image swap.
    """

    values = (
        anchor_polarity,
        same_state_polarity,
        opposite_state_polarity,
        anchor_reader_support,
        opposite_reader_support,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("clinical-selectivity inputs must be finite")
    if not 0.0 <= anchor_reader_support <= 1.0:
        raise ValueError("anchor reader support must lie in [0, 1]")
    if not 0.0 <= opposite_reader_support <= 1.0:
        raise ValueError("opposite reader support must lie in [0, 1]")
    reference_delta = anchor_reader_support - opposite_reader_support
    if reference_delta == 0.0:
        raise ValueError("opposite-state reference must change reader support")
    direction = 1.0 if reference_delta > 0.0 else -1.0
    clinical_change = direction * (anchor_polarity - opposite_state_polarity)
    absolute_clinical_change = abs(anchor_polarity - opposite_state_polarity)
    nuisance_change = abs(anchor_polarity - same_state_polarity)
    return {
        "signed_clinical_change": clinical_change,
        "absolute_clinical_change": absolute_clinical_change,
        "absolute_nuisance_change": nuisance_change,
        "clinical_selectivity_gap": clinical_change - nuisance_change,
        "unsigned_selectivity_gap": absolute_clinical_change - nuisance_change,
        "directionally_aligned": float(clinical_change > 0.0),
        "unsigned_responsive": float(absolute_clinical_change > nuisance_change),
        "misdirected_responsive": float(
            absolute_clinical_change > nuisance_change and clinical_change < 0.0
        ),
        "reference_separation": abs(reference_delta),
    }


def epistemic_state(
    polarity: float,
    commitment: float,
    polarity_tau: float,
    commitment_tau: float,
) -> str:
    """Map the claim plane to support, refutation, conflict, or ignorance.

    Conflict and ignorance are distinct evidence states even when both should
    be verbalized conservatively as an uncertain clinical claim.
    """

    if polarity_tau < 0 or commitment_tau < 0:
        raise ValueError("epistemic thresholds must be non-negative")
    if commitment < commitment_tau:
        return "ignorance"
    if abs(polarity) < polarity_tau:
        return "conflict"
    return "supported" if polarity > 0 else "refuted"


def softmax_states(logits: Mapping[str, float]) -> dict[str, float]:
    missing = set(STATES[:3]) - set(logits)
    if missing:
        raise ValueError(f"missing three-state logits: {sorted(missing)}")
    maximum = max(float(logits[state]) for state in STATES[:3])
    values = {state: math.exp(float(logits[state]) - maximum) for state in STATES[:3]}
    total = sum(values.values())
    return {state: value / total for state, value in values.items()}


def reader_calibrated_state_distribution(
    support_probability: float,
    clarity_probability: float,
) -> dict[str, float]:
    """Compose independent polarity and reader-clarity channels.

    ``support_probability`` answers which polarity is supported, conditional
    on making a definite claim. ``clarity_probability`` estimates whether
    independent readers would agree on a definite endpoint at all. Keeping
    these channels separate avoids reconstructing the missing third state from
    one signed score::

        P(S) = clarity * support
        P(R) = clarity * (1 - support)
        P(U) = 1 - clarity

    Both inputs must be calibrated on a locked development split. This helper
    is only the representation contract; it does not make an uncalibrated
    probe clinically valid.
    """

    values = (float(support_probability), float(clarity_probability))
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("support and clarity probabilities must lie in [0,1]")
    support, clarity = values
    return {
        "supported": clarity * support,
        "refuted": clarity * (1.0 - support),
        "undetermined": 1.0 - clarity,
    }


def _normalized_state_distribution(
    probabilities: Mapping[str, float],
    name: str,
) -> dict[str, float]:
    missing = set(STATES[:3]) - set(probabilities)
    if missing:
        raise ValueError(f"missing {name} probabilities: {sorted(missing)}")
    values = {state: float(probabilities[state]) for state in STATES[:3]}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError(f"{name} probabilities must have positive mass")
    return {state: value / total for state, value in values.items()}


def evidence_bounded_commitment_projection(
    decoder_probabilities: Mapping[str, float],
    evidence_probabilities: Mapping[str, float],
    commitment_slack: float = 0.0,
    polarity_margin: float = 0.0,
) -> tuple[dict[str, float], dict[str, object]]:
    """Project language commitment into a reader-calibrated evidence envelope.

    The projection changes the decoder distribution as little as possible in
    forward KL while enforcing two constraints: definite language mass cannot
    exceed calibrated evidence clarity plus ``commitment_slack``; and, when
    calibrated evidence has a decisive polarity, language may not retain the
    opposite polarity.

    Capping definite mass preserves the decoder's support/refute odds. If its
    polarity contradicts decisive evidence, the closest feasible point is the
    support/refute boundary; the audit marks that boundary as undetermined.
    Thus this primitive can hedge an unsupported commitment, but never invents
    the opposite positive or negative claim. Omission recovery remains a
    separate ontology-scanning decision evaluated at matched coverage.
    """

    if not math.isfinite(commitment_slack) or not 0.0 <= commitment_slack <= 1.0:
        raise ValueError("commitment_slack must lie in [0,1]")
    if not math.isfinite(polarity_margin) or not 0.0 <= polarity_margin <= 1.0:
        raise ValueError("polarity_margin must lie in [0,1]")
    decoder = _normalized_state_distribution(decoder_probabilities, "decoder")
    evidence = _normalized_state_distribution(evidence_probabilities, "evidence")
    if any(decoder[state] <= 0.0 for state in STATES[:3]):
        raise ValueError(
            "decoder probabilities must be strictly positive for KL projection"
        )

    evidence_definite = evidence["supported"] + evidence["refuted"]
    maximum_definite = min(1.0, evidence_definite + commitment_slack)
    decoder_definite = decoder["supported"] + decoder["refuted"]
    evidence_polarity = (
        (evidence["supported"] - evidence["refuted"]) / evidence_definite
        if evidence_definite > 0.0
        else 0.0
    )
    positive_constraint_violated = (
        evidence_polarity > polarity_margin
        and decoder["supported"] < decoder["refuted"]
    )
    negative_constraint_violated = (
        evidence_polarity < -polarity_margin
        and decoder["refuted"] < decoder["supported"]
    )
    polarity_clipped = positive_constraint_violated or negative_constraint_violated

    if polarity_clipped:
        # Forward-KL I-projection onto the violated polarity half-space lies on
        # S=R. If that point exceeds the clarity cap, both constraints bind.
        geometric_mean = math.sqrt(decoder["supported"] * decoder["refuted"])
        denominator = 2.0 * geometric_mean + decoder["undetermined"]
        boundary_state_mass = geometric_mean / denominator
        boundary_definite = 2.0 * boundary_state_mass
        projected_definite = min(boundary_definite, maximum_definite)
        projected = {
            "supported": projected_definite / 2.0,
            "refuted": projected_definite / 2.0,
            "undetermined": 1.0 - projected_definite,
        }
    elif decoder_definite > maximum_definite:
        projected_definite = maximum_definite
        scale = maximum_definite / decoder_definite
        projected = {
            "supported": decoder["supported"] * scale,
            "refuted": decoder["refuted"] * scale,
            "undetermined": 1.0 - maximum_definite,
        }
    else:
        projected_definite = decoder_definite
        projected = dict(decoder)

    commitment_capped = projected_definite < decoder_definite - 1e-12

    tolerance = 1e-12
    if (
        abs(projected["supported"] - projected["refuted"]) <= tolerance
        or projected["undetermined"]
        >= max(projected["supported"], projected["refuted"]) - tolerance
    ):
        projected_state = "undetermined"
    else:
        projected_state = (
            "supported"
            if projected["supported"] > projected["refuted"]
            else "refuted"
        )

    epsilon = 1e-12
    forward_kl = sum(
        value * math.log(max(value, epsilon) / max(decoder[state], epsilon))
        for state, value in projected.items()
        if value > 0.0
    )
    audit = {
        "decoder_probabilities": decoder,
        "evidence_probabilities": evidence,
        "projected_probabilities": projected,
        "decoder_definite_mass": decoder_definite,
        "evidence_definite_mass": evidence_definite,
        "maximum_definite_mass": maximum_definite,
        "projected_definite_mass": projected_definite,
        "evidence_conditional_polarity": evidence_polarity,
        "commitment_capped": commitment_capped,
        "polarity_clipped_to_boundary": polarity_clipped,
        "projected_state": projected_state,
        "forward_kl_from_decoder": forward_kl,
        "commitment_slack": commitment_slack,
        "polarity_margin": polarity_margin,
    }
    return projected, audit


def bounded_state(evidence: float, tau: float) -> str:
    probabilities = softmax_states(tristate_logits(evidence, tau))
    return max(STATES[:3], key=probabilities.get)  # type: ignore[arg-type]


def commitment_bounded_claims(
    draft: Sequence[ClinicalClaim],
    evidence_by_finding: Mapping[str, float],
    ontology: Iterable[str],
    tau: float,
    add_threshold: float,
    required_findings: Iterable[str] = (),
) -> tuple[list[ClinicalClaim], list[dict[str, object]]]:
    """Bound draft commitment and add required, strongly supported omissions.

    Weak-evidence draft claims are retained with uncertainty rather than being
    deleted.  Claims outside the image-grounded ontology are retained and
    marked unobservable so that knowledge/context errors are evaluated apart.
    A fixed ontology defines what can be audited, not what every answer must
    mention.  Omitted findings are therefore added only when the task contract
    explicitly marks them as required (for example, abnormality listing).
    """

    normalized_evidence = {
        normalize_term(finding): float(value)
        for finding, value in evidence_by_finding.items()
    }
    normalized_ontology = tuple(dict.fromkeys(normalize_term(x) for x in ontology))
    normalized_required = {
        normalize_term(finding) for finding in required_findings
    }
    unknown_required = normalized_required - set(normalized_ontology)
    if unknown_required:
        raise ValueError(
            f"required findings outside ontology: {sorted(unknown_required)}"
        )
    output: list[ClinicalClaim] = []
    audit: list[dict[str, object]] = []
    seen: set[str] = set()
    for claim in draft:
        seen.add(claim.finding)
        if claim.provenance != "image_grounded" or claim.finding not in normalized_ontology:
            revised = replace(claim, uncertainty="unobservable")
            reason = "not_image_verifiable"
        else:
            evidence = normalized_evidence.get(claim.finding)
            if evidence is None:
                revised = replace(claim, uncertainty="unobservable")
                reason = "evidence_missing"
            else:
                state = bounded_state(evidence, tau)
                if state == "supported":
                    revised = replace(claim, polarity="present", uncertainty="definite")
                elif state == "refuted":
                    revised = replace(claim, polarity="absent", uncertainty="definite")
                else:
                    revised = replace(claim, uncertainty="uncertain")
                reason = f"bounded_to_{state}"
        output.append(revised)
        audit.append(
            {
                "finding": claim.finding,
                "action": reason,
                "draft_state": claim.state,
                "final_state": revised.state,
                "evidence": normalized_evidence.get(claim.finding),
            }
        )

    for finding in normalized_ontology:
        evidence = normalized_evidence.get(finding)
        if finding in seen or evidence is None or evidence < add_threshold:
            continue
        if finding not in normalized_required:
            audit.append(
                {
                    "finding": finding,
                    "action": "not_added_without_required_relevance",
                    "draft_state": "absent_from_draft",
                    "final_state": "unmentioned",
                    "evidence": evidence,
                }
            )
            continue
        added = ClinicalClaim(finding=finding)
        output.append(added)
        audit.append(
            {
                "finding": finding,
                "action": "added_omitted_high_support",
                "draft_state": "absent_from_draft",
                "final_state": added.state,
                "evidence": evidence,
            }
        )
    return output, audit


def polarity_preserving_commitment_claims(
    draft: Sequence[ClinicalClaim],
    agreement_probability_by_finding: Mapping[str, float],
    clear_threshold: float,
) -> tuple[list[ClinicalClaim], list[dict[str, object]]]:
    """Realize certainty without changing claim content or polarity.

    This is the only OE rewrite authorized by the reader-agreement mechanism.
    It may hedge or unhedge an existing image-grounded claim, but cannot add,
    delete, negate, localize, or otherwise alter it. Missing gate scores leave
    the draft unchanged and remain visible in the audit.
    """

    if not 0.0 <= clear_threshold <= 1.0:
        raise ValueError("clear_threshold must lie in [0,1]")
    probabilities = {
        normalize_term(finding): float(value)
        for finding, value in agreement_probability_by_finding.items()
    }
    for finding, probability in probabilities.items():
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid agreement probability for {finding}: {probability}")

    output = []
    audit = []
    for claim in draft:
        probability = probabilities.get(claim.finding)
        if claim.provenance != "image_grounded" or claim.uncertainty == "unobservable":
            revised = claim
            action = "outside_image_grounded_gate"
        elif probability is None:
            revised = claim
            action = "missing_gate_score_no_change"
        else:
            uncertainty = "definite" if probability >= clear_threshold else "uncertain"
            revised = replace(claim, uncertainty=uncertainty)
            action = "retained_definite" if uncertainty == "definite" else "hedged"
            if claim.uncertainty == "uncertain" and uncertainty == "definite":
                action = "unhedged"
        if revised.key != claim.key or revised.polarity != claim.polarity:
            raise RuntimeError("commitment gate violated claim or polarity invariance")
        output.append(revised)
        audit.append(
            {
                "finding": claim.finding,
                "action": action,
                "draft_state": claim.state,
                "final_state": revised.state,
                "agreement_probability": probability,
                "clear_threshold": clear_threshold,
            }
        )
    if len(output) != len(draft):
        raise RuntimeError("commitment gate changed claim coverage")
    return output, audit


def evidence_conserving_claim_exchange(
    draft: Sequence[ClinicalClaim],
    directed_support_by_finding: Mapping[str, float],
    ontology: Iterable[str],
    minimum_exchange_margin: float = 0.0,
    maximum_exchanges: int | None = None,
) -> tuple[list[ClinicalClaim], list[dict[str, object]]]:
    """Exchange weak draft findings for stronger omitted findings at fixed budget.

    This primitive is restricted to finding-level positive abnormality listing.
    It neither changes the number of in-ontology positive claims nor treats a
    hedge as deletion.  Each authorized exchange replaces one drafted finding
    with one previously omitted finding whose majority-directed early support
    is larger by at least ``minimum_exchange_margin``.  Knowledge/context,
    negative, and out-of-ontology claims are untouched.

    The function is method plumbing, not evidence that the score is clinically
    valid. Formal use requires the DCR and commitment-tetrad gates to pass.
    """

    if minimum_exchange_margin < 0.0 or not math.isfinite(minimum_exchange_margin):
        raise ValueError("minimum_exchange_margin must be finite and non-negative")
    if maximum_exchanges is not None and maximum_exchanges < 0:
        raise ValueError("maximum_exchanges must be non-negative")
    normalized_ontology = tuple(dict.fromkeys(normalize_term(value) for value in ontology))
    ontology_set = set(normalized_ontology)
    scores = {
        normalize_term(finding): float(score)
        for finding, score in directed_support_by_finding.items()
    }
    invalid = {
        finding: score for finding, score in scores.items() if not math.isfinite(score)
    }
    if invalid:
        raise ValueError(f"directed support scores must be finite: {invalid}")

    eligible: dict[str, tuple[int, ClinicalClaim]] = {}
    for index, claim in enumerate(draft):
        if (
            claim.provenance == "image_grounded"
            and claim.polarity == "present"
            and claim.finding in ontology_set
        ):
            if claim.finding in eligible:
                raise ValueError(
                    f"finding-level exchange requires unique draft findings: {claim.finding}"
                )
            eligible[claim.finding] = (index, claim)

    weak_draft = sorted(
        (
            (scores[finding], finding, index, claim)
            for finding, (index, claim) in eligible.items()
            if finding in scores
        ),
        key=lambda value: (value[0], value[1]),
    )
    strong_omitted = sorted(
        (
            (scores[finding], finding)
            for finding in normalized_ontology
            if finding not in eligible and finding in scores
        ),
        key=lambda value: (-value[0], value[1]),
    )
    limit = min(len(weak_draft), len(strong_omitted))
    if maximum_exchanges is not None:
        limit = min(limit, maximum_exchanges)

    replacements: dict[int, tuple[ClinicalClaim, float, str, float]] = {}
    for old, new in zip(weak_draft[:limit], strong_omitted[:limit]):
        old_score, old_finding, index, old_claim = old
        new_score, new_finding = new
        if new_score - old_score < minimum_exchange_margin:
            continue
        replacement = ClinicalClaim(
            finding=new_finding,
            polarity="present",
            uncertainty=old_claim.uncertainty,
            provenance="image_grounded",
        )
        replacements[index] = (replacement, new_score, old_finding, old_score)

    output = []
    audit = []
    for index, claim in enumerate(draft):
        if index not in replacements:
            output.append(claim)
            audit.append(
                {
                    "action": "retained",
                    "draft_finding": claim.finding,
                    "final_finding": claim.finding,
                    "draft_score": scores.get(claim.finding),
                    "final_score": scores.get(claim.finding),
                }
            )
            continue
        replacement, new_score, old_finding, old_score = replacements[index]
        output.append(replacement)
        audit.append(
            {
                "action": "exchanged",
                "draft_finding": old_finding,
                "final_finding": replacement.finding,
                "draft_score": old_score,
                "final_score": new_score,
                "score_gain": new_score - old_score,
            }
        )

    def in_scope_positive_count(claims: Sequence[ClinicalClaim]) -> int:
        return sum(
            claim.provenance == "image_grounded"
            and claim.polarity == "present"
            and claim.finding in ontology_set
            for claim in claims
        )

    if len(output) != len(draft):
        raise RuntimeError("claim exchange changed total claim count")
    if in_scope_positive_count(output) != in_scope_positive_count(draft):
        raise RuntimeError("claim exchange changed the positive claim budget")
    return output, audit


def oe_prediction_axes(row: Mapping[str, object]) -> dict[str, str]:
    """Recover independent content-polarity and linguistic-certainty axes.

    A hedged positive claim (for example, ``possible effusion``) is not a
    polarity-free third class.  It retains positive clinical content while
    expressing uncertain commitment.  Formal OE rows must therefore make the
    polarity of every emitted ``undetermined`` claim explicit; otherwise a
    method could hide a fabricated positive merely by hedging it.
    """

    emitted = bool(row["emitted"])
    state = str(row["prediction_state"])
    if state not in STATES:
        raise ValueError(f"invalid prediction_state: {state}")
    if not emitted:
        return {"polarity": "unmentioned", "uncertainty": "unmentioned"}

    explicit_polarity = row.get("prediction_polarity")
    explicit_uncertainty = row.get("prediction_uncertainty")
    if state == "supported":
        polarity, uncertainty = "present", "definite"
    elif state == "refuted":
        polarity, uncertainty = "absent", "definite"
    elif state == "unobservable":
        polarity, uncertainty = "unobservable", "unobservable"
    else:
        if explicit_polarity is None:
            raise ValueError(
                "an emitted undetermined OE claim requires prediction_polarity; "
                "hedging must not erase positive or negative claim content"
            )
        polarity = str(explicit_polarity)
        uncertainty = str(explicit_uncertainty or "uncertain")

    allowed_polarities = {"present", "absent", "conflict", "unobservable"}
    allowed_uncertainties = {"definite", "uncertain", "mixed", "unobservable"}
    if polarity not in allowed_polarities:
        raise ValueError(f"invalid prediction_polarity: {polarity}")
    if uncertainty not in allowed_uncertainties:
        raise ValueError(f"invalid prediction_uncertainty: {uncertainty}")
    if explicit_polarity is not None and str(explicit_polarity) != polarity:
        raise ValueError("prediction_polarity contradicts prediction_state")
    if explicit_uncertainty is not None and str(explicit_uncertainty) != uncertainty:
        raise ValueError("prediction_uncertainty contradicts prediction_state")
    return {"polarity": polarity, "uncertainty": uncertainty}


def evaluate_claim_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Evaluate a fixed candidate universe under one frozen claim contract.

    Required fields are ``reader_support`` in [0,1] and ``prediction_state``.
    The candidate universe must include unmentioned ontology findings; callers
    should encode those as ``undetermined`` or an explicit negative prediction,
    never drop them from the denominator.
    """

    grounded = [row for row in rows if row.get("prediction_state") != "unobservable"]
    if not grounded:
        raise ValueError("no image-grounded claim rows")
    counters = {
        "unanimous_positive": 0,
        "unanimous_negative": 0,
        "reader_disagreement": 0,
        "definite_predictions": 0,
        "positive_assertions": 0,
        "fabricated_positive": 0,
        "unsupported_positive": 0,
        "omitted_unanimous_positive": 0,
        "false_negative": 0,
        "overcommitted_disagreement": 0,
    }
    brier = []
    nll = []
    coverage_risk = []
    signed_gaps = []
    absolute_gaps = []
    overcommitment_gaps = []
    transfer_gaps = []
    visual_support_gaps = []
    tri_correct = []
    for row in grounded:
        support = float(row["reader_support"])
        state = str(row["prediction_state"])
        if not 0.0 <= support <= 1.0:
            raise ValueError(f"reader_support outside [0,1]: {support}")
        if state not in STATES[:3]:
            raise ValueError(f"invalid prediction_state: {state}")
        reference = (
            "supported" if support == 1.0 else "refuted" if support == 0.0 else "undetermined"
        )
        counters["unanimous_positive"] += support == 1.0
        counters["unanimous_negative"] += support == 0.0
        counters["reader_disagreement"] += 0.0 < support < 1.0
        definite = state != "undetermined"
        counters["definite_predictions"] += definite
        counters["positive_assertions"] += state == "supported"
        counters["fabricated_positive"] += state == "supported" and support == 0.0
        counters["unsupported_positive"] += state == "supported" and support < 0.5
        counters["omitted_unanimous_positive"] += support == 1.0 and state != "supported"
        counters["false_negative"] += support == 1.0 and state == "refuted"
        counters["overcommitted_disagreement"] += 0.0 < support < 1.0 and definite
        probability = state_probability(state)
        brier.append((probability - support) ** 2)
        nll.append(
            -support * math.log(probability)
            - (1.0 - support) * math.log(1.0 - probability)
        )
        if definite:
            coverage_risk.append(1.0 - support if state == "supported" else support)
        signed_support = 2.0 * support - 1.0
        commitment = signed_commitment(state)
        signed_gaps.append(commitment - signed_support)
        absolute_gaps.append(abs(commitment - signed_support))
        overcommitment_gaps.append(
            max(0.0, abs(commitment) - abs(signed_support))
        )
        decomposition = support_commitment_decomposition(
            support, probability, state
        )
        transfer_gaps.append(decomposition["signed_language_transfer_gap"])
        visual_support_gaps.append(decomposition["signed_visual_support_gap"])
        tri_correct.append(state == reference)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    n = len(grounded)
    return {
        "version": VERSION,
        "n_image_grounded": n,
        "n_unobservable": len(rows) - n,
        "counts": counters,
        "tri_state_accuracy": sum(tri_correct) / n,
        "coverage": counters["definite_predictions"] / n,
        "matched_claim_coverage_risk": (
            sum(coverage_risk) / len(coverage_risk) if coverage_risk else None
        ),
        "positive_claim_precision": (
            1.0
            - ratio(counters["unsupported_positive"], counters["positive_assertions"])
            if counters["positive_assertions"]
            else None
        ),
        "positive_claim_hallucination_rate": ratio(
            counters["unsupported_positive"], counters["positive_assertions"]
        ),
        "fabricated_claim_rate": ratio(
            counters["fabricated_positive"], counters["positive_assertions"]
        ),
        "unanimous_positive_omission_rate": ratio(
            counters["omitted_unanimous_positive"], counters["unanimous_positive"]
        ),
        "unanimous_positive_false_negative_rate": ratio(
            counters["false_negative"], counters["unanimous_positive"]
        ),
        "disagreement_overcommitment_rate": ratio(
            counters["overcommitted_disagreement"], counters["reader_disagreement"]
        ),
        "reader_distribution_brier": sum(brier) / n,
        "reader_distribution_nll": sum(nll) / n,
        "mean_signed_support_commitment_gap": sum(signed_gaps) / n,
        "mean_absolute_support_commitment_gap": sum(absolute_gaps) / n,
        "mean_overcommitment_strength_gap": sum(overcommitment_gaps) / n,
        "mean_signed_language_transfer_gap": sum(transfer_gaps) / n,
        "mean_absolute_language_transfer_gap": sum(map(abs, transfer_gaps)) / n,
        "mean_signed_visual_support_gap": sum(visual_support_gaps) / n,
        "mean_absolute_visual_support_gap": sum(map(abs, visual_support_gaps)) / n,
    }


def evaluate_oe_claim_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Evaluate one method on a fixed OE/report claim universe.

    Required fields are ``claim_id``, ``reader_support``, ``emitted``,
    ``prediction_state``, and ``assertion_score``. Emitted uncertain claims
    additionally require ``prediction_polarity``. Non-emitted claims remain in
    the input so omissions cannot disappear from the denominator.
    """

    if not rows:
        raise ValueError("OE claim universe is empty")
    identifiers = [str(row["claim_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate claim_id within one method")
    grounded = [
        row
        for row in rows
        if str(
            row.get(
                "reference_observability",
                "unobservable"
                if row.get("prediction_state") == "unobservable"
                else "image_grounded",
            )
        )
        == "image_grounded"
    ]
    if not grounded:
        raise ValueError("OE claim universe has no image-grounded rows")
    relevance = {
        str(row["claim_id"]): str(row.get("reference_relevance", "required"))
        for row in grounded
    }
    invalid_relevance = {
        value for value in relevance.values() if value not in REFERENCE_RELEVANCE
    }
    if invalid_relevance:
        raise ValueError(f"invalid reference_relevance: {sorted(invalid_relevance)}")
    emitted = [row for row in grounded if bool(row["emitted"])]
    axes = {str(row["claim_id"]): oe_prediction_axes(row) for row in grounded}
    support_probabilities = []
    for row in grounded:
        score = float(row["assertion_score"])
        probability = float(row.get("support_probability", sigmoid(score)))
        if not math.isfinite(score) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "every fixed-universe claim requires a finite assertion_score "
                "and support_probability in [0,1]"
            )
        support_probabilities.append(probability)
    definite = [
        row
        for row in emitted
        if axes[str(row["claim_id"])]["uncertainty"] == "definite"
    ]
    positive = [
        row
        for row in emitted
        if axes[str(row["claim_id"])]["polarity"] in {"present", "conflict"}
    ]
    definite_positive = [
        row
        for row in positive
        if axes[str(row["claim_id"])]["uncertainty"] == "definite"
    ]
    hedged_positive = [
        row
        for row in positive
        if axes[str(row["claim_id"])]["uncertainty"] in {"uncertain", "mixed"}
    ]
    hedged = [
        row
        for row in emitted
        if axes[str(row["claim_id"])]["uncertainty"] in {"uncertain", "mixed"}
    ]
    contradictory = [
        row
        for row in emitted
        if axes[str(row["claim_id"])]["polarity"] == "conflict"
    ]
    unobservable_predictions = [
        row for row in emitted if str(row["prediction_state"]) == "unobservable"
    ]
    unanimous_positive = [row for row in grounded if float(row["reader_support"]) == 1.0]
    required_unanimous_positive = [
        row
        for row in unanimous_positive
        if relevance[str(row["claim_id"])] == "required"
    ]
    recovered_positive = [
        row
        for row in unanimous_positive
        if bool(row["emitted"])
        and axes[str(row["claim_id"])]["polarity"] in {"present", "conflict"}
    ]
    recovered_required_positive = [
        row
        for row in required_unanimous_positive
        if bool(row["emitted"])
        and axes[str(row["claim_id"])]["polarity"] in {"present", "conflict"}
    ]
    optional_supported = [
        row
        for row in unanimous_positive
        if relevance[str(row["claim_id"])] == "optional"
    ]
    mentioned_optional_supported = [
        row
        for row in optional_supported
        if bool(row["emitted"])
        and axes[str(row["claim_id"])]["polarity"] in {"present", "conflict"}
    ]
    out_of_scope_emitted = [
        row
        for row in emitted
        if relevance[str(row["claim_id"])] == "out_of_scope"
    ]
    risks = []
    for row in definite:
        support = float(row["reader_support"])
        risks.append(
            1.0 - support
            if str(row["prediction_state"]) == "supported"
            else support
        )
    unsupported_positive = [row for row in positive if float(row["reader_support"]) < 0.5]
    fabricated_positive = [row for row in positive if float(row["reader_support"]) == 0.0]
    legacy_positive = [
        row for row in emitted if str(row["prediction_state"]) == "supported"
    ]
    legacy_unsupported_positive = [
        row for row in legacy_positive if float(row["reader_support"]) < 0.5
    ]
    legacy_fabricated_positive = [
        row for row in legacy_positive if float(row["reader_support"]) == 0.0
    ]
    fabricated_hedged_positive = [
        row for row in hedged_positive if float(row["reader_support"]) == 0.0
    ]
    negative = [
        row
        for row in emitted
        if axes[str(row["claim_id"])]["polarity"] in {"absent", "conflict"}
    ]
    reader_disagreement = [
        row for row in grounded if 0.0 < float(row["reader_support"]) < 1.0
    ]
    overcommitted_disagreement = [
        row
        for row in reader_disagreement
        if bool(row["emitted"])
        and str(row["prediction_state"]) in {"supported", "refuted"}
    ]
    unanimous = [
        row for row in grounded if float(row["reader_support"]) in {0.0, 1.0}
    ]
    false_negative = [
        row
        for row in unanimous_positive
        if bool(row["emitted"]) and str(row["prediction_state"]) == "refuted"
    ]
    brier = []
    nll = []
    signed_gaps = []
    absolute_gaps = []
    overcommitment_gaps = []
    transfer_gaps = []
    visual_support_gaps = []
    decomposition_residuals = []
    tristate_correct = []
    clear_correct = []
    eps = 1e-7
    for row, probability in zip(grounded, support_probabilities):
        support = float(row["reader_support"])
        state = str(row["prediction_state"])
        if not 0.0 <= support <= 1.0 or state not in STATES:
            raise ValueError("invalid reader support or prediction state")
        # Non-emission and prediction-side `unobservable` both mean no language
        # commitment; neither may remove a reference-grounded claim.
        effective_state = (
            state
            if bool(row["emitted"]) and state in {"supported", "refuted", "undetermined"}
            else "undetermined"
        )
        reference_state = (
            "supported" if support == 1.0 else
            "refuted" if support == 0.0 else
            "undetermined"
        )
        brier.append((probability - support) ** 2)
        bounded_probability = max(eps, min(1.0 - eps, probability))
        nll.append(
            -support * math.log(bounded_probability)
            - (1.0 - support) * math.log(1.0 - bounded_probability)
        )
        commitment = signed_commitment(effective_state)
        signed_support = 2.0 * support - 1.0
        signed_gaps.append(commitment - signed_support)
        absolute_gaps.append(abs(commitment - signed_support))
        overcommitment_gaps.append(
            max(0.0, abs(commitment) - abs(signed_support))
        )
        decomposition = support_commitment_decomposition(
            support, probability, effective_state
        )
        transfer_gaps.append(decomposition["signed_language_transfer_gap"])
        visual_support_gaps.append(decomposition["signed_visual_support_gap"])
        decomposition_residuals.append(decomposition["decomposition_residual"])
        tristate_correct.append(effective_state == reference_state)
        if support in {0.0, 1.0}:
            clear_correct.append(effective_state == reference_state)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "version": VERSION,
        "claim_universe_size": len(grounded),
        "unobservable_count": len(rows) - len(grounded),
        "emitted_claim_count": len(emitted),
        "output_claim_coverage": len(emitted) / len(grounded),
        "definite_emitted_count": len(definite),
        "definite_output_coverage": len(definite) / len(grounded),
        "positive_assertion_count": len(positive),
        "positive_content_claim_count": len(positive),
        "definite_positive_assertion_count": len(definite_positive),
        "hedged_positive_claim_count": len(hedged_positive),
        "fabricated_hedged_positive_count": len(fabricated_hedged_positive),
        "legacy_third_state_positive_count": len(legacy_positive),
        "polarity_erased_hedged_positive_count": len(hedged_positive),
        "hedged_claim_count": len(hedged),
        "negative_claim_count": len(negative),
        "contradictory_claim_count": len(contradictory),
        "unobservable_prediction_count": len(unobservable_predictions),
        "reference_relevance_counts": {
            value: sum(item == value for item in relevance.values())
            for value in REFERENCE_RELEVANCE
        },
        "out_of_scope_emitted_count": len(out_of_scope_emitted),
        "out_of_scope_emission_rate": ratio(
            len(out_of_scope_emitted),
            sum(value == "out_of_scope" for value in relevance.values()),
        ),
        "hedged_emitted_rate": ratio(len(hedged), len(emitted)),
        "negative_emitted_rate": ratio(len(negative), len(emitted)),
        "empty_or_refusal": len(emitted) == 0,
        "uniform_negative": bool(emitted) and all(
            axes[str(row["claim_id"])]["polarity"] == "absent"
            for row in emitted
        ),
        "uniform_uncertain": bool(emitted) and all(
            axes[str(row["claim_id"])]["uncertainty"] == "uncertain"
            for row in emitted
        ),
        "mean_expected_definite_claim_risk": (
            sum(risks) / len(risks) if risks else None
        ),
        "soft_positive_grounding_precision": (
            sum(float(row["reader_support"]) for row in positive) / len(positive)
            if positive
            else None
        ),
        "positive_claim_hallucination_rate": ratio(
            len(unsupported_positive), len(positive)
        ),
        "fabricated_claim_rate": ratio(len(fabricated_positive), len(positive)),
        "legacy_third_state_positive_claim_hallucination_rate": ratio(
            len(legacy_unsupported_positive), len(legacy_positive)
        ),
        "legacy_third_state_fabricated_claim_rate": ratio(
            len(legacy_fabricated_positive), len(legacy_positive)
        ),
        "unanimous_positive_recall": ratio(
            len(recovered_positive), len(unanimous_positive)
        ),
        "unanimous_positive_omission_rate": (
            1.0 - len(recovered_positive) / len(unanimous_positive)
            if unanimous_positive
            else None
        ),
        "required_unanimous_positive_recall": ratio(
            len(recovered_required_positive), len(required_unanimous_positive)
        ),
        "required_unanimous_positive_omission_rate": (
            1.0 - len(recovered_required_positive) / len(required_unanimous_positive)
            if required_unanimous_positive
            else None
        ),
        "optional_unanimous_positive_mention_rate": ratio(
            len(mentioned_optional_supported), len(optional_supported)
        ),
        "unanimous_positive_false_negative_rate": ratio(
            len(false_negative), len(unanimous_positive)
        ),
        "disagreement_overcommitment_rate": ratio(
            len(overcommitted_disagreement), len(reader_disagreement)
        ),
        "reader_distribution_brier": mean(brier),
        "reader_distribution_nll": mean(nll),
        "mean_signed_support_commitment_gap": mean(signed_gaps),
        "mean_absolute_support_commitment_gap": mean(absolute_gaps),
        "mean_overcommitment_strength_gap": mean(overcommitment_gaps),
        "mean_signed_language_transfer_gap": mean(transfer_gaps),
        "mean_absolute_language_transfer_gap": mean(map(abs, transfer_gaps)),
        "mean_signed_visual_support_gap": mean(visual_support_gaps),
        "mean_absolute_visual_support_gap": mean(map(abs, visual_support_gaps)),
        "maximum_absolute_decomposition_residual": max(
            map(abs, decomposition_residuals)
        ),
        "tri_state_accuracy": mean(tristate_correct),
        "clear_case_accuracy": mean(clear_correct) if unanimous else None,
    }


def evaluate_oe_methods_matched_coverage(
    rows: Sequence[Mapping[str, object]],
    baseline_method: str,
    target_count: int | None = None,
    maximum_natural_coverage_drop: float = 0.01,
    require_reference_provenance: bool = False,
) -> dict[str, object]:
    """Compare OE methods on exactly the same claim universe and output count.

    The default matched count is the smallest natural emitted count.  This
    makes a terse method unable to claim a hallucination win without forcing
    every comparator to the same claim count, while its natural omission and
    coverage remain visible.  A zero-count comparison is invalid, not perfect.
    """

    reference_audit = (
        validate_oe_reference_provenance(rows)
        if require_reference_provenance
        else {
            "formal_reference_provenance_required": False,
            "status": "plumbing_only",
        }
    )
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        method = str(row["method"])
        grouped.setdefault(method, []).append(row)
    if baseline_method not in grouped:
        raise ValueError(f"baseline method not found: {baseline_method}")
    if len(grouped) < 2:
        raise ValueError("matched coverage requires at least two methods")

    canonical: dict[str, tuple[float, str, str]] | None = None
    for method, method_rows in grouped.items():
        universe = {
            str(row["claim_id"]): (
                float(row["reader_support"]),
                str(
                    row.get(
                        "reference_observability",
                        "unobservable"
                        if row.get("prediction_state") == "unobservable"
                        else "image_grounded",
                    )
                ),
                str(row.get("reference_relevance", "required")),
            )
            for row in method_rows
        }
        if len(universe) != len(method_rows):
            raise ValueError(f"{method}: duplicate claim IDs")
        if canonical is None:
            canonical = universe
        elif universe != canonical:
            raise ValueError(f"{method}: claim universe/reference mismatch")

    natural = {
        method: evaluate_oe_claim_rows(method_rows)
        for method, method_rows in grouped.items()
    }
    emitted_counts = {
        method: int(metrics["emitted_claim_count"])
        for method, metrics in natural.items()
    }
    positive_content_counts = {
        method: sum(
            bool(row["emitted"])
            and oe_prediction_axes(row)["polarity"] in {"present", "conflict"}
            for row in method_rows
        )
        for method, method_rows in grouped.items()
    }
    matched_count = (
        min(positive_content_counts.values())
        if target_count is None
        else target_count
    )
    matched_all_count = min(emitted_counts.values())
    grounded_count = int(natural[baseline_method]["claim_universe_size"])
    if not 0 <= matched_count <= grounded_count:
        raise ValueError("target_count must lie within the grounded claim universe")
    if any(count < matched_count for count in positive_content_counts.values()):
        raise ValueError(
            "a method cannot supply the requested matched positive-content claim count"
        )

    matched = {}
    if matched_count > 0:
        for method, method_rows in grouped.items():
            candidates = [
                row
                for row in method_rows
                if bool(row["emitted"])
                and oe_prediction_axes(row)["polarity"] in {"present", "conflict"}
            ]
            chosen_ids = {
                str(row["claim_id"])
                for row in sorted(
                    candidates,
                    key=lambda row: (-float(row["assertion_score"]), str(row["claim_id"])),
                )[:matched_count]
            }
            selected_rows = [
                dict(row, emitted=str(row["claim_id"]) in chosen_ids)
                for row in method_rows
            ]
            matched[method] = evaluate_oe_claim_rows(selected_rows)

    baseline_coverage = float(natural[baseline_method]["output_claim_coverage"])
    coverage_guard = {
        method: (
            baseline_coverage - float(metrics["output_claim_coverage"])
            <= maximum_natural_coverage_drop
        )
        for method, metrics in natural.items()
    }
    omission_guard = {}
    baseline_omission = natural[baseline_method][
        "required_unanimous_positive_omission_rate"
    ]
    for method, metrics in natural.items():
        value = metrics["required_unanimous_positive_omission_rate"]
        omission_guard[method] = bool(
            value is not None
            and baseline_omission is not None
            and float(value) <= float(baseline_omission)
        )
    baseline_clear = natural[baseline_method]["clear_case_accuracy"]
    clear_case_guard = {
        method: bool(
            metrics["clear_case_accuracy"] is not None
            and baseline_clear is not None
            and float(metrics["clear_case_accuracy"]) >= float(baseline_clear)
        )
        for method, metrics in natural.items()
    }
    baseline_tristate = float(natural[baseline_method]["tri_state_accuracy"])
    tristate_guard = {
        method: float(metrics["tri_state_accuracy"]) >= baseline_tristate
        for method, metrics in natural.items()
    }
    return {
        "version": VERSION,
        "baseline_method": baseline_method,
        "methods": sorted(grouped),
        "fixed_claim_universe": True,
        "reference_audit": reference_audit,
        "natural": natural,
        "matched_claim_basis": (
            "emitted positive-content abnormality claims, including hedged positives"
        ),
        "matched_claim_count": matched_count,
        "matched_coverage": matched_count / grounded_count if grounded_count else None,
        "matched_valid": matched_count > 0,
        "matched": matched,
        "matched_all_emitted_claim_count_diagnostic": matched_all_count,
        "natural_positive_content_counts": positive_content_counts,
        "coverage_guard_maximum_drop": maximum_natural_coverage_drop,
        "coverage_guard_pass": coverage_guard,
        "omission_nonincrease_pass": omission_guard,
        "clear_case_non_degradation_pass": clear_case_guard,
        "tristate_non_degradation_pass": tristate_guard,
        "anti_cheat": {
            method: {
                "empty_or_refusal": bool(metrics["empty_or_refusal"]),
                "uniform_negative": bool(metrics["uniform_negative"]),
                "uniform_uncertain": bool(metrics["uniform_uncertain"]),
                "natural_coverage_drop_vs_baseline": baseline_coverage
                - float(metrics["output_claim_coverage"]),
            }
            for method, metrics in natural.items()
        },
        "claim_ceiling": (
            "point estimates only; statistical significance and physician-verified "
            "claim structure are still required for a method claim"
        ),
    }


def validate_oe_reference_provenance(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reject automatic or inconsistent reference truth in formal OE runs."""

    if not rows:
        raise ValueError("cannot validate reference provenance for empty rows")
    sources: set[str] = set()
    contracts: set[str] = set()
    canonical: dict[str, tuple[object, ...]] = {}
    for row in rows:
        claim_id = str(row["claim_id"])
        source = str(row.get("reference_source", ""))
        contract = str(row.get("reference_contract_version", ""))
        if source not in FORMAL_REFERENCE_SOURCES:
            raise ValueError(
                f"{claim_id}: invalid formal reference_source {source!r}; "
                "automatic labelers and LLM judges cannot define truth"
            )
        if not contract:
            raise ValueError(f"{claim_id}: missing reference_contract_version")
        if contract != VERSION:
            raise ValueError(
                f"{claim_id}: stale reference_contract_version {contract!r}; "
                f"formal evaluation requires {VERSION!r}"
            )
        observability = str(row.get("reference_observability", ""))
        if observability not in {"image_grounded", "unobservable"}:
            raise ValueError(f"{claim_id}: invalid reference_observability {observability!r}")
        if "reference_relevance" not in row:
            raise ValueError(f"{claim_id}: missing reference_relevance")
        relevance = str(row["reference_relevance"])
        if relevance not in REFERENCE_RELEVANCE:
            raise ValueError(f"{claim_id}: invalid reference_relevance {relevance!r}")
        sources.add(source)
        contracts.add(contract)
        if source == "vindr_reader_votes":
            positive_votes = int(row["positive_votes"])
            reader_count = int(row["reader_count"])
            if reader_count != 3 or positive_votes not in range(4):
                raise ValueError(f"{claim_id}: VinDr training reference must be 0..3 of 3")
            raw_reader_votes = row.get("reader_votes")
            if not isinstance(raw_reader_votes, list) or len(raw_reader_votes) != reader_count:
                raise ValueError(
                    f"{claim_id}: formal VinDr reference requires three reader-level votes"
                )
            reader_pairs: list[tuple[str, int]] = []
            for item in raw_reader_votes:
                if not isinstance(item, Mapping):
                    raise ValueError(f"{claim_id}: invalid reader_votes entry")
                rad_id = str(item.get("rad_id", ""))
                vote = int(item.get("vote", -1))
                if not rad_id or vote not in {0, 1}:
                    raise ValueError(f"{claim_id}: invalid reader-level vote")
                reader_pairs.append((rad_id, vote))
            reader_pairs.sort()
            if len({rad_id for rad_id, _ in reader_pairs}) != reader_count:
                raise ValueError(f"{claim_id}: duplicate VinDr rad_ID")
            if sum(vote for _, vote in reader_pairs) != positive_votes:
                raise ValueError(
                    f"{claim_id}: positive_votes disagree with reader-level votes"
                )
            if "reader_ids" in row and list(row["reader_ids"]) != [
                rad_id for rad_id, _ in reader_pairs
            ]:
                raise ValueError(f"{claim_id}: reader_ids disagree with reader_votes")
            if not math.isclose(
                float(row["reader_support"]),
                positive_votes / reader_count,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{claim_id}: reader_support disagrees with raw votes")
            detail: tuple[object, ...] = (
                positive_votes,
                reader_count,
                tuple(reader_pairs),
            )
        elif source == "physician_review":
            physician_count = int(row.get("physician_count", 0))
            if physician_count <= 0:
                raise ValueError(f"{claim_id}: physician review requires physician_count")
            detail = (physician_count, str(row.get("adjudication", "")))
        else:
            label_field = str(row.get("structured_label_field", ""))
            if not label_field:
                raise ValueError(f"{claim_id}: structured reference requires label field")
            detail = (label_field,)
        fingerprint = (
            float(row["reader_support"]),
            observability,
            relevance,
            source,
            contract,
            *detail,
        )
        if claim_id in canonical and canonical[claim_id] != fingerprint:
            raise ValueError(f"{claim_id}: formal reference differs across methods")
        canonical[claim_id] = fingerprint
    return {
        "formal_reference_provenance_required": True,
        "status": "valid",
        "sources": sorted(sources),
        "contract_versions": sorted(contracts),
        "claim_count": len(canonical),
        "automatic_truth_allowed": False,
    }


def claims_to_fixed_oe_rows(
    report_id: str,
    method: str,
    claims: Sequence[ClinicalClaim],
    reader_support_by_finding: Mapping[str, float],
    assertion_score_by_finding: Mapping[str, float],
    reference_metadata_by_finding: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Project generated claims onto a fixed ontology without hiding omissions.

    Out-of-ontology claims are returned for adjudication and make the audit
    incomplete.  Conflicting duplicate mentions are represented as
    ``undetermined`` rather than resolved in the method's favor.
    """

    supports = {
        normalize_term(finding): float(value)
        for finding, value in reader_support_by_finding.items()
    }
    scores = {
        normalize_term(finding): float(value)
        for finding, value in assertion_score_by_finding.items()
    }
    metadata = {
        normalize_term(finding): dict(value)
        for finding, value in reference_metadata_by_finding.items()
    }
    if not supports or set(scores) != set(supports) or set(metadata) != set(supports):
        raise ValueError("support, assertion-score, and reference metadata ontologies must match")
    for finding, support in supports.items():
        if not 0.0 <= support <= 1.0 or not math.isfinite(scores[finding]):
            raise ValueError(f"invalid support/assertion score for {finding}")

    grouped: dict[str, list[ClinicalClaim]] = {finding: [] for finding in supports}
    out_of_ontology: list[dict[str, object]] = []
    for claim in claims:
        if claim.finding not in grouped:
            out_of_ontology.append(claim.to_dict())
        else:
            grouped[claim.finding].append(claim)

    rows: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for finding in sorted(supports):
        mentions = grouped[finding]
        states = {claim.state for claim in mentions}
        polarities = {claim.polarity for claim in mentions}
        uncertainties = {claim.uncertainty for claim in mentions}
        if not mentions:
            prediction_state = "undetermined"
        elif len(states) == 1:
            prediction_state = next(iter(states))
        else:
            prediction_state = "undetermined"
            conflicts.append(
                {
                    "finding": finding,
                    "mention_states": sorted(states),
                    "resolution": "undetermined",
                }
            )
        if not mentions:
            prediction_polarity = "unmentioned"
            prediction_uncertainty = "unmentioned"
        else:
            prediction_polarity = (
                next(iter(polarities)) if len(polarities) == 1 else "conflict"
            )
            prediction_uncertainty = (
                next(iter(uncertainties)) if len(uncertainties) == 1 else "mixed"
            )
        reference = metadata[finding]
        if "reference_observability" not in reference:
            raise ValueError(f"{finding}: reference metadata lacks observability")
        if "reference_relevance" not in reference:
            raise ValueError(f"{finding}: reference metadata lacks relevance")
        if str(reference["reference_relevance"]) not in REFERENCE_RELEVANCE:
            raise ValueError(f"{finding}: invalid reference relevance")
        rows.append(
            {
                "method": method,
                "report_id": str(report_id),
                "claim_id": f"{report_id}:{finding}",
                "finding": finding,
                "reader_support": supports[finding],
                "emitted": bool(mentions),
                "prediction_state": prediction_state,
                "prediction_polarity": prediction_polarity,
                "prediction_uncertainty": prediction_uncertainty,
                "assertion_score": scores[finding],
                **reference,
            }
        )
    audit = {
        "version": VERSION,
        "fixed_ontology_size": len(supports),
        "emitted_mapped_finding_count": sum(bool(value) for value in grouped.values()),
        "out_of_ontology_claims": out_of_ontology,
        "duplicate_state_conflicts": conflicts,
        "adjudication_complete": not out_of_ontology,
    }
    return rows, audit
