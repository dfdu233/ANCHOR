"""Backend-neutral interfaces enforced by the unified evaluation contract."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from corrected_sgta.clinical_claims import ClinicalClaim

from .schema import (
    ClaimEvidence,
    EvalSample,
    EvaluationRecord,
    GenerationRecord,
    GenerationSpec,
    RetrievalRecord,
)


@runtime_checkable
class GenerationBackend(Protocol):
    def generate(
        self, sample: EvalSample, generation_contract: GenerationSpec
    ) -> GenerationRecord: ...


@runtime_checkable
class ClaimScorer(Protocol):
    def score_claim(
        self,
        sample: EvalSample,
        normalized_claim: ClinicalClaim,
        layers: Sequence[int | str],
    ) -> Sequence[ClaimEvidence]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, split_policy: str, *, sample_id: str
    ) -> RetrievalRecord: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        generation: GenerationRecord,
        reference_contract: object,
    ) -> EvaluationRecord: ...


def validate_claim_evidence(
    requested_layers: Iterable[int | str], records: Sequence[ClaimEvidence]
) -> None:
    expected = list(requested_layers)
    observed = [record.layer for record in records]
    if observed != expected:
        raise ValueError(
            f"claim scorer returned layers {observed!r}; expected {expected!r}"
        )
