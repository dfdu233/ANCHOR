"""Typed common contract, intentionally independent of any model backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .hashing import sha256_json


class EvaluationTrack(str, Enum):
    PAPER_NATIVE = "paper_native"
    COMMON_PROTOCOL = "common_protocol"


class TaskKind(str, Enum):
    CE_DECISION = "ce_decision"
    CE_GENERATION = "ce_generation"
    OE_VQA = "oe_vqa"
    REPORT_GENERATION = "report_generation"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: TaskKind
    dataset_path: str
    image_root: str
    evaluator: str
    prompt_template: str
    sample_id_field: str = "question_id"
    image_field: str = "image"
    question_field: str = "question"
    reference_field: str = "answer"
    cluster_field: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        required = (self.name, self.dataset_path, self.evaluator, self.prompt_template)
        if any(not item for item in required):
            raise ValueError("task name, dataset_path, evaluator, and prompt_template are required")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    adapter: str
    conversation_template: str
    revision: str | None = None
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    implementation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    supported_models: tuple[str, ...] = ()
    supported_tasks: tuple[TaskKind, ...] = ()


@dataclass(frozen=True)
class GenerationSpec:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    do_sample: bool = False
    seed: int = 42
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.num_beams < 1:
            raise ValueError("invalid generation setting")
        if self.do_sample and self.temperature == 0:
            raise ValueError("sampling requires a positive temperature")


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    image_paths: tuple[str, ...]
    question: str
    reference: Any
    cluster_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.image_paths or not self.question:
            raise ValueError("sample_id, image_paths, and question are required")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class PredictionRecord:
    protocol_id: str
    run_fingerprint: str
    sample_fingerprint: str
    sample_id: str
    cluster_id: str
    prediction: str
    prompt: str
    status: str = "ok"
    error: str | None = None
    generation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "error"}:
            raise ValueError("status must be ok or error")
        if self.status == "error" and not self.error:
            raise ValueError("error records must retain the error")


@dataclass(frozen=True)
class GenerationRecord:
    """Backend-neutral raw generation artifact used by CE, OE, and reports."""

    sample_id: str
    model_id: str
    raw_text: str
    token_ids: tuple[int, ...]
    stop_reason: str
    generation_config: dict[str, Any]
    artifact_hashes: dict[str, str]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.model_id:
            raise ValueError("sample_id and model_id are required")
        if self.stop_reason not in {"eos", "template", "eos_or_template", "length", "error"}:
            raise ValueError(f"invalid stop_reason: {self.stop_reason}")


@dataclass(frozen=True)
class ClaimEvidence:
    """Two-coordinate claim evidence without imposing a decoding policy."""

    sample_id: str
    finding: str
    layer: int | str
    logits: dict[str, float]
    polarity: float
    clarity: float
    null_controls: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {"supported", "refuted", "undetermined"}
        if set(self.logits) != required:
            raise ValueError(f"claim logits must contain exactly {sorted(required)}")


@dataclass(frozen=True)
class RetrievalRecord:
    sample_id: str
    query: str
    split_policy: str
    index_version: str
    documents: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.query or not self.split_policy:
            raise ValueError("retrieval sample, query, and split policy are required")
        for document in self.documents:
            if not {"doc_id", "rank", "score", "sha256"} <= set(document):
                raise ValueError("retrieval documents require doc_id/rank/score/sha256")


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    evaluator_version: str
    decision: str | None = None
    claim_matches: tuple[dict[str, Any], ...] = ()
    coverage: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    grader_provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.evaluator_version:
            raise ValueError("sample_id and evaluator_version are required")


def as_canonical_dict(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    for key, item in list(payload.items()):
        if isinstance(item, Enum):
            payload[key] = item.value
        elif isinstance(item, tuple):
            payload[key] = [member.value if isinstance(member, Enum) else member for member in item]
    return payload
