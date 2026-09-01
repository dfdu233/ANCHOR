"""Strict, provenance-first evaluation primitives for medical VLMs."""

from .schema import (
    EvalSample,
    EvaluationTrack,
    GenerationSpec,
    MethodSpec,
    ModelSpec,
    PredictionRecord,
    TaskKind,
    TaskSpec,
)

__all__ = [
    "EvalSample",
    "EvaluationTrack",
    "GenerationSpec",
    "MethodSpec",
    "ModelSpec",
    "PredictionRecord",
    "TaskKind",
    "TaskSpec",
]
