"""Task-aware protocol utilities for open-ended medical report generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


VERSION = "anchor-report-protocol-v2"


@dataclass(frozen=True)
class ReportTask:
    task: str
    modality: str
    dataset: str
    clinical_metric_family: str | None


def _joined(sample: Mapping[str, object]) -> str:
    fields = (
        sample.get("dataset", ""), sample.get("domain", ""),
        sample.get("task", ""), sample.get("evaluation_group", ""),
        sample.get("question", ""), sample.get("prompt", ""),
    )
    return " ".join(str(value).lower() for value in fields)


def infer_report_task(sample: Mapping[str, object]) -> ReportTask:
    """Infer task/modality from observable metadata, never the reference."""
    text = _joined(sample)
    dataset = str(sample.get("dataset") or sample.get("evaluation_group") or sample.get("domain") or "unknown").lower()
    report_markers = ("report", "generate a report", "write a report", "radiology report", "medical report")
    task = "report_generation" if any(marker in text for marker in report_markers) else "open_vqa"
    if any(marker in text for marker in ("harvard", "fundus", "ophthalm", "retina")):
        modality = "ophthalmology"
    elif any(marker in text for marker in ("pmc-oa", "quilt", "pathology", "histopath")):
        modality = "pathology"
    elif any(marker in text for marker in ("mimic", "iu-xray", "iuxray", "chexpert", "chest x-ray", "chest radiograph", "radiology")):
        modality = "chest_radiograph"
    else:
        modality = "unknown"
    clinical = "chest_radiograph" if task == "report_generation" and modality == "chest_radiograph" else None
    return ReportTask(task, modality, dataset, clinical)


def is_report_generation_row(sample: Mapping[str, object]) -> bool:
    return infer_report_task(sample).task == "report_generation"


def _base_instruction(modality: str) -> tuple[str, str, str]:
    if modality == "ophthalmology":
        return "ophthalmologist", "fundus image", "a"
    if modality == "pathology":
        return "pathologist", "pathology image", "a"
    return "radiologist", "X-ray image", "an"


def report_prompt(sample: Mapping[str, object], mode: str = "official_zero_shot", retrieved_reports: Iterable[str] = ()) -> str:
    """Build a released-template-compatible, reference-safe report prompt."""
    dataset_prompt = str(sample.get("question") or sample.get("prompt") or "").replace("<image>", "").strip()
    if mode == "dataset":
        if not dataset_prompt:
            raise ValueError("dataset prompt requested but no question/prompt is present")
        return dataset_prompt
    task = infer_report_task(sample)
    if task.task != "report_generation":
        if not dataset_prompt:
            raise ValueError("non-report OE row has no dataset prompt")
        return dataset_prompt
    role, image_name, article = _base_instruction(task.modality)
    reports = [str(value).strip() for value in retrieved_reports if str(value).strip()]
    if mode in {"mmedrag", "official_zero_shot"}:
        return (f"You are a professional {role}. You are provided with {article} {image_name}. "
                "Please generate a report based on the image. Please only include the content of the report in your response.")
    if mode == "official_rag":
        if not reports:
            raise ValueError("official_rag requires at least one retrieved report")
        numbered = " ".join(f"{index}. {value}" for index, value in enumerate(reports, 1))
        return (f"You are a professional {role}. You are provided with {article} {image_name} and {len(reports)} reference report(s): {numbered} "
                "Please generate a report based on the image. It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, but should only be used for reference and comparison. Please only include the content of the report in your response.")
    if mode == "structured":
        section = "Ophthalmic findings" if task.modality == "ophthalmology" else "Findings"
        return (f"You are a professional {role}. You are provided with {article} {image_name}. Write a concise report with two sections:\n{section}:\nImpression:\nOnly describe findings visible in the image.")
    if mode == "impression":
        return f"You are a professional {role}. You are provided with {article} {image_name}. Provide only a concise impression of the visible findings."
    if mode == "abnormality_focused":
        return (f"You are a professional {role}. Carefully inspect the {image_name}. Describe visible abnormal findings, support devices, and relevant normal negatives. Do not call the image normal when an abnormality is visible.")
    raise ValueError(f"unknown report prompt mode: {mode}")


_FINDINGS = re.compile(r"\b(?:pneumoni(?:a|tis)|pneumothorax|cardiomegal(?:y|ic)|effusions?|edema|consolidations?|opacit(?:y|ies)|atelectasis|fractures?|congestion|masses?|nodules?|infiltrates?)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|without|absent|negative for|free of|rule out)\b", re.I)
_NORMAL = re.compile(r"\b(?:normal|unremarkable|no acute (?:cardiopulmonary )?(?:abnormalit(?:y|ies)|disease|process|finding)|lungs? (?:are|is) clear)\b", re.I)


def has_unnegated_abnormal_finding(text: str) -> bool:
    """Lightweight sanity flag; clinical claims must use clinical metrics."""
    normalized = " ".join(str(text).split())
    for match in _FINDINGS.finditer(normalized):
        prefix = re.split(r"[.;:]", normalized[max(0, match.start() - 160):match.start()])[-1]
        if not _NEGATION.search(prefix):
            return True
    return False


def is_normal_template(text: str) -> bool:
    normalized = " ".join(str(text).split())
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    return bool(len(words) <= 35 and _NORMAL.search(normalized) and not has_unnegated_abnormal_finding(normalized))
