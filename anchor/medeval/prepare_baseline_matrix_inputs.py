#!/usr/bin/env python3
"""Freeze the seven paper-baseline input manifests without changing splits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "corrected_runs/unified_eval/inputs/baseline_matrix_v1"
CHOICE_RE = re.compile(r"(?:^|[,;]\s*)([A-Z])\s*[.:)]\s*(.*?)(?=(?:[,;]\s*[A-Z]\s*[.:)])|$)", re.I)


def normal(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def parsed_choice_labels(choices: object) -> dict[str, str]:
    text = str(choices or "")
    if re.fullmatch(
        r"\s*[A-Z](?:\s*[,;/]\s*[A-Z])+(?:\s*[,;/]\s*)?", text, re.I
    ):
        labels = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text, re.I)
        return {label.upper(): label.upper() for label in labels}
    matches = list(re.finditer(r"(?<!\w)([A-Z])\s*[.:),]", text))
    parsed = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parsed[match.group(1).upper()] = text[match.end():end].strip(" ,;")
    return parsed


def normalize_question_type(source_type: object, answer: object, choices: object) -> str:
    answer_normal = normal(answer)
    source = str(source_type or "").strip().lower()
    if source == "binary":
        if answer_normal in {"maybe", "uncertain", "undetermined", "unclear", "equivocal"}:
            return "ternary"
        # MedHEval's official evaluator assigns the task from source
        # question_type, not from the surface form of the reference answer.
        return "binary"
    if source in {"multi-choice", "multiple-choice", "multichoice", "mcq", "choice"}:
        if len(parsed_choice_labels(choices)) < 2:
            raise ValueError(f"source multi-choice row has fewer than two parsed options: {choices!r}")
        return "choice"
    raise ValueError(f"unsupported source question_type: {source_type!r}")


def patient_or_image_cluster(image_name: str) -> str:
    parts = Path(image_name).parts
    # MIMIC paths contain both a shard directory (for example ``p19``) and
    # the actual patient directory (``p19454978``).  The longest numeric
    # component is the patient, not the first matching component.
    patient_parts = [
        part[1:] for part in parts if part.startswith("p") and part[1:].isdigit()
    ]
    patient = max(patient_parts, key=len, default=None)
    if patient:
        return patient
    study = next((part for part in parts if part.startswith("CXR") and "_IM-" in part), None)
    return study or image_name


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected a JSON list at {path}")
    return rows


def freeze(
    *,
    name: str,
    source: Path,
    image_root: Path,
    task: str,
    expected: int,
    prefix: str,
    passthrough: bool = False,
) -> dict[str, Any]:
    rows = load(source)
    if len(rows) != expected:
        raise ValueError(f"{name}: expected {expected} rows, found {len(rows)}")
    normalized = []
    missing = []
    qids = set()
    for index, row in enumerate(rows):
        original_qid = row.get("qid", row.get("id", index))
        qid = str(original_qid) if passthrough else f"{prefix}-{original_qid}"
        if qid in qids:
            raise ValueError(f"{name}: duplicate qid {qid}")
        qids.add(qid)
        image_name = str(row["img_name"])
        image_path = image_root / image_name
        if not image_path.is_file() and (image_root / "IU-Xray" / image_name).is_file():
            image_name = str(Path("IU-Xray") / image_name)
            image_path = image_root / image_name
        if not image_path.is_file():
            missing.append(image_name)
        question = str(row["question"])
        source_question_type = row.get("question_type", row.get("ground_truth_type"))
        choices = row.get("choices", "")
        question_type = normalize_question_type(source_question_type, row["answer"], choices) if task == "mixed_ce" else source_question_type
        if task == "mixed_ce" and question_type in {"binary", "ternary"}:
            question = question.rstrip() + "\nAnswer with exactly one of: Yes, No, Uncertain."
        elif task == "mixed_ce" and question_type == "choice":
            question = (
                question.rstrip()
                + f"\nChoices: {choices}\nAnswer with the option only."
            )
        elif task == "mixed_ce" and question_type == "short_answer":
            question = question.rstrip() + "\nAnswer with a concise answer."
        normalized.append({
            "id": qid,
            "qid": qid,
            "img_name": image_name,
            "patient_id": patient_or_image_cluster(image_name),
            "question": question,
            "source_question": str(row["question"]),
            "answer": str(row["answer"]),
            "task": task,
            "prompt_contract": "anchor-ce-v1" if task == "mixed_ce" else "source-exact",
            "dataset": name,
            "source_row": index,
            "source_qid": str(original_qid),
            "question_type": question_type,
            "source_question_type": source_question_type,
            "choices": choices,
        })
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} images missing; examples={missing[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{name}.json"
    if passthrough:
        destination.write_bytes(source.read_bytes())
    else:
        destination.write_text(json.dumps(normalized, indent=2) + "\n")
    return {
        "dataset": name,
        "task": task,
        "rows": len(normalized),
        "source": str(source.resolve()),
        "source_sha256": digest(source),
        "manifest": str(destination.resolve()),
        "manifest_sha256": digest(destination),
        "image_root": str(image_root.resolve()),
        "missing_images": 0,
    }


def main() -> None:
    med = ROOT / "data/medheval"
    specs = [
        ("cxr_vishal", med / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json", med / "images", "mixed_ce", 5587, "cxr-vishal", False),
        ("knowledge_mimic_ce", med / "benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json", med / "images", "mixed_ce", 2000, "knowledge-mimic", False),
        ("slake_fine_grained", med / "benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json", med / "images/Slake", "mixed_ce", 1536, "slake-fg", False),
        ("vqa_rad_official_oe", ROOT / "corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json", Path("/home/dbw/datasets/public/vqa_rad_hf/test_images"), "open_vqa", 200, "vqa-rad-oe", True),
        ("visual_mimic_oe", ROOT / "corrected_runs/high_efficiency/inputs/medheval.visual_oe_mimic.open_vqa.json", med / "images", "open_vqa", 490, "visual-mimic-oe", True),
        ("iu_xray_report", ROOT / "corrected_runs/high_efficiency/inputs/mmedrag.iuxray.report_generation.json", med / "images/IU-Xray", "report_generation", 590, "iu-report", True),
        ("mimic_cxr_report", ROOT / "corrected_runs/high_efficiency/inputs/mmedrag.mimic.report_generation.json", med / "images", "report_generation", 694, "mimic-report", True),
    ]
    audit = [freeze(name=n, source=s, image_root=i, task=t, expected=e, prefix=p, passthrough=x) for n, s, i, t, e, p, x in specs]
    official_type_source = ROOT / "data/medheval/code/evaluation/close_ended_evaluation/utils/type1_utils.py"
    (OUT / "audit.json").write_text(json.dumps({
        "protocol": "baseline-matrix-inputs-v2-source-question-type",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": digest(Path(__file__).resolve()),
        "official_type_source": str(official_type_source.resolve()),
        "official_type_source_sha256": digest(official_type_source),
        "datasets": audit,
    }, indent=2) + "\n")
    print(json.dumps({row["dataset"]: row["rows"] for row in audit}, indent=2))


if __name__ == "__main__":
    main()
