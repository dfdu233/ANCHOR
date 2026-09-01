#!/usr/bin/env python3
"""Build a blinded physician pack for Specificity Ratchet edge admission.

The builder treats every model-derived edge as a linguistic proposal. It never
reads VQA-RAD ground-truth answers and never assigns clinical support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROTOCOL_ID = "specificity-ratchet-physician-pack-v2"
MODEL_ORDER = ("huatuo", "hulu", "llava")
# Manual language-quality exclusions were frozen before physician labels. These
# questions yielded normal-anatomy/educational modifier edges rather than a
# clinical parent-to-child specificity proposal.
EXCLUDED_QIDS = {
    "vqa-rad-test-0050",  # large vein / right atrium in catheter explanation
    "vqa-rad-test-0098",  # normal renal-vein drainage anatomy
    "vqa-rad-test-0209",  # large organ / upper-right anatomy description
    "vqa-rad-test-0442",  # small/large intestine words in anatomy explanation
}
EXCLUDED_EDGE_TYPES_BY_QID = {
    "vqa-rad-test-0236": {"subtype"},  # neurological impact, not a child finding
    "vqa-rad-test-0309": {"size_morph"},  # focal modifies a negated aside
}

LATERALITY = re.compile(
    r"\b(left(?:-sided)?|right(?:-sided)?|bilateral|unilateral)\b", re.I
)
SIZE_MORPH = re.compile(
    r"\b(small|large|mild|moderate|severe|marked|massive|extensive|subtle|tiny|"
    r"diffuse|focal|well-defined|heterogeneous|rounded|round|irregular|multiple)\b",
    re.I,
)
SUBTYPE = re.compile(
    r"\b(indicative of|consistent with|known as|identified as|could represent|"
    r"may represent|likely represents?|likely (?:a|an|to be)|suggests|suggesting)\b",
    re.I,
)
ETIOLOGY = re.compile(
    r"\b(due to|caused by|secondary to|associated with|underlying conditions?|"
    r"results? from|can occur due to|may be caused by)\b",
    re.I,
)
EXPLICIT_CONTEXT = re.compile(
    r"\b(patient history|medical history|clinical correlation|clinical context|"
    r"symptoms?|laboratory|biopsy|histopatholog\w*|additional imaging|further "
    r"(?:evaluation|imaging)|cannot (?:determine|be determined)|depending (?:on|upon))\b",
    re.I,
)
CLINICAL_ANCHOR = re.compile(
    r"\b(opacit\w*|lesion\w*|mass(?:es)?|nodule\w*|effusion\w*|infiltrate\w*|"
    r"edema|hemorrhag\w*|hematoma\w*|infarct\w*|cyst\w*|calcif\w*|cavit\w*|"
    r"obstruction\w*|thrombo\w*|hydrocephalus|ventric\w*|air|fluid|fracture\w*|"
    r"abnormal\w*|patholog\w*|kidney|liver|spleen|lung\w*|heart|brain|bowel|colon)\b",
    re.I,
)
SPECIFICITY_TARGET = re.compile(
    r"\b(lesion\w*|mass(?:es)?|opacit\w*|nodule\w*|effusion\w*|infiltrate\w*|"
    r"edema|hemorrhag\w*|hematoma\w*|infarct\w*|cyst\w*|calcif\w*|cavit\w*|"
    r"obstruction\w*|thrombo\w*|hydrocephalus|ventric\w*|fracture\w*|artery|vein|"
    r"kidney|liver|spleen|lung\w*|heart|brain|bowel|colon|pancreas|lobe|hemisphere|"
    r"pleur\w*|fluid|air)\b",
    re.I,
)
DIAGNOSTIC_TARGET = re.compile(
    r"\b(pneumonia|infection\w*|inflamm\w*|abscess\w*|tumou?r\w*|neoplasm\w*|"
    r"malignan\w*|metasta\w*|carcinoma\w*|glioma\w*|hematoma\w*|hemorrhag\w*|"
    r"infarct\w*|stroke|ischemi\w*|edema|effusion\w*|atelectasis|pneumothorax|"
    r"hydrocephalus|cyst\w*|calcif\w*|fibrosis|sarcoidosis|tuberculosis|granuloma\w*|"
    r"obstruction\w*|thrombo\w*|agenesis|lipoma\w*|intussusception|pneumatosis|"
    r"perforation|fracture\w*|mass(?:es)?|nodule\w*|cavit\w*|airspace disease|"
    r"polycystic|atherosclero\w*|emphysema\w*|hernia\w*|hydronephrosis|"
    r"hydroureteronephrosis|demyelin\w*|multiple sclerosis|occlusion\w*|atrophy)\b",
    re.I,
)
NONCLINICAL_QUESTION = re.compile(
    r"\b((?:what|which).{0,30}(?:plane|modality|sequence|cross section|type of image)|"
    r"how.{0,30}(?:image|scan).{0,30}(?:taken|acquired)|how is the patient (?:oriented|"
    r"positioned)|patient position|patient orientation|orientation of the patient|"
    r"biological sex|male or female|artifact|pa cxr vs a lateral cxr|axial or sa(?:g|gg)ital)\b",
    re.I,
)
LATERALITY_REQUEST = re.compile(
    r"\b(left|right|side|sided|where|location|located|lobe|hemisphere|bilateral|"
    r"unilateral|apex|quadrant)\b",
    re.I,
)
SIZE_REQUEST = re.compile(
    r"\b(size|sized|large|small|severity|severe|how (?:big|large|wide)|measure\w*)\b",
    re.I,
)


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Edge:
    edge_type: str
    answer_span: str
    parent_proposal: str
    child_proposal: str
    added_constraint_proposal: str
    prompt_requested_increment: bool
    observability_screen: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, value: str, width: int = 12) -> str:
    return prefix + hashlib.sha256((PROTOCOL_ID + "|" + value).encode()).hexdigest()[:width]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_sentences(text: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right <= left:
            continue
        start = match.start() + left
        end = match.start() + right
        sentences.append(Sentence(text=text[start:end], start=start, end=end))
    return sentences


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,;:-")


def remove_constraint(text: str, pattern: re.Pattern[str]) -> str:
    parent = pattern.sub("", text)
    parent = re.sub(r"\s+([,.;:])", r"\1", parent)
    parent = re.sub(r"\s{2,}", " ", parent)
    return parent.strip()


def unique_matches(pattern: re.Pattern[str], text: str) -> str:
    seen: list[str] = []
    for match in pattern.finditer(text):
        value = normalize_space(match.group(0).lower())
        if value not in seen:
            seen.append(value)
    return " | ".join(seen)


def clause_parent(sentences: list[Sentence], index: int, match: re.Match[str]) -> str:
    before = normalize_space(sentences[index].text[: match.start()])
    weak = len(before.split()) < 5 or re.match(
        r"^(this|these|it|they|such|which|that)\b", before, re.I
    )
    if weak and index > 0:
        return sentences[index - 1].text
    return before or (sentences[index - 1].text if index > 0 else sentences[index].text)


def connector_edge(
    answer: str,
    sentences: list[Sentence],
    pattern: re.Pattern[str],
    edge_type: str,
    question: str,
) -> Edge | None:
    for index, sentence in enumerate(sentences):
        match = pattern.search(sentence.text)
        anchor_context = sentence.text + (" " + sentences[index - 1].text if index else "")
        if not match or not CLINICAL_ANCHOR.search(anchor_context):
            continue
        added_text = sentence.text[match.end() :]
        if not DIAGNOSTIC_TARGET.search(added_text):
            continue
        parent = clause_parent(sentences, index, match)
        start = sentences[index - 1].start if parent == sentences[index - 1].text and index else sentence.start
        span = answer[start : sentence.end]
        added = normalize_space(sentence.text[match.start() :])
        context = bool(EXPLICIT_CONTEXT.search(span)) or edge_type == "etiology"
        return Edge(
            edge_type=edge_type,
            answer_span=span,
            parent_proposal=parent,
            child_proposal=sentence.text,
            added_constraint_proposal=added,
            prompt_requested_increment=False,
            observability_screen=(
                "explicit_or_likely_nonvisual_context" if context else "potentially_single_image_decidable"
            ),
        )
    return None


def modifier_edge(
    answer: str,
    sentences: list[Sentence],
    pattern: re.Pattern[str],
    edge_type: str,
    question: str,
) -> Edge | None:
    for sentence in sentences:
        matches = list(pattern.finditer(sentence.text))
        if not matches or not CLINICAL_ANCHOR.search(sentence.text):
            continue
        local_matches = []
        for match in matches:
            if edge_type == "size_morph":
                # Most retained modifiers are prenominal (large mass, multiple
                # infarcts). Requiring a nearby following clinical target avoids
                # false edges such as "multiple slices" in an educational aside.
                local = sentence.text[match.end() : match.end() + 70]
            else:
                local = sentence.text[max(0, match.start() - 55) : match.end() + 55]
            if SPECIFICITY_TARGET.search(local):
                local_matches.append(match)
        if not local_matches:
            continue
        parent = remove_constraint(sentence.text, pattern)
        if parent == sentence.text or len(parent.split()) < 4:
            continue
        requested = bool(
            (LATERALITY_REQUEST if edge_type == "laterality" else SIZE_REQUEST).search(question)
        )
        return Edge(
            edge_type=edge_type,
            answer_span=answer[sentence.start : sentence.end],
            parent_proposal=parent,
            child_proposal=sentence.text,
            added_constraint_proposal=unique_matches(pattern, sentence.text),
            prompt_requested_increment=requested,
            observability_screen="potentially_single_image_decidable",
        )
    return None


def extract_edges(question: str, answer: str) -> dict[str, Edge]:
    sentences = split_sentences(answer)
    edges = {
        "laterality": modifier_edge(answer, sentences, LATERALITY, "laterality", question),
        "size_morph": modifier_edge(answer, sentences, SIZE_MORPH, "size_morph", question),
        "subtype": connector_edge(answer, sentences, SUBTYPE, "subtype", question),
        "etiology": connector_edge(answer, sentences, ETIOLOGY, "etiology", question),
    }
    return {name: edge for name, edge in edges.items() if edge is not None}


def classify_modality(question: str, answer: str) -> str:
    text = (question + " " + answer).lower()
    if any(term in text for term in ("mri", "t1-weighted", "t2-weighted", "flair", "diffusion")):
        return "MRI"
    if any(term in text for term in ("ct scan", "computed tomography")):
        return "CT"
    if any(term in text for term in ("x-ray", "xray", "radiograph", "cxr")):
        return "XR"
    if "ultrasound" in text:
        return "US"
    return "other_or_unstated"


def classify_anatomy(question: str, answer: str) -> str:
    text = (question + " " + answer).lower()
    if any(
        term in text
        for term in (
            "brain",
            "cerebr",
            "ventric",
            "frontal lobe",
            "temporal lobe",
            "parietal",
            "occipital",
            "cerebell",
            "pons",
        )
    ):
        return "neuro"
    if any(term in text for term in ("lung", "chest", "pleur", "heart", "thorax", "mediast")):
        return "thorax"
    if any(
        term in text
        for term in (
            "liver",
            "kidney",
            "abdomen",
            "abdominal",
            "bowel",
            "colon",
            "spleen",
            "pancrea",
            "gallbladder",
            "pelvis",
            "ovarian",
        )
    ):
        return "abdomen_pelvis"
    return "other"


def length_stratum(answer: str) -> str:
    words = len(answer.split())
    if words <= 50:
        return "short_le_50"
    if words <= 100:
        return "medium_51_100"
    return "long_gt_100"


def category_flags(question: str, answer: str) -> set[str]:
    return set(extract_edges(question, answer))


def quality_score(
    edges: dict[str, Edge], repeat_by_type: dict[str, list[str]], answer: str
) -> tuple[int, int, int]:
    repeated = sum(max(0, len(models) - 1) for models in repeat_by_type.values())
    explicit_context = sum(
        edge.observability_screen == "explicit_or_likely_nonvisual_context"
        for edge in edges.values()
    )
    # Prefer diverse edges and cross-model recurrence; use shorter text only as tie-break.
    return (4 * len(edges) + 2 * repeated + explicit_context, repeated, -len(answer.split()))


def select_edges_balanced(cases: list[dict]) -> None:
    counts: Counter[str] = Counter()
    ordered = sorted(cases, key=lambda row: row["case_id"])
    for case in ordered:
        available: dict[str, Edge] = case.pop("all_edges")
        ranked = sorted(
            available,
            key=lambda edge_type: (
                counts[edge_type],
                0 if edge_type in {"etiology", "subtype"} else 1,
                edge_type,
            ),
        )
        chosen = ranked[:3]
        case["edges"] = [available[edge_type] for edge_type in chosen]
        counts.update(chosen)


def choose_cases(
    manifest_rows: list[dict], answers: dict[str, dict[str, dict]], target_images: int
) -> tuple[list[dict], dict[str, int]]:
    manifest = {row["id"]: row for row in manifest_rows}
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        qid = row["id"]
        question = row["question"]
        if qid in EXCLUDED_QIDS or NONCLINICAL_QUESTION.search(question):
            continue
        primary = answers["huatuo"][qid]
        edges = extract_edges(question, primary["text"])
        edges = {
            edge_type: edge
            for edge_type, edge in edges.items()
            if edge_type not in EXCLUDED_EDGE_TYPES_BY_QID.get(qid, set())
        }
        if not edges:
            continue
        repeat_by_type: dict[str, list[str]] = {}
        for edge_type in edges:
            repeat_by_type[edge_type] = [
                model
                for model in MODEL_ORDER
                if edge_type in category_flags(question, answers[model][qid]["text"])
            ]
        case_id = stable_id("SR2-", row["img_name"])
        by_image[row["img_name"]].append(
            {
                "case_id": case_id,
                "qid": qid,
                "question": question,
                "image_name": row["img_name"],
                "source_row": row.get("source_row"),
                "answer": primary["text"],
                "answer_line": primary["_line"],
                "all_edges": edges,
                "repeat_by_type": repeat_by_type,
                "modality": classify_modality(question, primary["text"]),
                "anatomy": classify_anatomy(question, primary["text"]),
                "answer_length": length_stratum(primary["text"]),
                "quality": quality_score(edges, repeat_by_type, primary["text"]),
            }
        )

    best_by_image = [max(rows, key=lambda row: row["quality"]) for rows in by_image.values()]
    if len(best_by_image) <= target_images:
        selected = best_by_image
    else:
        # Preserve rare modality/anatomy cells: remove only the globally weakest
        # cases while retaining at least 90% of each cell (floor one).
        cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for case in best_by_image:
            cells[(case["modality"], case["anatomy"])].append(case)
        keep: list[dict] = []
        desired_total = target_images
        quotas = {
            cell: max(1, int(round(len(rows) * desired_total / len(best_by_image))))
            for cell, rows in cells.items()
        }
        while sum(quotas.values()) > desired_total:
            cell = max(
                (cell for cell in quotas if quotas[cell] > 1),
                key=lambda item: (quotas[item], len(cells[item]), item),
            )
            quotas[cell] -= 1
        while sum(quotas.values()) < desired_total:
            eligible = [cell for cell in quotas if quotas[cell] < len(cells[cell])]
            cell = max(eligible, key=lambda item: (len(cells[item]) - quotas[item], item))
            quotas[cell] += 1
        for cell, rows in cells.items():
            keep.extend(sorted(rows, key=lambda row: row["quality"], reverse=True)[: quotas[cell]])
        selected = keep

    selected = sorted(selected, key=lambda row: row["case_id"])
    select_edges_balanced(selected)
    return selected, {
        "strict_candidate_images": len(best_by_image),
        "requested_images": target_images,
        "selected_images": len(selected),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"),
    )
    parser.add_argument("--target-images", type=int, default=80)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = repo / "corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json"
    answer_paths = {
        "huatuo": repo / "corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v3_512/answers.jsonl",
        "hulu": repo / "corrected_runs/unified_eval/full/hulu_native_vqa_rad_oe_v1/answers.jsonl",
        "llava": repo / "corrected_runs/unified_eval/full/llava_native_vqa_rad_oe_v1/answers.jsonl",
    }
    manifest_rows = json.loads(manifest_path.read_text())
    answers: dict[str, dict[str, dict]] = {}
    for model, path in answer_paths.items():
        rows = read_jsonl(path)
        answers[model] = {}
        for line, row in enumerate(rows, start=1):
            row = dict(row)
            row["_line"] = line
            answers[model][row["question_id"]] = row
        if len(answers[model]) != 200:
            raise ValueError(f"{model}: expected 200 unique answers")

    cases, selection = choose_cases(manifest_rows, answers, args.target_images)
    if not cases:
        raise ValueError("no strict language candidates")

    reviewer_rows: list[dict] = []
    private_rows: list[dict] = []
    edge_type_counts: Counter[str] = Counter()
    observability_counts: Counter[str] = Counter()
    for case in cases:
        for edge_index, edge in enumerate(case["edges"], start=1):
            edge_id = f"{case['case_id']}-E{edge_index:02d}"
            reviewer = {
                "case_id": case["case_id"],
                "edge_id": edge_id,
                "question": case["question"],
                "image_relpath": f"test_images/{case['image_name']}",
                "answer_span": edge.answer_span,
                "parent_proposal": edge.parent_proposal,
                "child_proposal": edge.child_proposal,
                "added_constraint_proposal": edge.added_constraint_proposal,
                "edge_type": edge.edge_type,
                "modality_stratum": case["modality"],
                "anatomy_stratum": case["anatomy"],
                "answer_length_stratum": case["answer_length"],
                "observability_screen": edge.observability_screen,
                "prompt_requested_increment": edge.prompt_requested_increment,
                "proposal_only": True,
            }
            reviewer_rows.append(reviewer)
            edge_type_counts[edge.edge_type] += 1
            observability_counts[edge.observability_screen] += 1
            private_rows.append(
                {
                    "case_id": case["case_id"],
                    "edge_id": edge_id,
                    "question_id": case["qid"],
                    "source_model": "huatuo",
                    "source_answer_path": str(answer_paths["huatuo"].relative_to(repo)),
                    "source_answer_line": case["answer_line"],
                    "source_row": case["source_row"],
                    "image_name": case["image_name"],
                    "same_type_models_screening_only": case["repeat_by_type"][edge.edge_type],
                }
            )

    write_jsonl(output / "candidates.blinded.jsonl", reviewer_rows)
    write_jsonl(output / "provenance.private.jsonl", private_rows)

    annotation_fields = [
        *list(reviewer_rows[0]),
        "reviewer_id",
        "edge_entailment_admitted",
        "parent_visual_support",
        "child_visual_support",
        "increment_observability",
        "logical_scope_preserved",
        "reviewer_confidence",
        "clinical_usefulness_if_backed_off",
        "clinically_harmful_if_wrong",
        "rationale",
    ]
    blank_rows = [
        {
            **row,
            "reviewer_id": "",
            "edge_entailment_admitted": "",
            "parent_visual_support": "",
            "child_visual_support": "",
            "increment_observability": "",
            "logical_scope_preserved": "",
            "reviewer_confidence": "",
            "clinical_usefulness_if_backed_off": "",
            "clinically_harmful_if_wrong": "",
            "rationale": "",
        }
        for row in reviewer_rows
    ]
    write_csv(output / "annotations.reviewer_1.csv", annotation_fields, blank_rows)
    write_csv(output / "annotations.reviewer_2.csv", annotation_fields, blank_rows)

    adjudication_fields = [
        "case_id",
        "edge_id",
        "r1_edge_entailment_admitted",
        "r2_edge_entailment_admitted",
        "r1_parent_visual_support",
        "r2_parent_visual_support",
        "r1_child_visual_support",
        "r2_child_visual_support",
        "r1_increment_observability",
        "r2_increment_observability",
        "r1_logical_scope_preserved",
        "r2_logical_scope_preserved",
        "r1_clinical_usefulness_if_backed_off",
        "r2_clinical_usefulness_if_backed_off",
        "r1_clinically_harmful_if_wrong",
        "r2_clinically_harmful_if_wrong",
        "r1_reviewer_confidence",
        "r2_reviewer_confidence",
        "r1_rationale",
        "r2_rationale",
        "final_edge_entailment_admitted",
        "final_parent_visual_support",
        "final_child_visual_support",
        "final_increment_observability",
        "final_logical_scope_preserved",
        "final_clinical_usefulness_if_backed_off",
        "final_clinically_harmful_if_wrong",
        "adjudicator_id",
        "disagreement_reason",
        "adjudication_rationale",
    ]
    write_csv(
        output / "adjudication.csv",
        adjudication_fields,
        ({"case_id": row["case_id"], "edge_id": row["edge_id"]} for row in reviewer_rows),
    )

    annotation_schema = {
        "protocol_id": PROTOCOL_ID,
        "unit": "one adjacent proposed parent-to-child edge",
        "proposal_warning": "All text-derived edges are proposals, never truth.",
        "fields": {
            "edge_entailment_admitted": ["yes", "no", "uncertain"],
            "parent_visual_support": [
                "supported",
                "refuted",
                "undetermined",
                "unobservable",
            ],
            "child_visual_support": [
                "supported",
                "refuted",
                "undetermined",
                "unobservable",
            ],
            "increment_observability": [
                "observable_on_supplied_image",
                "requires_other_view_or_sequence",
                "requires_history_lab_pathology_or_prior",
                "fundamentally_nonvisual_knowledge",
                "uncertain",
            ],
            "logical_scope_preserved": ["yes", "no", "not_applicable"],
            "reviewer_confidence": ["low", "medium", "high"],
            "clinical_usefulness_if_backed_off": [
                "improves",
                "unchanged",
                "minor_loss",
                "major_loss",
                "uncertain",
            ],
            "clinically_harmful_if_wrong": ["no", "minor", "major", "uncertain"],
        },
        "state_definitions": {
            "undetermined": "The claim is in principle image-observable, but this supplied image is insufficient to support or refute it.",
            "unobservable": "The claim requires an unavailable evidence source such as history, laboratory, pathology, prior study, or another required acquisition.",
        },
        "rules": [
            "First judge edge validity, then label parent and child from the image independently; enforce child-implies-parent consistency only after adjudication for admitted edges.",
            "A child is not supported because it repeats across models or resembles the short reference.",
            "Preserve A OR B as one uncertain alternative set.",
            "Reject the edge if the child changes category rather than adding one constraint.",
            "Use unobservable rather than guessing when another view, history, lab, pathology, or prior is required.",
        ],
    }
    (output / "annotation_schema.json").write_text(
        json.dumps(annotation_schema, indent=2, ensure_ascii=False) + "\n"
    )

    source_hashes = {
        "manifest": sha256_path(manifest_path),
        **{model: sha256_path(path) for model, path in answer_paths.items()},
    }
    fingerprint = hashlib.sha256(
        ("|".join(source_hashes.values()) + "|" + PROTOCOL_ID).encode()
    ).hexdigest()
    summary = {
        "protocol_id": PROTOCOL_ID,
        "dataset": "vqa_rad_official_test_oe",
        "primary_model": "HuatuoGPT-Vision-7B formal admissible output",
        "screening_only_priority_models": ["Hulu-Med-4B", "LLaVA-Med-v1.5-7B"],
        "seed": None,
        "command": (
            "python anchor/corrected_sgta/build_specificity_ratchet_physician_pack_v2.py "
            "--target-images 80"
        ),
        "verification_command": (
            "python anchor/corrected_sgta/verify_specificity_ratchet_physician_pack_v2.py"
        ),
        "builder_sha256": sha256_path(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "source_fingerprint": fingerprint,
        "selection": selection,
        "n_edges": len(reviewer_rows),
        "max_edges_per_image": max(Counter(row["case_id"] for row in reviewer_rows).values()),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "observability_screen_counts": dict(sorted(observability_counts.items())),
        "modality_counts_images": dict(sorted(Counter(case["modality"] for case in cases).items())),
        "anatomy_counts_images": dict(sorted(Counter(case["anatomy"] for case in cases).items())),
        "answer_length_counts_images": dict(
            sorted(Counter(case["answer_length"] for case in cases).items())
        ),
        "annotation_burden": {
            "images_per_reviewer": len(cases),
            "edges_per_reviewer": len(reviewer_rows),
            "estimated_minutes_per_image": "2.0-3.0 when edges are grouped by image",
            "estimated_hours_per_reviewer": f"{2.0 * len(cases) / 60:.1f}-{3.0 * len(cases) / 60:.1f}",
            "estimated_adjudication_hours": "1.0-2.0 depending on disagreement",
        },
        "hard_prohibition": (
            "Model text, cross-model repetition, VQA-RAD short references, VinDr co-occurrence, "
            "and automatic graph relations cannot define truth or an ontology."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
