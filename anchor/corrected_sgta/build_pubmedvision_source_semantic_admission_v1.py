#!/usr/bin/env python3
"""Build a source-only semantic admission pack from Huatuo training text.

The primary text is the assistant response in the two public PubMedVision VQA
training stages.  Human questions are audited separately for presupposition;
Original Caption text is used only as an image-provenance cross-check.  The
tool has no target-dataset or model-output input.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


VERSION = "pubmedvision-source-semantic-admission-v1.1"
# Preserve the scientific partition and blind-review sample from v1.  Version
# 1.1 repairs only the source-index schema-key diagnostic; it must not silently
# change which examples are assigned to a split or selected for review.
PARTITION_VERSION = "pubmedvision-source-semantic-admission-v1"
DEFAULT_SOURCE_INDEX = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr_v2/source_index.jsonl"
)
DEFAULT_ALIGNMENT = Path(
    "/home/dbw/data/PubMedVision/PubMedVision_Alignment_VQA.json"
)
DEFAULT_INSTRUCTION = Path(
    "/home/dbw/data/PubMedVision/PubMedVision_InstructionTuning_VQA.json"
)
DEFAULT_OUTPUT = Path(
    "/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr_v2/"
    "source_semantic_admission_v1_1"
)
STATES = ("positive", "negative", "uncertain", "unmentioned")
QUESTION_LABELS = (
    "none",
    "neutral_query",
    "presupposes_positive",
    "presupposes_negative",
    "uncertain",
)


@dataclass(frozen=True)
class FindingSpec:
    finding: str
    aliases: tuple[str, ...]

    @property
    def alias_pattern(self) -> re.Pattern[str]:
        return re.compile("(?:" + "|".join(self.aliases) + ")", re.IGNORECASE)


FINDINGS = (
    FindingSpec(
        "aortic_enlargement",
        (
            r"\baortic enlargement\b",
            r"\baortic (?:dilatation|dilation|ectasia)\b",
            r"\b(?:enlarged|dilated|ectatic) (?:thoracic )?aorta\b",
            r"\b(?:thoracic )?aorta (?:is |appears )?(?:enlarged|dilated|ectatic)\b",
        ),
    ),
    FindingSpec(
        "cardiomegaly",
        (
            r"\bcardiomegaly\b",
            r"\bcardiac enlargement\b",
            r"\benlarged (?:heart|cardiac silhouette)\b",
            r"\b(?:heart|cardiac silhouette) (?:is |appears )enlarged\b",
        ),
    ),
    FindingSpec(
        "pleural_effusion",
        (
            r"\bpleural effusions?\b",
            r"\bpleural fluid(?: collection)?\b",
            r"\bfluid in the pleural (?:space|cavity)\b",
        ),
    ),
    FindingSpec(
        "pulmonary_fibrosis",
        (
            r"\bpulmonary fibrosis\b",
            r"\blung fibrosis\b",
            r"\binterstitial fibrosis\b",
            r"\bpulmonary fibrotic changes?\b",
            r"\bfibrotic changes? (?:in|of) (?:the )?lungs?\b",
            r"\bfibrosing interstitial lung disease\b",
        ),
    ),
)


UNCERTAIN_CUES = re.compile(
    r"\b(?:cannot|can not|can't|could not) (?:be )?exclude(?:d)?\b"
    r"|\bnot (?:entirely )?excluded\b"
    r"|\b(?:possible|possibly|probable|probably|likely|perhaps|presumed)\b"
    r"|\b(?:may|might|could) (?:be|represent|reflect|indicate)\b"
    r"|\b(?:suggestive of|suspicious for|concerning for|compatible with|"
    r"consistent with|equivocal|indeterminate|questionable)\b"
    r"|\b(?:history of|previous|previously|prior|resolved|resolution of)\b",
    re.IGNORECASE,
)
HEDGED_NEGATION = re.compile(
    r"\bno (?:definite|significant|large|new|obvious|clear|gross)\b",
    re.IGNORECASE,
)
NEUTRAL_QUESTION = re.compile(
    r"\b(?:is|are) there\b|\bdoes (?:the |this )?(?:image|x-ray|radiograph) show\b"
    r"|\bpresence or absence\b|\bwhether\b|\bevaluate for\b|\bassess for\b"
    r"|\bcan (?:you )?(?:identify|see|detect)\b",
    re.IGNORECASE,
)
GENERIC_ALIGNMENT_PROMPT = re.compile(
    r"^(?:please )?(?:analyze|describe) (?:the |this |these )?images?\b"
    r"|^what (?:is|are) (?:depicted|shown|visible) in (?:the |this |these )?images?\??$"
    r"|^what do(?:es)? (?:the |this |these )?images? show\??$"
    r"|^(?:please )?provide (?:a )?(?:comprehensive |detailed )?"
    r"(?:analysis|description) of (?:the |this |these )?images?\b",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[Any]:
    """Stream a top-level JSON array using only the Python standard library."""

    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        finished = False
        eof = False
        while not finished:
            if not eof and (len(buffer) - position < chunk_size // 2):
                buffer = buffer[position:] + handle.read(chunk_size)
                position = 0
                if len(buffer) < chunk_size and handle.tell() == path.stat().st_size:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON array: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"top-level JSON value is not an array: {path}")
                position += 1
                started = True
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                finished = True
                break
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"unterminated JSON array: {path}")
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = buffer[position:] + handle.read(chunk_size)
                position = 0
                continue
            yield value
            position = end
        if not finished:
            raise ValueError(f"unterminated JSON array: {path}")


def source_group(image: str) -> str:
    name = Path(image).stem
    match = re.match(r"^(pmc_\d+)(?:_\d+)?$", name, re.IGNORECASE)
    return match.group(1).lower() if match else name.lower()


def source_split(group: str) -> str:
    bucket = int(
        hashlib.sha256(f"{PARTITION_VERSION}:split:{group}".encode()).hexdigest(), 16
    ) % 100
    if bucket < 70:
        return "source_train"
    if bucket < 85:
        return "source_dev"
    return "source_review"


def _context(text: str, start: int, end: int, radius: int = 110) -> str:
    """Return an alias-local sentence/semicolon clause.

    Uncertainty and negation cues outside this span are not allowed to scope
    over the mention.  The radius is an additional conservative bound for very
    long publication-caption sentences.
    """

    boundaries = ".;!?\n"
    clause_left = max((text.rfind(mark, 0, start) for mark in boundaries), default=-1) + 1
    right_candidates = [text.find(mark, end) for mark in boundaries]
    right_candidates = [value for value in right_candidates if value >= 0]
    clause_right = min(right_candidates) if right_candidates else len(text)
    left = max(clause_left, start - radius)
    right = min(clause_right, end + radius)
    return text[left:right].strip()


def _negative_match(text: str, alias: str) -> tuple[bool, str | None]:
    patterns = (
        rf"\bno\s+(?:evidence|signs?)\s+of\s+(?:an?\s+)?{alias}",
        rf"\bno\s+(?:an?\s+)?{alias}",
        rf"\bwithout\s+(?:an?\s+)?{alias}",
        rf"\babsence\s+of\s+(?:an?\s+)?{alias}",
        rf"\bnegative\s+for\s+(?:an?\s+)?{alias}",
        rf"{alias}\s+(?:is|are|was|were)?\s*(?:absent|not present|not seen|not identified)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, match.group(0)
    return False, None


def classify_finding(text: str, spec: FindingSpec) -> dict[str, Any]:
    normalized = " ".join(str(text or "").split())
    mentions = list(spec.alias_pattern.finditer(normalized))
    if not mentions:
        return {"state": "unmentioned", "evidence": []}
    evidence = []
    local_states = []
    for mention in mentions:
        context = _context(normalized, mention.start(), mention.end())
        alias = re.escape(mention.group(0))
        negative, negative_cue = _negative_match(context, alias)
        uncertain_match = UNCERTAIN_CUES.search(context)
        hedged = HEDGED_NEGATION.search(context)
        if uncertain_match or hedged:
            state = "uncertain"
            cue = (uncertain_match or hedged).group(0)
        elif negative:
            state = "negative"
            cue = negative_cue
        else:
            state = "positive"
            cue = None
        local_states.append(state)
        evidence.append(
            {
                "mention": mention.group(0),
                "context": context,
                "local_state": state,
                "cue": cue,
            }
        )
    unique = set(local_states)
    if "uncertain" in unique or len(unique) > 1:
        state = "uncertain"
    elif "positive" in unique:
        state = "positive"
    else:
        state = "negative"
    return {"state": state, "evidence": evidence}


def classify_question(text: str, spec: FindingSpec) -> dict[str, Any]:
    surface = classify_finding(text, spec)
    state = surface["state"]
    if state == "unmentioned":
        label = "none"
    elif state == "negative":
        label = "presupposes_negative"
    elif state == "uncertain":
        label = "uncertain"
    elif NEUTRAL_QUESTION.search(str(text)):
        label = "neutral_query"
    else:
        label = "presupposes_positive"
    return {"surface_state": state, "presupposition": label, "evidence": surface["evidence"]}


def is_generic_alignment_prompt(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).strip()
    return bool(GENERIC_ALIGNMENT_PROMPT.search(normalized))


def load_source_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    keys = Counter()
    types: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts = Counter()
    modality_counts = Counter()
    cxr_caption_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"source index row {line_number} is not an object")
        image = str(row.get("image", ""))
        if not image or image in rows:
            raise ValueError(f"missing/duplicate source image at row {line_number}: {image}")
        rows[image] = row
        keys.update(row.keys())
        for key, value in row.items():
            types[key][type(value).__name__] += 1
        split_counts[str(row.get("split"))] += 1
        modality_counts[str(row.get("raw_modality"))] += 1
        if re.search(r"\b(?:chest\s+)?(?:x[- ]?ray|radiograph)\b", str(row.get("caption", "")), re.I):
            cxr_caption_count += 1
    return rows, {
        "rows": len(rows),
        "keys": dict(sorted(keys.items())),
        "types": {key: dict(sorted(value.items())) for key, value in sorted(types.items())},
        "provided_split_counts": dict(sorted(split_counts.items())),
        "raw_modality_counts": dict(sorted(modality_counts.items())),
        "caption_explicit_cxr_term_count": cxr_caption_count,
    }


def conversation_units(item: Mapping[str, Any], stage: str) -> Iterator[dict[str, Any]]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{stage} item lacks conversations list: {item.get('id')}")
    question = ""
    assistant_index = 0
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or not isinstance(turn.get("value"), str):
            raise ValueError(f"malformed conversation turn: {item.get('id')}:{turn_index}")
        role = str(turn.get("from"))
        if role == "human":
            question = turn["value"]
        elif role == "gpt":
            assistant_index += 1
            yield {
                "response_unit_id": f"{item.get('id')}:gpt{assistant_index}",
                "stage": stage,
                "question": question,
                "assistant_response": turn["value"],
            }


def scan_stage(
    path: Path,
    stage: str,
    source_rows: Mapping[str, Mapping[str, Any]],
    *,
    streaming: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    counters = Counter()
    field_counts = Counter()
    role_patterns = Counter()
    if streaming:
        raw_items: Iterable[Any] = iter_json_array(path)
        loader = "stdlib_streaming_fallback"
        total_items = None
    else:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(f"{stage} top-level JSON value is not an array")
        total_items = len(loaded)
        expected_prefix = (
            "Alignment_VQA_" if stage == "alignment" else "Instruction-Tuning_"
        )
        source_numbers = set()
        for row in source_rows.values():
            match = re.fullmatch(r"Original_Caption_(\d+)", str(row.get("id", "")))
            if not match:
                source_numbers.clear()
                break
            source_numbers.add(int(match.group(1)))
        direct = []
        if source_numbers:
            for number in sorted(source_numbers):
                index = number - 1
                if index < 0 or index >= len(loaded):
                    direct = []
                    break
                item = loaded[index]
                if not isinstance(item, dict) or item.get("id") != f"{expected_prefix}{number}":
                    direct = []
                    break
                direct.append(item)
        if direct and len(direct) == len(source_numbers):
            raw_items = direct
            loader = "stdlib_json_load_verified_direct_source_index"
        else:
            raw_items = loaded
            loader = "stdlib_json_load_full_scan_fallback"
    for raw in raw_items:
        counters["items_scanned"] += 1
        if not isinstance(raw, dict):
            raise ValueError(f"{stage} item {counters['items_scanned']} is not an object")
        field_counts.update(raw.keys())
        images_raw = raw.get("image", [])
        images = [images_raw] if isinstance(images_raw, str) else list(images_raw)
        if not all(isinstance(value, str) for value in images):
            raise ValueError(f"{stage} item has malformed image list: {raw.get('id')}")
        matched = sorted({image for image in images if image in source_rows})
        if not matched:
            continue
        counters["items_with_source_image"] += 1
        counters["matched_source_images"] += len(matched)
        roles = tuple(str(turn.get("from")) for turn in raw.get("conversations", []) if isinstance(turn, dict))
        role_patterns[str(roles)] += 1
        groups = sorted({source_group(image) for image in matched})
        group = groups[0] if len(groups) == 1 else "multi:" + hashlib.sha256("|".join(groups).encode()).hexdigest()[:20]
        captions = [str(source_rows[image].get("caption", "")) for image in matched]
        source_ids = [str(source_rows[image].get("id", "")) for image in matched]
        for unit in conversation_units(raw, stage):
            counters["assistant_response_units"] += 1
            unit.update(
                {
                    "vqa_id": str(raw.get("id", "")),
                    "matched_source_images": matched,
                    "matched_source_ids": source_ids,
                    "source_group": group,
                    "source_split": source_split(group),
                    "captions": captions,
                    "vqa_modality": raw.get("modality"),
                    "vqa_body_part": raw.get("body_part"),
                    "generic_alignment_prompt": (
                        stage == "alignment" and is_generic_alignment_prompt(unit["question"])
                    ),
                }
            )
            records.append(unit)
    result = {
        **dict(sorted(counters.items())),
        "loader": loader,
        "total_items_in_file": total_items,
        "item_field_counts": dict(sorted(field_counts.items())),
        "matched_role_patterns": dict(sorted(role_patterns.items())),
    }
    if not streaming:
        del loaded
        gc.collect()
    return records, result


def deterministic_key(seed: int, *parts: object) -> str:
    payload = ":".join(str(value) for value in (PARTITION_VERSION, seed, *parts))
    return hashlib.sha256(payload.encode()).hexdigest()


def select_review_rows(records: Sequence[Mapping[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in records if row["source_split"] == "source_review"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(str(row["stage"]), str(row["finding"]), str(row["assistant_state"]))].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: deterministic_key(seed, *key, row["response_unit_id"]))
    stage_findings = [(stage, spec.finding) for stage in ("alignment", "instruction_tuning") for spec in FINDINGS]
    base_quota = max(1, size // (len(stage_findings) * len(STATES)))
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for stage, finding in stage_findings:
        for state in STATES:
            for row in groups.get((stage, finding, state), [])[:base_quota]:
                identity = (row["response_unit_id"], row["finding"])
                if identity not in used:
                    used.add(identity)
                    selected.append(row)
    per_stage_finding_target = max(1, size // len(stage_findings))
    for stage, finding in stage_findings:
        have = sum(row["stage"] == stage and row["finding"] == finding for row in selected)
        if have >= per_stage_finding_target:
            continue
        remainder = sorted(
            (
                row
                for row in candidates
                if row["stage"] == stage
                and row["finding"] == finding
                and (row["response_unit_id"], row["finding"]) not in used
            ),
            key=lambda row: deterministic_key(
                seed, stage, finding, row["assistant_state"], row["response_unit_id"]
            ),
        )
        for row in remainder[: per_stage_finding_target - have]:
            used.add((row["response_unit_id"], row["finding"]))
            selected.append(row)
    if len(selected) < size:
        remainder = sorted(
            (
                row
                for row in candidates
                if (row["response_unit_id"], row["finding"]) not in used
            ),
            key=lambda row: deterministic_key(
                seed,
                row["stage"],
                row["finding"],
                row["assistant_state"],
                row["response_unit_id"],
            ),
        )
        selected.extend(remainder[: size - len(selected)])
    if len(selected) < size:
        raise ValueError(f"source_review contains only {len(selected)} unique review units; required {size}")
    selected = selected[:size]
    finding_counts = Counter(row["finding"] for row in selected)
    insufficient = {
        spec.finding: finding_counts[spec.finding]
        for spec in FINDINGS
        if finding_counts[spec.finding] < 25
    }
    if insufficient:
        raise ValueError(f"blind review has fewer than 25 rows for a finding: {insufficient}")
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.review_size < 100:
        raise ValueError("blind source review must contain at least 100 rows")
    output_dir = args.output_dir
    final_names = (
        "stats.json",
        "source_records.jsonl",
        "blind_review.csv",
        "blind_review_key.jsonl",
        "caption_crosscheck.jsonl",
        "_COMPLETE.json",
    )
    existing = [name for name in final_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "source semantic admission output is write-once; existing="
            + ",".join(existing)
        )
    source_rows, source_schema = load_source_index(args.source_index)
    stages = (
        ("alignment", args.alignment),
        ("instruction_tuning", args.instruction),
    )
    all_units = []
    stage_schema = {}
    for stage, path in stages:
        units, schema = scan_stage(
            path, stage, source_rows, streaming=args.streaming_fallback
        )
        all_units.extend(units)
        stage_schema[stage] = schema

    semantic_rows = []
    caption_rows = []
    for unit in all_units:
        caption_text = "\n".join(unit["captions"])
        for spec in FINDINGS:
            assistant = classify_finding(unit["assistant_response"], spec)
            question = classify_question(unit["question"], spec)
            caption = classify_finding(caption_text, spec)
            row = {
                **unit,
                "finding": spec.finding,
                "assistant_state": assistant["state"],
                "assistant_evidence": assistant["evidence"],
                "question_surface_state": question["surface_state"],
                "question_presupposition": question["presupposition"],
                "question_evidence": question["evidence"],
                "caption_crosscheck_state": caption["state"],
            }
            semantic_rows.append(row)
            if assistant["state"] != "unmentioned" or caption["state"] != "unmentioned":
                caption_rows.append(
                    {
                        "response_unit_id": unit["response_unit_id"],
                        "stage": unit["stage"],
                        "finding": spec.finding,
                        "assistant_state": assistant["state"],
                        "caption_state": caption["state"],
                        "matched_source_images": unit["matched_source_images"],
                        "matched_source_ids": unit["matched_source_ids"],
                        "captions": unit["captions"],
                        "caption_evidence": caption["evidence"],
                    }
                )

    review = select_review_rows(semantic_rows, args.review_size, args.seed)
    public_rows = []
    private_rows = []
    for index, row in enumerate(review, 1):
        review_id = hashlib.sha256(
            f"{PARTITION_VERSION}:{args.seed}:{row['response_unit_id']}:{row['finding']}".encode()
        ).hexdigest()[:20]
        public_rows.append(
            {
                "review_id": review_id,
                "stage": row["stage"],
                "finding": row["finding"],
                "question": row["question"],
                "assistant_response": row["assistant_response"],
                "reviewer_assistant_state": "",
                "reviewer_question_presupposition": "",
                "reviewer_evidence_span": "",
                "reviewer_notes": "",
            }
        )
        private_rows.append(
            {
                "review_id": review_id,
                "response_unit_id": row["response_unit_id"],
                "vqa_id": row["vqa_id"],
                "stage": row["stage"],
                "finding": row["finding"],
                "source_split": row["source_split"],
                "source_group": row["source_group"],
                "matched_source_images": row["matched_source_images"],
                "matched_source_ids": row["matched_source_ids"],
                "automatic_assistant_state": row["assistant_state"],
                "automatic_assistant_evidence": row["assistant_evidence"],
                "automatic_question_surface_state": row["question_surface_state"],
                "automatic_question_presupposition": row["question_presupposition"],
                "caption_crosscheck_state": row["caption_crosscheck_state"],
            }
        )

    state_counts: dict[str, Any] = {}
    question_counts: dict[str, Any] = {}
    for stage, _ in stages:
        state_counts[stage] = {}
        question_counts[stage] = {}
        for spec in FINDINGS:
            subset = [row for row in semantic_rows if row["stage"] == stage and row["finding"] == spec.finding]
            state_counts[stage][spec.finding] = {
                split: {
                    state: Counter(
                        row["assistant_state"]
                        for row in subset
                        if row["source_split"] == split
                    )[state]
                    for state in STATES
                }
                for split in ("source_train", "source_dev", "source_review")
            }
            question_counts[stage][spec.finding] = dict(
                sorted(Counter(row["question_presupposition"] for row in subset).items())
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    records_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic_rows)
    private_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in private_rows)
    caption_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in caption_rows)
    atomic_write(output_dir / "source_records.jsonl", records_text)
    atomic_write(output_dir / "blind_review_key.jsonl", private_text)
    (output_dir / "blind_review_key.jsonl").chmod(0o600)
    atomic_write(output_dir / "caption_crosscheck.jsonl", caption_text)
    csv_path = output_dir / "blind_review.csv"
    temporary_csv = csv_path.with_suffix(csv_path.suffix + f".tmp.{os.getpid()}")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    temporary_csv.replace(csv_path)

    primary_reporting_prior: dict[str, Any] = {}
    for spec in FINDINGS:
        subset = [
            row
            for row in semantic_rows
            if row["stage"] == "alignment"
            and row["finding"] == spec.finding
            and row["generic_alignment_prompt"]
        ]
        primary_reporting_prior[spec.finding] = {
            split: {
                state: Counter(
                    row["assistant_state"]
                    for row in subset
                    if row["source_split"] == split
                )[state]
                for state in STATES
            }
            for split in ("source_train", "source_dev", "source_review")
        }

    stats = {
        "version": VERSION,
        "partition_version": PARTITION_VERSION,
        "repair_note": (
            "v1.1 supersedes v1 for reporting: schema key counting now counts field names; "
            "semantic extraction, partitions, and blind-review selection are unchanged"
        ),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "review_size": args.review_size,
        "evidence_status": "discovery_source_text_statistics_only",
        "scope": (
            "source-only PubMedVision assistant responses; questions audited separately; "
            "captions used only for image-source cross-check; no target data or model output"
        ),
        "findings": [spec.finding for spec in FINDINGS],
        "states": list(STATES),
        "question_labels": list(QUESTION_LABELS),
        "unmentioned_is_negative": False,
        "zero_count_policy": "raw zeros retained; no smoothing, pseudocount, or prior reliability claim",
        "source_index": str(args.source_index.resolve()),
        "source_index_sha256": sha256_file(args.source_index),
        "source_index_schema": source_schema,
        "vqa_files": {
            stage: {"path": str(path.resolve()), "sha256": sha256_file(path), "schema": stage_schema[stage]}
            for stage, path in stages
        },
        "source_split": {
            "unit": "PMC source group shared across both VQA stages",
            "policy": "sha256 group buckets: train 0-69, dev 70-84, review 85-99",
            "group_counts": dict(sorted(Counter((row["source_group"], row["source_split"]) for row in all_units).items(), key=str)),
        },
        "assistant_state_counts_by_stage_finding_split": state_counts,
        "primary_reporting_prior": {
            "scope": (
                "assistant assertions under generic Alignment prompts only; "
                "InstructionTuning excluded from the primary reporting prior"
            ),
            "counts_by_finding_split": primary_reporting_prior,
        },
        "generic_alignment_prompt_counts": dict(
            sorted(
                Counter(
                    row["source_split"]
                    for row in all_units
                    if row["stage"] == "alignment" and row["generic_alignment_prompt"]
                ).items()
            )
        ),
        "question_presupposition_counts_by_stage_finding": question_counts,
        "assistant_response_units": len(all_units),
        "semantic_rows": len(semantic_rows),
        "caption_crosscheck_rows": len(caption_rows),
        "blind_review_rows": len(public_rows),
        "blind_review_private_state_counts": {
            stage: {
                finding: dict(
                    sorted(
                        Counter(
                            row["automatic_assistant_state"]
                            for row in private_rows
                            if row["stage"] == stage and row["finding"] == finding
                        ).items()
                    )
                )
                for finding in [spec.finding for spec in FINDINGS]
            }
            for stage, _ in stages
        },
        "artifacts": {
            "source_records": "source_records.jsonl",
            "blind_review": "blind_review.csv",
            "blind_review_key": "blind_review_key.jsonl",
            "caption_crosscheck": "caption_crosscheck.jsonl",
        },
    }
    # Replace tuple-key diagnostic with compact split counts before JSON serialization.
    stats["source_split"]["response_unit_counts"] = dict(
        sorted(Counter(row["source_split"] for row in all_units).items())
    )
    stats["source_split"].pop("group_counts")
    stats["source_split"]["unique_group_counts"] = dict(
        sorted(
            Counter(
                split for _, split in {(row["source_group"], row["source_split"]) for row in all_units}
            ).items()
        )
    )
    stats["artifact_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in (
            "source_records.jsonl",
            "blind_review.csv",
            "blind_review_key.jsonl",
            "caption_crosscheck.jsonl",
        )
    }
    stats_text = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    atomic_write(output_dir / "stats.json", stats_text)
    artifact_hashes = {
        name: sha256_file(output_dir / name)
        for name in final_names
        if name != "_COMPLETE.json"
    }
    completion = {
        "version": VERSION,
        "status": "complete_source_only_discovery_pack",
        "assistant_response_units": len(all_units),
        "semantic_rows": len(semantic_rows),
        "blind_review_rows": len(public_rows),
        "artifact_sha256": artifact_hashes,
    }
    atomic_write(
        output_dir / "_COMPLETE.json",
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=83017)
    parser.add_argument("--streaming-fallback", action="store_true")
    args = parser.parse_args()
    for path in (args.source_index, args.alignment, args.instruction):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = build(args)
    print(
        json.dumps(
            {
                "version": result["version"],
                "assistant_response_units": result["assistant_response_units"],
                "semantic_rows": result["semantic_rows"],
                "blind_review_rows": result["blind_review_rows"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
