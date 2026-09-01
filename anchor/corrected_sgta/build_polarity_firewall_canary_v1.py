"""Build a target-label-free 32-case polarity-firewall canary on cached BM25 RAG.

No model is invoked. Retrieved reports are converted into deterministic,
claim-neutral terminology cards. Patient-specific finding state, laterality,
temporal comparison, measurements, and recommendations are never copied.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1/rag/bm25/cxr_vishal")
RETRIEVAL = ROOT / "retrieval.jsonl"
NO_CONTEXT = ROOT / "no_context.json"
RAW_RAG = ROOT / "rag.json"
OUT = Path("corrected_runs/polarity_firewall_canary_v1")

PROMPT_PREFIX = (
    "Use the medical image as primary evidence. The following reports are from "
    "different patients and may be irrelevant; do not copy unsupported findings."
)
# Eight strata have sufficient target-label-free positive- and negative-leaning
# retrieval contexts. Order is deterministic and resolves multi-concept queries.
QUERY_GROUPS = {
    "effusion": r"\b(?:effusions?|pleural fluid)\b",
    "cardiac_size": r"\b(?:cardiomegaly|heart size|cardiac silhouette|cardiomediastinal silhouette)\b",
    "airspace": r"\b(?:opacit(?:y|ies)|consolidations?|pneumonia|infiltrates?)\b",
    "edema": r"\b(?:edema|vascular congestion|fluid overload|pulmonary vasculature)\b",
    "atelectasis": r"\b(?:atelectasis|collapse)\b",
    "skeletal": r"\b(?:fractures?|osseous|bones?|spine|ribs?|humerus)\b",
    "mediastinum": r"\b(?:mediastin(?:um|al)|hilar contours?)\b",
    "hyperinflation": r"\b(?:hyperinflation|hyperinflated|emphysema|emphysematous)\b",
}

BASE_NEGATION = (
    r"\b(?:no|without|absent|absence|free of|negative for|"
    r"not (?:seen|identified|present|visualized)|no evidence of|cannot be seen)\b"
)
GROUP_NEGATION = {
    "cardiac_size": r"\b(?:normal|within normal limits|not enlarged)\b",
    "airspace": r"\b(?:clear|no acute disease)\b",
    "edema": r"\b(?:normal|not congested)\b",
    "skeletal": r"\b(?:normal|unremarkable|intact)\b",
    "mediastinum": r"\b(?:normal|unremarkable|within normal)\b",
}
UNCERTAINTY = re.compile(
    r"\b(?:may|might|could|possible|possibly|probable|likely|suggest(?:s|ed)?|"
    r"concerning|cannot exclude|uncertain|questionable)\b",
    re.I,
)

# Fixed knowledge written independently of any target or retrieved patient.
# Every cue is a noun phrase, never a claim about presence or absence.
TERM_CARDS = {
    "pleural_effusion": {
        "patterns": [r"\b(?:pleural effusions?|pleural fluid|hydrothorax)\b"],
        "label": "Pleural effusion",
        "cues": "costophrenic-angle blunting; dependent layering; pleural-space opacity",
        "location_axes": "laterality; costophrenic angle; pleural space",
    },
    "pneumothorax": {
        "patterns": [r"\bpneumothora(?:x|ces)\b"],
        "label": "Pneumothorax",
        "cues": "visceral pleural line; peripheral lung-marking loss; apical or lateral pleural space",
        "location_axes": "laterality; apex; lateral pleural space",
    },
    "cardiac_size": {
        "patterns": [r"\b(?:cardiomegaly|heart size|cardiac silhouette|cardiomediastinal silhouette|cardiac enlargement|enlarged cardiac)\b"],
        "label": "Cardiac silhouette size",
        "cues": "cardiothoracic proportion; projection and inspiration effects",
        "location_axes": "cardiac and cardiomediastinal silhouette",
    },
    "airspace_opacity": {
        "patterns": [r"\b(?:airspace (?:disease|opacity)|opacit(?:y|ies)|consolidations?|infiltrates?|pneumonia)\b"],
        "label": "Airspace opacity",
        "cues": "focal or multifocal density; air bronchogram; silhouette interaction",
        "location_axes": "laterality; lung zone; lobe; perihilar or peripheral distribution",
    },
    "pulmonary_edema": {
        "patterns": [r"\b(?:pulmonary edema|interstitial edema|vascular congestion|fluid overload)\b"],
        "label": "Pulmonary edema pattern",
        "cues": "vascular redistribution; interstitial markings; bilateral alveolar opacity",
        "location_axes": "central versus peripheral; bilateral distribution",
    },
    "atelectasis": {
        "patterns": [r"\b(?:atelectasis|lobar collapse|lung collapse|volume loss)\b"],
        "label": "Atelectatic change",
        "cues": "volume loss; fissural displacement; linear or wedge-shaped opacity",
        "location_axes": "laterality; lobe; lung base",
    },
    "hyperinflation": {
        "patterns": [r"\b(?:hyperinflation|hyperinflated|emphysema|emphysematous)\b"],
        "label": "Hyperinflation pattern",
        "cues": "large lung volumes; diaphragmatic flattening; pulmonary lucency",
        "location_axes": "both lungs; diaphragm",
    },
    "nodule_mass": {
        "patterns": [r"\b(?:pulmonary |lung )?(?:nodules?|masses?|lesions?)\b"],
        "label": "Pulmonary focal lesion",
        "cues": "focal density; margin; size class; multiplicity",
        "location_axes": "laterality; lung zone; lobe",
    },
    "mediastinum": {
        "patterns": [r"\b(?:mediastin(?:um|al)|hilar contours?|lymphadenopathy)\b"],
        "label": "Mediastinal and hilar contour",
        "cues": "contour width; convexity; hilar size and symmetry",
        "location_axes": "right versus left contour; superior mediastinum; hila",
    },
    "skeletal": {
        "patterns": [r"\b(?:fractures?|osseous|bones?|spine|ribs?|clavicle|humerus|degenerative)\b"],
        "label": "Visible osseous structures",
        "cues": "cortical continuity; alignment; deformity; degenerative morphology",
        "location_axes": "ribs; clavicles; shoulders; thoracic spine",
    },
    "support_device": {
        "patterns": [
            r"\b(?:endotracheal tube|ett|enteric tube|ng tube|nasogastric tube|picc|"
            r"central (?:venous )?line|catheter|chest tube|pacemaker|defibrillator|stent)\b"
        ],
        "label": "Support device",
        "cues": "device identity; course; tip relationship to anatomical landmarks",
        "location_axes": "trachea and carina; central veins; pleural space; diaphragm",
    },
    "diaphragm": {
        "patterns": [r"\b(?:diaphragm|hemidiaphragm|costophrenic angle|pleural sinus)\b"],
        "label": "Diaphragm and costophrenic angles",
        "cues": "contour; height; sharpness; adjacent opacity",
        "location_axes": "laterality; diaphragm; costophrenic angle",
    },
}

FORBIDDEN_STATE_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsent\b", r"\bpresent\b",
    r"\bidentified\b", r"\bseen\b", r"\bnoted\b", r"\bunchanged\b",
    r"\bincreased\b", r"\bdecreased\b", r"\bimproved\b", r"\bworsened\b",
    r"\bpatient\b", r"\brecommend\b", r"\bcm\b", r"\bmm\b",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_question(query: str) -> str:
    return query.split("\nAnswer with exactly one of:", 1)[0].strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def query_group(query: str) -> str | None:
    text = source_question(query)
    for group, pattern in QUERY_GROUPS.items():
        if re.search(pattern, text, re.I):
            return group
    return None


def retrieval_polarity(row: dict, group: str) -> dict:
    term = re.compile(QUERY_GROUPS[group], re.I)
    negative = re.compile(BASE_NEGATION + "|" + GROUP_NEGATION.get(group, r"(?!x)x"), re.I)
    counts = Counter()
    evidence = {"positive": [], "negative": [], "uncertain": []}
    for document in row["documents"]:
        for sentence in split_sentences(document["report"]):
            if not term.search(sentence):
                continue
            if negative.search(sentence):
                state = "negative"
            elif UNCERTAINTY.search(sentence):
                state = "uncertain"
            else:
                state = "positive"
            counts[state] += 1
            evidence[state].append({"doc_id": document["doc_id"], "sentence": sentence})
    if counts["positive"] >= 1 and counts["negative"] == 0:
        polarity = "positive"
    elif counts["negative"] >= 1 and counts["positive"] == 0:
        polarity = "negative"
    else:
        polarity = "mixed_or_unresolved"
    return {"polarity": polarity, "counts": dict(counts), "evidence": evidence}


def concepts(text: str) -> list[str]:
    found = []
    for key, spec in TERM_CARDS.items():
        if any(re.search(pattern, text, re.I) for pattern in spec["patterns"]):
            found.append(key)
    return found


def neutral_card(document: dict, group: str) -> tuple[str, dict]:
    raw = document["report"]
    recognized = concepts(raw)
    found = list(recognized)
    fallback = False
    fallback_map = {
        "effusion": "pleural_effusion", "cardiac_size": "cardiac_size",
        "airspace": "airspace_opacity", "edema": "pulmonary_edema",
        "atelectasis": "atelectasis", "skeletal": "skeletal",
        "mediastinum": "mediastinum", "hyperinflation": "hyperinflation",
    }
    if not found:
        fallback = True
        found = [fallback_map[group]]
    preferred = fallback_map[group]
    found.sort(key=lambda key: (key != preferred, list(TERM_CARDS).index(key)))
    retained = found
    lines = ["Neutral terms — state unresolved from retrieval."]
    for key in retained:
        spec = TERM_CARDS[key]
        lines.append(f"- {spec['label']}: {spec['cues']} @ {spec['location_axes']}")
    card = "\n".join(lines)
    violations = [pattern for pattern in FORBIDDEN_STATE_PATTERNS if re.search(pattern, card, re.I)]
    return card, {
        "doc_id": document["doc_id"],
        "raw_sha256": document["sha256"],
        "raw_chars": len(raw),
        "card_chars": len(card),
        "recognized_concepts": recognized,
        "retained_concepts": retained,
        "dropped_concepts": [key for key in recognized if key not in retained],
        "fallback_to_query_term": fallback,
        "forbidden_state_pattern_hits": violations,
    }


def group_term_key(group: str) -> str:
    return {
        "effusion": "pleural_effusion", "cardiac_size": "cardiac_size",
        "airspace": "airspace_opacity", "edema": "pulmonary_edema",
        "atelectasis": "atelectasis", "skeletal": "skeletal",
        "mediastinum": "mediastinum", "hyperinflation": "hyperinflation",
    }[group]


def query_term_card(group: str) -> str:
    spec = TERM_CARDS[group_term_key(group)]
    return (
        "Neutral query term — state unresolved from retrieval.\n"
        f"- {spec['label']}: {spec['cues']} @ {spec['location_axes']}"
    )


def length_match_card(card: str, raw_report: str) -> tuple[str, dict]:
    """Match exact characters and whitespace tokens without adding assertions.

    Long neutral cards are prefix-truncated at token boundaries. Short cards keep
    as much content as possible and use trailing whitespace for exact character
    matching. Any missing token slots use the neutral placeholder ``term`` (or
    ``x`` if required by a very short source). This is deliberately audited as a
    formatting control, not presented as semantic preservation.
    """
    source_tokens = raw_report.split()
    target_tokens = len(source_tokens)
    target_chars = len(raw_report)
    card_tokens = card.split()
    if target_tokens == 0:
        return " " * target_chars, {
            "target_chars": target_chars, "matched_chars": target_chars,
            "target_whitespace_tokens": 0, "matched_whitespace_tokens": 0,
            "retained_card_tokens": 0, "filler_tokens": 0, "trailing_space_padding": target_chars,
        }

    filler_cycle = ["form", "axis", "zone", "view", "size", "edge", "path", "site", "cues"]
    chosen = None
    for kept in range(min(len(card_tokens), target_tokens), -1, -1):
        missing = target_tokens - kept
        candidate = card_tokens[:kept] + [filler_cycle[i % len(filler_cycle)] for i in range(missing)]
        text = " ".join(candidate)
        if len(text) <= target_chars:
            chosen = (candidate, kept, "neutral_glossary_cycle")
            break
    if chosen is None:
        # One-character placeholders guarantee feasibility whenever a string has
        # at least as many characters as whitespace tokens, true for normal text.
        for kept in range(min(len(card_tokens), target_tokens), -1, -1):
            missing = target_tokens - kept
            candidate = card_tokens[:kept] + ["x"] * missing
            text = " ".join(candidate)
            if len(text) <= target_chars:
                chosen = (candidate, kept, "x")
                break
    if chosen is None:
        raise RuntimeError("cannot satisfy deterministic character/token match")
    candidate, kept, filler = chosen
    text = " ".join(candidate)
    padding = target_chars - len(text)
    text += " " * padding
    assert len(text) == target_chars
    assert len(text.split()) == target_tokens
    return text, {
        "target_chars": target_chars,
        "matched_chars": len(text),
        "target_whitespace_tokens": target_tokens,
        "matched_whitespace_tokens": len(text.split()),
        "retained_card_tokens": kept,
        "original_card_tokens": len(card_tokens),
        "filler_tokens": target_tokens - kept,
        "filler_lexeme": filler,
        "trailing_space_padding": padding,
    }


def prompt(prefix: str, contexts: list[str], query: str) -> str:
    body = "[none]" if not contexts else "\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, 1))
    return f"{prefix}\nRetrieved reports:\n{body}\nQuestion:\n{query}"


def safe_metadata(row: dict) -> dict:
    allowed = [
        "id", "qid", "img_name", "patient_id", "source_question", "task",
        "prompt_contract", "dataset", "source_row", "source_qid", "question_type",
        "source_question_type", "choices",
    ]
    return {key: row[key] for key in allowed if key in row}


def main() -> None:
    retrieval = read_jsonl(RETRIEVAL)
    no_context_source = json.loads(NO_CONTEXT.read_text())
    raw_rag_source = json.loads(RAW_RAG.read_text())
    safe_by_qid = {str(row["qid"]): safe_metadata(row) for row in no_context_source}
    source_none_question = {str(row["qid"]): row["question"] for row in no_context_source}
    source_raw_question = {str(row["qid"]): row["question"] for row in raw_rag_source}

    candidates = []
    for row in retrieval:
        if "Answer with exactly one of: Yes, No, Uncertain." not in row["query"]:
            continue
        group = query_group(row["query"])
        if group is None:
            continue
        state = retrieval_polarity(row, group)
        if state["polarity"] not in {"positive", "negative"}:
            continue
        candidates.append({"row": row, "group": group, "state": state})

    selected = []
    used = set()
    for group in QUERY_GROUPS:
        for polarity in ("positive", "negative"):
            pool = [
                item for item in candidates
                if item["group"] == group and item["state"]["polarity"] == polarity
            ]
            pool.sort(key=lambda item: digest(f"polarity-firewall-v1|{group}|{polarity}|{item['row']['sample_id']}"))
            chosen = []
            for item in pool:
                qid = str(item["row"]["sample_id"])
                if qid not in used:
                    chosen.append(item)
                    used.add(qid)
                if len(chosen) == 2:
                    break
            if len(chosen) != 2:
                raise RuntimeError(f"insufficient canary candidates for {group}/{polarity}: {len(chosen)}")
            selected.extend(chosen)

    raw_arm, none_arm, depolarized_arm = [], [], []
    length_matched_arm, query_term_arm, audits, length_audits = [], [], [], []
    all_failures = []
    risk_pool = []
    for item in selected:
        row, group, state = item["row"], item["group"], item["state"]
        qid = str(row["sample_id"])
        metadata = safe_by_qid[qid]
        raw_context = [doc["report"] for doc in row["documents"]]
        cards, document_audits = [], []
        for document in row["documents"]:
            card, audit = neutral_card(document, group)
            cards.append(card)
            document_audits.append(audit)
            risk_pool.append({
                "question_id": qid,
                **audit,
                "raw_excerpt": document["report"][:800],
                "card": card,
                "length_ratio": len(card) / max(1, len(document["report"])),
            })
            if audit["fallback_to_query_term"] or audit["forbidden_state_pattern_hits"]:
                all_failures.append({"question_id": qid, **audit, "raw_excerpt": document["report"][:500], "card": card})

        matched_cards, sample_length_audits = [], []
        for card, document in zip(cards, row["documents"]):
            matched, match_audit = length_match_card(card, document["report"])
            matched_cards.append(matched)
            sample_length_audits.append({"doc_id": document["doc_id"], **match_audit})

        common = {
            **metadata,
            "selection_group": group,
            "selection_retrieval_polarity": state["polarity"],
            "selection_is_target_label_free": True,
            "retrieved_doc_ids": [doc["doc_id"] for doc in row["documents"]],
        }
        raw_arm.append({**common, "context_condition": "raw_rag", "question": prompt(PROMPT_PREFIX, raw_context, row["query"])})
        none_arm.append({**common, "context_condition": "no_context", "retrieved_doc_ids": [], "question": prompt(PROMPT_PREFIX, [], row["query"])})
        depolarized_arm.append({**common, "context_condition": "depolarized_rag", "question": prompt(PROMPT_PREFIX, cards, row["query"])})
        length_matched_arm.append({
            **common,
            "context_condition": "length_matched_depolarized_rag",
            "question": prompt(PROMPT_PREFIX, matched_cards, row["query"]),
        })
        query_term_arm.append({
            **common,
            "context_condition": "query_term_only_neutral_rag",
            "question": prompt(PROMPT_PREFIX, [query_term_card(group)], row["query"]),
        })
        length_audits.append({"question_id": qid, "documents": sample_length_audits})
        audits.append({
            "question_id": qid,
            "group": group,
            "retrieval_polarity": state,
            "documents": document_audits,
            "raw_context_chars": sum(len(x) for x in raw_context),
            "depolarized_context_chars": sum(len(x) for x in cards),
        })

    def contains_forbidden_label(obj) -> bool:
        return any(key in obj for key in ("answer", "gt_ans", "ground_truth", "label"))

    all_document_audits = [doc for sample in audits for doc in sample["documents"]]
    lossy_examples = sorted(
        [
            {"question_id": sample["question_id"], **doc}
            for sample in audits for doc in sample["documents"] if doc["dropped_concepts"]
        ],
        key=lambda row: (-len(row["dropped_concepts"]), row["question_id"], row["doc_id"]),
    )
    risk_examples = []
    seen_risk = set()
    ranked_risks = [
        ("largest_length_expansion", sorted(risk_pool, key=lambda x: -x["length_ratio"])),
        ("largest_information_compression", sorted(risk_pool, key=lambda x: x["length_ratio"])),
        ("broadest_concept_priming", sorted(risk_pool, key=lambda x: -len(x["recognized_concepts"]))),
    ]
    for risk_type, ranked in ranked_risks:
        for row in ranked:
            key = (row["question_id"], row["doc_id"])
            if key in seen_risk:
                continue
            risk_examples.append({**row, "risk_type": risk_type})
            seen_risk.add(key)
            break
    failure_examples = all_failures + risk_examples
    raw_state_counts = Counter()
    for sample in audits:
        raw_state_counts.update(sample["retrieval_polarity"]["counts"])
    arm_rows = {
        "raw_rag": raw_arm,
        "no_context": none_arm,
        "depolarized_rag": depolarized_arm,
        "length_matched_depolarized_rag": length_matched_arm,
        "query_term_only_neutral_rag": query_term_arm,
    }
    arm_qids = {name: [str(row["qid"]) for row in rows] for name, rows in arm_rows.items()}
    reference_qids = arm_qids["raw_rag"]

    def outside_context(question: str) -> tuple[str, str]:
        prefix, remainder = question.split("\nRetrieved reports:\n", 1)
        _, suffix = remainder.rsplit("\nQuestion:\n", 1)
        return prefix, suffix

    def context_body(question: str) -> str:
        _, remainder = question.split("\nRetrieved reports:\n", 1)
        body, _ = remainder.rsplit("\nQuestion:\n", 1)
        return body

    raw_outside = [outside_context(row["question"]) for row in raw_arm]
    prompt_drift = {
        name: sum(outside_context(row["question"]) != raw_outside[i] for i, row in enumerate(rows))
        for name, rows in arm_rows.items()
    }
    context_state_hits = {
        name: sum(
            len(re.findall(pattern, context_body(row["question"]), re.I))
            for row in rows for pattern in FORBIDDEN_STATE_PATTERNS
        )
        for name, rows in arm_rows.items()
    }
    flat_length_audit = [doc for sample in length_audits for doc in sample["documents"]]
    full_prompt_length_matches = [
        {
            "question_id": str(raw_arm[i]["qid"]),
            "raw_chars": len(raw_arm[i]["question"]),
            "matched_chars": len(length_matched_arm[i]["question"]),
            "raw_whitespace_tokens": len(raw_arm[i]["question"].split()),
            "matched_whitespace_tokens": len(length_matched_arm[i]["question"].split()),
        }
        for i in range(len(raw_arm))
    ]
    result = {
        "status": "completed_cpu_manifest_only_no_model_run",
        "protocol": "polarity-firewall-canary-v1",
        "source_artifacts": {
            "retrieval": str(RETRIEVAL),
            "retrieval_sha256": file_digest(RETRIEVAL),
            "safe_metadata_source": str(NO_CONTEXT),
            "safe_metadata_source_sha256": file_digest(NO_CONTEXT),
            "raw_rag_source": str(RAW_RAG),
            "raw_rag_source_sha256": file_digest(RAW_RAG),
        },
        "selection": {
            "n": len(selected),
            "contract": "8 query concepts x 2 positive-retrieval x 2 negative-retrieval; SHA256 deterministic selection",
            "uses_target_label": False,
            "group_polarity_counts": dict(Counter(f"{x['group']}:{x['state']['polarity']}" for x in selected)),
        },
        "transformation": {
            "mode": "controlled-ontology reconstruction, not lexical negation deletion",
            "preserved": "clinical term definitions, generic visual cues, and abstract assessment regions",
            "removed": "other-patient state, laterality value, temporal state, uncertainty, measurements, history, and management",
            "fallback": "query-concept neutral card when a retrieved document has no recognized controlled term",
            "term_cards": TERM_CARDS,
            "forbidden_state_patterns": FORBIDDEN_STATE_PATTERNS,
        },
        "coverage_fidelity": {
            "samples": len(audits),
            "documents": len(all_document_audits),
            "documents_with_recognized_report_concept": sum(not x["fallback_to_query_term"] for x in all_document_audits),
            "documents_falling_back_to_query_term": sum(x["fallback_to_query_term"] for x in all_document_audits),
            "cards_with_forbidden_state_pattern": sum(bool(x["forbidden_state_pattern_hits"]) for x in all_document_audits),
            "recognized_report_concepts": sum(len(x["recognized_concepts"]) for x in all_document_audits),
            "retained_report_concepts": sum(
                len(set(x["recognized_concepts"]) & set(x["retained_concepts"])) for x in all_document_audits
            ),
            "documents_with_secondary_concept_drops": sum(bool(x["dropped_concepts"]) for x in all_document_audits),
            "mean_raw_chars": sum(x["raw_chars"] for x in all_document_audits) / len(all_document_audits),
            "mean_card_chars": sum(x["card_chars"] for x in all_document_audits) / len(all_document_audits),
            "card_to_raw_char_ratio": sum(x["card_chars"] for x in all_document_audits) / sum(x["raw_chars"] for x in all_document_audits),
            "raw_query_concept_sentence_states": dict(raw_state_counts),
            "post_transform_forbidden_state_hits": sum(
                len(x["forbidden_state_pattern_hits"]) for x in all_document_audits
            ),
        },
        "leakage_audit": {
            "forbidden_target_fields": ["answer", "gt_ans", "ground_truth", "label"],
            "forbidden_fields_present_by_arm": {
                "raw_rag": sum(contains_forbidden_label(x) for x in raw_arm),
                "no_context": sum(contains_forbidden_label(x) for x in none_arm),
                "depolarized_rag": sum(contains_forbidden_label(x) for x in depolarized_arm),
                "length_matched_depolarized_rag": sum(contains_forbidden_label(x) for x in length_matched_arm),
                "query_term_only_neutral_rag": sum(contains_forbidden_label(x) for x in query_term_arm),
            },
            "note": "Selection and transformation use query text and retrieved reports only; target values are never accessed.",
        },
        "matched_arm_audit": {
            "raw_rag_questions_exact_match_to_frozen_source": sum(
                row["question"] == source_raw_question[str(row["qid"])] for row in raw_arm
            ),
            "no_context_questions_exact_match_to_frozen_source": sum(
                row["question"] == source_none_question[str(row["qid"])] for row in none_arm
            ),
            "expected_per_arm": len(selected),
            "only_intended_prompt_change": "retrieved context body; neutral-card instructions are contained inside that body",
            "all_arm_qid_order_identical": all(qids == reference_qids for qids in arm_qids.values()),
            "per_arm_n": {name: len(rows) for name, rows in arm_rows.items()},
            "outside_context_prompt_drift_count": prompt_drift,
            "context_forbidden_state_pattern_hits": context_state_hits,
        },
        "length_matched_audit": {
            "token_definition": "Python whitespace split proxy; model-tokenizer equality is not claimed",
            "documents": len(flat_length_audit),
            "exact_character_matches": sum(x["target_chars"] == x["matched_chars"] for x in flat_length_audit),
            "exact_whitespace_token_matches": sum(
                x["target_whitespace_tokens"] == x["matched_whitespace_tokens"] for x in flat_length_audit
            ),
            "mean_retained_card_token_fraction": sum(
                x["retained_card_tokens"] / max(1, x["original_card_tokens"]) for x in flat_length_audit
            ) / len(flat_length_audit),
            "total_filler_tokens": sum(x["filler_tokens"] for x in flat_length_audit),
            "total_trailing_space_padding": sum(x["trailing_space_padding"] for x in flat_length_audit),
            "exact_full_prompt_character_matches": sum(
                x["raw_chars"] == x["matched_chars"] for x in full_prompt_length_matches
            ),
            "exact_full_prompt_whitespace_token_matches": sum(
                x["raw_whitespace_tokens"] == x["matched_whitespace_tokens"] for x in full_prompt_length_matches
            ),
            "expected_full_prompts": len(full_prompt_length_matches),
            "warning": "Exact char matching partly uses trailing spaces; this controls prompt size but is not semantic content.",
        },
        "hard_failure_count": len(all_failures),
        "lossy_document_count": len(lossy_examples),
        "documented_risk_example_count": len(risk_examples),
        "limitations": [
            "A neutral term card can still prime a disease concept even though its polarity is removed.",
            "Controlled cues are radiographic heuristics, not complete clinical definitions.",
            "Report-concept extraction is deterministic lexical matching and can miss synonyms.",
            "This artifact validates transformation and balance only; no accuracy claim exists before generation.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in arm_rows.items():
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    result["output_sha256"] = {
        name: file_digest(OUT / f"{name}.json") for name in arm_rows
    }
    with (OUT / "transformation_audit.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row) + "\n")
    with (OUT / "length_matching_audit.jsonl").open("w") as handle:
        for row in length_audits:
            handle.write(json.dumps(row) + "\n")
    (OUT / "failure_examples.json").write_text(json.dumps(failure_examples[:32], indent=2) + "\n")
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
