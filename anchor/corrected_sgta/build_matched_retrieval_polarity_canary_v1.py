"""Build a label-blind four-arm retrieval-polarity causal canary.

CPU/manifest only.  Present and absent reports must describe the same frozen
finding, come from different patients, and differ in word length by at most
10%.  They are greedily matched by corpus TF-IDF cosine without target labels.
Neutral removes every claim-bearing sentence from the present report;
random-deletion removes the same number of words elsewhere while preserving
the claim.  Claims that cannot satisfy every hard gate are reported and not
relaxed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from anchor.corrected_sgta.analyze_retrieval_polarity_transplantation_v1 import (
    local_clause,
    mention_assertion,
    window_by_tokens,
)


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
CORPUS = ROOT / "rag/combined_corpus/corpus.jsonl"
VINDR = Path(
    "corrected_runs/vindr_v2/clinical_presupposition_huatuo_generation_v1/selected_manifest.jsonl"
)
CXR = ROOT / "rag/bm25/cxr_vishal/no_context.json"
OUT_DIR = Path("corrected_runs/matched_retrieval_polarity_canary_v1")
CANARY_OUT = OUT_DIR / "canary.jsonl"
PAIRS_OUT = OUT_DIR / "matched_pairs.jsonl"
RESULT_OUT = OUT_DIR / "result.json"
SEED = 20260810
MAX_PER_FINDING_PER_COHORT = 8
MAX_LENGTH_GAP = 0.10
ARMS = ("present", "absent", "neutral", "random_deletion")


# Each entry is an ordered tuple of (mention regex, intrinsic finding sign).
# Negative direct descriptions have intrinsic -1.  Local NegBio-style cues can
# reverse either sign.  ``other_lesion`` is deliberately unsupported rather
# than given a vague fallback.
FINDINGS: dict[str, tuple[tuple[str, int], ...]] = {
    "aortic_enlargement": (
        (r"\b(?:aortic enlargement|enlarged (?:thoracic )?aorta|aorta is dilated|dilatation of the aorta|aortic aneurysm)\b", 1),
        (r"\b(?:aorta|aortic contour)\b.{0,18}\b(?:normal|unremarkable)\b", -1),
    ),
    "cardiomegaly": (
        (r"\b(?:cardiomegaly|cardiac enlargement|enlarged cardiac silhouette|heart is enlarged|enlarged heart)\b", 1),
        (r"\b(?:heart size|cardiac silhouette|cardiomediastinal silhouette)\b.{0,24}\b(?:normal|within normal limits|not enlarged)\b", -1),
    ),
    "lung_opacity": ((r"\b(?:lung|pulmonary|parenchymal|airspace) opacit(?:y|ies)\b", 1),),
    "nodule_mass": ((r"\b(?:pulmonary |lung )?(?:nodules?|masses?)\b", 1),),
    "other_lesion": (),
    "pleural_effusion": ((r"\b(?:pleural effusions?|pleural fluid|hydrothorax)\b", 1),),
    "pleural_thickening": ((r"\bpleural thicken(?:ing|ed)\b", 1),),
    "pulmonary_fibrosis": ((r"\b(?:pulmonary fibrosis|fibrotic (?:change|changes|disease)|lung fibrosis)\b", 1),),
    "pneumothorax": ((r"\bpneumothora(?:x|ces)\b", 1),),
    "pneumonia": ((r"\bpneumonia\b", 1),),
    "focal_consolidation": ((r"\b(?:focal )?consolidation\b", 1),),
    "atelectasis": ((r"\batelecta(?:sis|tic)\b|\blung collapse\b", 1),),
    "pulmonary_edema": ((r"\b(?:pulmonary|interstitial) (?:edema|oedema)\b|\bfluid overload\b", 1),),
    "emphysema": ((r"(?<!subcutaneous )\bemphysema\b|\bhyperinflat(?:ed|ion)\b", 1),),
    "fracture": ((r"\bfractur(?:e|ed|es)\b", 1),),
}

DISPLAY = {
    "aortic_enlargement": "aortic enlargement",
    "cardiomegaly": "cardiomegaly",
    "lung_opacity": "lung opacity",
    "nodule_mass": "a pulmonary nodule or mass",
    "other_lesion": "another focal lesion",
    "pleural_effusion": "pleural effusion",
    "pleural_thickening": "pleural thickening",
    "pulmonary_fibrosis": "pulmonary fibrosis",
    "pneumothorax": "pneumothorax",
    "pneumonia": "pneumonia",
    "focal_consolidation": "focal consolidation",
    "atelectasis": "atelectasis",
    "pulmonary_edema": "pulmonary edema",
    "emphysema": "emphysema",
    "fracture": "a fracture",
}

CXR_QUERY_PATTERNS = (
    ("pneumothorax", r"\bpneumothorax\b"),
    ("pleural_effusion", r"\b(?:pleural )?effusion\b"),
    ("pulmonary_edema", r"\bpulmonary edema\b"),
    ("focal_consolidation", r"\b(?:focal )?consolidation\b"),
    ("pneumonia", r"\bpneumonia\b"),
    ("atelectasis", r"\batelectasis\b"),
    ("cardiomegaly", r"\bcardiomegaly\b|\b(?:heart|cardiac silhouette) (?:is )?enlarged\b"),
    ("emphysema", r"\bemphysema\b"),
    ("nodule_mass", r"\b(?:pulmonary |lung )?(?:nodule|mass)\b"),
    ("pulmonary_fibrosis", r"\b(?:pulmonary )?fibrosis\b"),
    ("lung_opacity", r"\b(?:lung|pulmonary|parenchymal|airspace) opacit(?:y|ies)\b"),
    ("fracture", r"\bfractur(?:e|ed)\b"),
)

# The simple donor polarity does not encode attributes.  These questions are
# excluded by semantics before matching, regardless of any target label.
COMPLEX_QUERY = re.compile(
    r"\b(?:normal|clear|intact|change|changed|compared|interval|increase|decrease|"
    r"reduced|improved|worsened|resolved|new|position|positioned|placement|correct|"
    r"adjust|large|small|mild|moderate|severe|bilateral|unilateral|right|left|"
    r"upper|lower|basilar|apical)\b",
    re.I,
)

PROMPT_PREFIX = (
    "Use the medical image as primary evidence. The following report is from a "
    "different patient and may be irrelevant; do not copy unsupported findings."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def doc_patient(row: dict[str, Any]) -> str:
    return str(row.get("patient_id") or row.get("study_id") or row.get("image_id") or row["doc_id"])


def sentence_assertions(sentence: str, finding: str) -> list[str]:
    output = []
    lowered = sentence.lower()
    for pattern, intrinsic in FINDINGS[finding]:
        for match in re.finditer(pattern, lowered, re.I):
            clause, start, end = local_clause(lowered, match.start(), match.end())
            # NegBio-style coordination boundary: an earlier ``no X with Y``
            # negates X, not Y.  Do not carry a cue across ``with``.
            with_boundary = clause.rfind(" with ", 0, start)
            if with_boundary >= 0:
                offset = with_boundary + len(" with ")
                clause, start, end = clause[offset:], start - offset, end - offset
            pre, post = window_by_tokens(clause, start, end, before=6, after=5)
            vicinity = f"{pre} {clause[start:end]} {post}"
            if re.search(r"\b(?:cannot|can't|could not) be excluded\b", vicinity, re.I):
                output.append("unknown")
            elif re.search(r"\bresolved\b", pre, re.I) or re.search(r"\b(?:has|have|had)?\s*resolved\b", post, re.I):
                output.append("negative" if intrinsic > 0 else "positive")
            elif re.search(r"\bnot typical of\b", pre, re.I):
                output.append("negative" if intrinsic > 0 else "positive")
            else:
                output.append(mention_assertion(clause, start, end, intrinsic))
    return output


def report_state(report: str, finding: str) -> tuple[str, list[dict[str, str]]]:
    assertions = []
    evidence = []
    for sentence in split_sentences(report):
        local = sentence_assertions(sentence, finding)
        for state in local:
            assertions.append(state)
            evidence.append({"assertion": state, "sentence": sentence})
    observed = set(assertions)
    if not assertions:
        state = "neutral"
    elif observed <= {"positive"}:
        state = "present"
    elif observed <= {"negative"}:
        state = "absent"
    else:
        state = "unresolved"
    return state, evidence


def neutral_and_random(report: str, finding: str, seed_key: str) -> dict[str, Any] | None:
    sentences = split_sentences(report)
    claim_mask = [bool(sentence_assertions(sentence, finding)) for sentence in sentences]
    if not any(claim_mask) or all(claim_mask):
        return None
    claim_word_count = sum(len(words(sentence)) for sentence, claim in zip(sentences, claim_mask) if claim)
    nonclaim_tokens = [
        (sentence_index, token_index)
        for sentence_index, (sentence, claim) in enumerate(zip(sentences, claim_mask))
        if not claim
        for token_index, _ in enumerate(sentence.split())
    ]
    if claim_word_count <= 0 or len(nonclaim_tokens) < claim_word_count:
        return None
    neutral = " ".join(sentence for sentence, claim in zip(sentences, claim_mask) if not claim)
    if report_state(neutral, finding)[0] != "neutral":
        return None

    rng = np.random.default_rng(int(stable_hash(seed_key)[:16], 16))
    chosen = set(
        tuple(nonclaim_tokens[index])
        for index in rng.choice(len(nonclaim_tokens), size=claim_word_count, replace=False)
    )
    rebuilt = []
    for sentence_index, sentence in enumerate(sentences):
        tokens = sentence.split()
        kept = [token for token_index, token in enumerate(tokens) if (sentence_index, token_index) not in chosen]
        if kept:
            rebuilt.append(" ".join(kept))
    random_deletion = " ".join(rebuilt)
    if report_state(random_deletion, finding)[0] != "present":
        return None
    if len(words(neutral)) != len(words(random_deletion)):
        return None
    return {
        "neutral": neutral,
        "random_deletion": random_deletion,
        "claim_sentence_count": int(sum(claim_mask)),
        "deleted_words": claim_word_count,
    }


def cxr_finding(question: str) -> str | None:
    if COMPLEX_QUERY.search(question):
        return None
    for finding, pattern in CXR_QUERY_PATTERNS:
        if re.search(pattern, question, re.I):
            return finding
    return None


def build_queries() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vin_rows = read_jsonl(VINDR)
    for row in vin_rows:
        for finding in row["claim_names"]:
            query = {
                "cohort": "vindr",
                "source_id": str(row["item_id"]),
                "image": str(row["dicom_relpath"]),
                "query_patient": f"vindr:{row['image_id']}",
                "finding": finding,
                "source_question": f"Is {DISPLAY[finding]} present on this chest X-ray? Answer with exactly one of: Yes, No, Uncertain.",
            }
            queries[finding].append(query)

    cxr_rows = json.loads(CXR.read_text())
    cxr_semantic_counts = Counter()
    for row in cxr_rows:
        # Do not read ``answer`` or any outcome field.
        if row.get("source_question_type") != "binary" or str(row.get("choices", "")).strip():
            cxr_semantic_counts["non_binary_or_choices"] += 1
            continue
        source_question = str(row.get("source_question", "")).split("\nAnswer with exactly one of:", 1)[0].strip()
        finding = cxr_finding(source_question)
        if finding is None:
            cxr_semantic_counts["unmapped_or_complex"] += 1
            continue
        cxr_semantic_counts[f"eligible:{finding}"] += 1
        queries[finding].append({
            "cohort": "cxr_vishal",
            "source_id": str(row["qid"]),
            "image": str(row["img_name"]),
            "query_patient": str(row.get("patient_id") or row.get("img_name") or row["qid"]),
            "finding": finding,
            "source_question": source_question + "\nAnswer with exactly one of: Yes, No, Uncertain.",
        })

    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inventory = {}
    for finding, local in queries.items():
        by_cohort = defaultdict(list)
        for row in local:
            by_cohort[row["cohort"]].append(row)
        inventory[finding] = {}
        for cohort, candidates in by_cohort.items():
            ordered = sorted(candidates, key=lambda row: stable_hash(f"query:{finding}:{cohort}:{row['source_id']}"))
            chosen = ordered[:MAX_PER_FINDING_PER_COHORT]
            selected[finding].extend(chosen)
            inventory[finding][cohort] = {"eligible": len(candidates), "selected_before_donor_gate": len(chosen)}
    return selected, {"by_finding_cohort": inventory, "cxr_semantic_gate": dict(cxr_semantic_counts)}


def pair_candidates(
    corpus: list[dict[str, Any]], matrix: Any, finding: str, states: list[str], transforms: dict[int, dict[str, Any]]
) -> tuple[list[tuple[int, int, float, float]], dict[str, Any]]:
    positive = [index for index, state in enumerate(states) if state == "present" and index in transforms]
    negative = [index for index, state in enumerate(states) if state == "absent"]
    if not positive or not negative:
        return [], {"present_transformable": len(positive), "absent": len(negative), "strict_pair_edges": 0}
    similarity = (matrix[positive] @ matrix[negative].T).toarray()
    candidates = []
    strict_edges = 0
    for p_row, p_index in enumerate(positive):
        p_len = len(words(corpus[p_index]["report"]))
        p_patient = doc_patient(corpus[p_index])
        for n_col, n_index in enumerate(negative):
            n_len = len(words(corpus[n_index]["report"]))
            gap = abs(p_len - n_len) / max(p_len, n_len)
            if gap > MAX_LENGTH_GAP or p_patient == doc_patient(corpus[n_index]):
                continue
            strict_edges += 1
            candidates.append((p_index, n_index, float(similarity[p_row, n_col]), float(gap)))
    candidates.sort(key=lambda item: (-item[2], item[3], corpus[item[0]]["doc_id"], corpus[item[1]]["doc_id"]))
    return candidates, {"present_transformable": len(positive), "absent": len(negative), "strict_pair_edges": strict_edges}


def prompt(context: str, question: str) -> str:
    return f"{PROMPT_PREFIX}\nRetrieved report:\n{context}\nQuestion:\n{question}"


def main() -> None:
    corpus = read_jsonl(CORPUS)
    queries, query_audit = build_queries()
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2,
        max_features=20_000, norm="l2", dtype=np.float64,
    )
    matrix = vectorizer.fit_transform([row["report"] for row in corpus])

    canary_rows = []
    pair_rows = []
    pool_audit = {}
    unmatched = defaultdict(Counter)
    for finding in sorted(queries):
        patterns = FINDINGS.get(finding, ())
        if not patterns:
            pool_audit[finding] = {
                "status": "unsupported_finding_no_relaxation",
                "reason": "no specific corpus lexical definition",
                "query_n": len(queries[finding]),
            }
            for query in queries[finding]:
                unmatched[finding][query["cohort"]] += 1
            continue
        states = []
        transforms = {}
        evidence = {}
        for index, document in enumerate(corpus):
            state, local_evidence = report_state(document["report"], finding)
            states.append(state)
            if local_evidence:
                evidence[index] = local_evidence
            if state == "present":
                transformed = neutral_and_random(document["report"], finding, f"{finding}:{document['doc_id']}")
                if transformed is not None:
                    transforms[index] = transformed
        candidates, local_audit = pair_candidates(corpus, matrix, finding, states, transforms)
        local_audit.update({
            "status": "strict_gates_applied_no_relaxation",
            "corpus_state_counts": dict(Counter(states)),
            "query_n": len(queries[finding]),
        })
        pool_audit[finding] = local_audit

        used_present = set()
        used_absent = set()
        for query in queries[finding]:
            chosen = None
            for candidate in candidates:
                present_index, absent_index, similarity, length_gap = candidate
                if present_index in used_present or absent_index in used_absent:
                    continue
                donor_patients = {doc_patient(corpus[present_index]), doc_patient(corpus[absent_index])}
                if query["query_patient"] in donor_patients:
                    continue
                chosen = candidate
                break
            if chosen is None:
                unmatched[finding][query["cohort"]] += 1
                continue
            present_index, absent_index, similarity, length_gap = chosen
            used_present.add(present_index)
            used_absent.add(absent_index)
            present_doc, absent_doc = corpus[present_index], corpus[absent_index]
            transformed = transforms[present_index]
            pair_id = stable_hash(
                f"matched-polarity-canary-v1:{finding}:{query['cohort']}:{query['source_id']}:{present_doc['doc_id']}:{absent_doc['doc_id']}"
            )[:20]
            pair = {
                "pair_id": pair_id,
                "cohort": query["cohort"],
                "source_id": query["source_id"],
                "finding": finding,
                "present_doc_id": present_doc["doc_id"],
                "absent_doc_id": absent_doc["doc_id"],
                "present_patient": doc_patient(present_doc),
                "absent_patient": doc_patient(absent_doc),
                "query_patient": query["query_patient"],
                "present_words": len(words(present_doc["report"])),
                "absent_words": len(words(absent_doc["report"])),
                "present_absent_length_gap": length_gap,
                "tfidf_cosine": similarity,
                "neutral_words": len(words(transformed["neutral"])),
                "random_deletion_words": len(words(transformed["random_deletion"])),
                "deleted_words": transformed["deleted_words"],
                "claim_sentence_count": transformed["claim_sentence_count"],
                "patient_disjoint": len({query["query_patient"], doc_patient(present_doc), doc_patient(absent_doc)}) == 3,
                "present_evidence": evidence[present_index],
                "absent_evidence": evidence.get(absent_index, []),
            }
            pair_rows.append(pair)
            contexts = {
                "present": present_doc["report"],
                "absent": absent_doc["report"],
                "neutral": transformed["neutral"],
                "random_deletion": transformed["random_deletion"],
            }
            for arm in ARMS:
                canary_rows.append({
                    "canary_id": f"{pair_id}:{arm}",
                    "pair_id": pair_id,
                    "arm": arm,
                    "cohort": query["cohort"],
                    "source_id": query["source_id"],
                    "image": query["image"],
                    "finding": finding,
                    "source_question": query["source_question"],
                    "question": prompt(contexts[arm], query["source_question"]),
                    "context": contexts[arm],
                    "context_sha256": stable_hash(contexts[arm]),
                    "selection_uses_target_label": False,
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANARY_OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in canary_rows))
    PAIRS_OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows))

    similarities = [row["tfidf_cosine"] for row in pair_rows]
    length_gaps = [row["present_absent_length_gap"] for row in pair_rows]
    expected_context_state = {
        "present": "present", "absent": "absent",
        "neutral": "neutral", "random_deletion": "present",
    }
    observed_arm_states = Counter()
    arm_state_mismatches = []
    for row in canary_rows:
        observed, _ = report_state(row["context"], row["finding"])
        observed_arm_states[f"{row['arm']}:{observed}"] += 1
        if observed != expected_context_state[row["arm"]]:
            arm_state_mismatches.append(row["canary_id"])
    pair_arm_counts = Counter(row["pair_id"] for row in canary_rows)
    quality_by_finding = {}
    for finding in sorted({row["finding"] for row in pair_rows}):
        local = [row for row in pair_rows if row["finding"] == finding]
        local_similarity = [row["tfidf_cosine"] for row in local]
        local_gap = [row["present_absent_length_gap"] for row in local]
        quality_by_finding[finding] = {
            "n": len(local),
            "tfidf_cosine_min": min(local_similarity),
            "tfidf_cosine_median": float(np.median(local_similarity)),
            "tfidf_cosine_mean": float(np.mean(local_similarity)),
            "length_gap_max": max(local_gap),
        }
    result = {
        "status": "completed_cpu_manifest_only_no_model_run",
        "protocol": "matched-retrieval-polarity-canary-v1",
        "hard_contract": {
            "same_finding": True,
            "query_present_absent_donor_patients_all_distinct": True,
            "present_absent_word_length_gap_max": MAX_LENGTH_GAP,
            "matching": "highest corpus TF-IDF cosine under hard gates; greedy unique present and absent donors per finding",
            "neutral": "remove every claim-bearing sentence from present report",
            "random_deletion": "remove exactly the same number of whitespace words from non-claim sentences while retaining present claim",
            "no_relaxation": True,
            "selection_uses_target_label": False,
            "cxr_complex_claims_excluded_before_matching": COMPLEX_QUERY.pattern,
        },
        "counts": {
            "queries_selected_before_donor_gate": sum(len(rows) for rows in queries.values()),
            "matched_queries": len(pair_rows),
            "arm_rows": len(canary_rows),
            "arm_counts": dict(Counter(row["arm"] for row in canary_rows)),
            "matched_by_cohort": dict(Counter(row["cohort"] for row in pair_rows)),
            "matched_by_finding": dict(Counter(row["finding"] for row in pair_rows)),
            "unmatched_by_finding_cohort": {finding: dict(counts) for finding, counts in unmatched.items()},
        },
        "matching_quality": {
            "all_patient_disjoint": all(row["patient_disjoint"] for row in pair_rows),
            "all_length_gap_within_10pct": all(value <= MAX_LENGTH_GAP for value in length_gaps),
            "neutral_random_word_counts_identical": all(row["neutral_words"] == row["random_deletion_words"] for row in pair_rows),
            "every_pair_has_exactly_four_arms": all(value == 4 for value in pair_arm_counts.values()) and len(pair_arm_counts) == len(pair_rows),
            "arm_semantic_state_counts": dict(observed_arm_states),
            "arm_semantic_state_mismatch_count": len(arm_state_mismatches),
            "arm_semantic_state_mismatch_examples": arm_state_mismatches[:20],
            "by_finding": quality_by_finding,
            "tfidf_cosine": {
                "min": min(similarities) if similarities else None,
                "median": float(np.median(similarities)) if similarities else None,
                "mean": float(np.mean(similarities)) if similarities else None,
                "max": max(similarities) if similarities else None,
            },
            "present_absent_length_gap": {
                "max": max(length_gaps) if length_gaps else None,
                "median": float(np.median(length_gaps)) if length_gaps else None,
                "mean": float(np.mean(length_gaps)) if length_gaps else None,
            },
        },
        "query_audit": query_audit,
        "donor_pool_audit": pool_audit,
        "leakage_audit": {
            "forbidden_target_fields_in_canary": sorted(
                set().union(*(set(row) & {"answer", "target", "label", "ground_truth", "gt_ans"} for row in canary_rows))
            ) if canary_rows else [],
            "note": "CXR loader accesses metadata/question fields only; VinDr source is its existing outcome-blind selected manifest. No answer, reader vote, or target field participates in selection.",
        },
        "artifacts": {
            "canary": str(CANARY_OUT),
            "canary_sha256": file_hash(CANARY_OUT),
            "pairs": str(PAIRS_OUT),
            "pairs_sha256": file_hash(PAIRS_OUT),
        },
        "provenance": {
            "script": str(Path(__file__)),
            "script_sha256": file_hash(Path(__file__)),
            "polarity_dependency": "anchor/corrected_sgta/analyze_retrieval_polarity_transplantation_v1.py",
            "polarity_dependency_sha256": file_hash(Path("anchor/corrected_sgta/analyze_retrieval_polarity_transplantation_v1.py")),
            "corpus": str(CORPUS), "corpus_sha256": file_hash(CORPUS),
            "vindr_manifest": str(VINDR), "vindr_manifest_sha256": file_hash(VINDR),
            "cxr_manifest": str(CXR), "cxr_manifest_sha256": file_hash(CXR),
            "seed": SEED,
        },
    }
    RESULT_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
