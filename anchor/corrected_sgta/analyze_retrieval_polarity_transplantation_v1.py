"""CPU-only retrieval-polarity transplantation audit over frozen outputs.

The chest-X-ray claim lexicon, local NegBio-style negation windows, ambiguity
cues, aggregation rule, and tests below are fixed before reading model outcomes.
No learned or LLM judge is used.  Unknown, mixed, absent, and unmapped states
are retained rather than coerced to a binary polarity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("corrected_runs/paper_baselines_v1/full_matrix_v1")
INPUT_ROOT = Path("corrected_runs/unified_eval/inputs/baseline_matrix_v1")
RAG_ROOT = ROOT / "rag/bm25"
SHARED_ROOT = ROOT / "shared_rag_generation"
CONTROL_ROOT = Path(
    "corrected_runs/unified_eval/rag/common_protocol_v1/mimic/visual_ce_v2"
)
OUT_DIR = Path("corrected_runs/retrieval_polarity_transplantation_v1")
RESULT_OUT = OUT_DIR / "result.json"
ROWS_OUT = OUT_DIR / "rows.jsonl"
CONTROL_ROWS_OUT = OUT_DIR / "shuffled_control_rows.jsonl"
SEED = 20260810
BOOTSTRAPS = 2_000


@dataclass(frozen=True)
class ClaimSpec:
    name: str
    question_pattern: str
    # Each mention is (regex, intrinsic proposition sign).  Local negation
    # reverses the sign.  +1 means evidence for a Yes answer to the claim.
    mentions: tuple[tuple[str, int], ...]


# Frozen, ordered, chest-radiograph claim lexicon.  Specific devices and
# normality propositions precede broader findings to avoid multi-match routing.
CLAIMS = (
    ClaimSpec("endotracheal_tube", r"\b(endotracheal|et) tube\b|\bintubat(?:ed|ion)\b", ((r"\b(?:endotracheal|et) tube\b|\bintubat(?:ed|ion)\b", 1),)),
    ClaimSpec("enteric_tube", r"\b(?:ng|enteric|nasogastric|nasoenteric|feeding) tube\b", ((r"\b(?:ng|enteric|nasogastric|nasoenteric|feeding) tube\b", 1),)),
    ClaimSpec("chest_tube", r"\bchest tube\b|\bpleural (?:tube|drain)\b", ((r"\bchest tube\b|\bpleural (?:tube|drain)\b", 1),)),
    ClaimSpec("central_line", r"\b(?:picc|central venous|central line|right ij|left ij|internal jugular|subclavian (?:line|catheter))\b", ((r"\b(?:picc|central venous (?:line|catheter)|central line|internal jugular (?:line|catheter)|subclavian (?:line|catheter))\b", 1),)),
    ClaimSpec("cardiac_device", r"\b(?:pacemaker|aicd|icd|cardiac device|pacing device)\b", ((r"\b(?:pacemaker|aicd|icd|cardiac device|pacing device)\b", 1),)),
    ClaimSpec("pneumomediastinum", r"\bpneumomediastinum\b", ((r"\bpneumomediastinum\b|\bmediastinal (?:air|gas)\b", 1),)),
    ClaimSpec("subdiaphragmatic_free_air", r"\b(?:subdiaphragmatic|subdiaphragmic|intraperitoneal|free) (?:air|gas)\b", ((r"\b(?:subdiaphragmatic|subdiaphragmic|intraperitoneal|free) (?:air|gas)\b", 1),)),
    ClaimSpec("pneumothorax", r"\bpneumothorax\b|\bpleural air\b", ((r"\bpneumothora(?:x|ces)\b|\bpleural air\b", 1),)),
    ClaimSpec("pleural_effusion", r"\b(?:pleural )?effusions?\b|\bpleural fluid\b", ((r"\b(?:pleural )?effusions?\b|\bpleural fluid\b", 1),)),
    ClaimSpec("aspiration", r"\baspirat(?:ion|ive|ed)\b", ((r"\baspirat(?:ion|ive|ed)\b", 1),)),
    ClaimSpec("pneumonia", r"\bpneumonia\b|\bpneumonitis\b", ((r"\bpneumonia\b|\bpneumonitis\b", 1),)),
    ClaimSpec("focal_consolidation", r"\b(?:focal )?consolidation\b|\bairspace disease\b", ((r"\b(?:focal )?consolidation\b|\bairspace disease\b", 1),)),
    ClaimSpec("atelectasis", r"\batelecta(?:sis|tic)\b|\blung collapse\b", ((r"\batelecta(?:sis|tic)\b|\blung collapse\b", 1),)),
    ClaimSpec("pulmonary_edema", r"\b(?:pulmonary|interstitial) (?:edema|oedema)\b|\bfluid overload\b", ((r"\b(?:pulmonary|interstitial) (?:edema|oedema)\b|\bfluid overload\b", 1),)),
    ClaimSpec("vascular_congestion", r"\b(?:pulmonary )?(?:vascular|venous) (?:congestion|engorgement|hypertension)\b|\bcongested pulmonary vasculature\b", ((r"\b(?:pulmonary )?(?:vascular|venous) (?:congestion|engorgement|hypertension)\b|\bcongested pulmonary vasculature\b", 1),)),
    ClaimSpec("cardiomegaly", r"\bcardiomegaly\b|\b(?:enlarged|enlargement of the) (?:heart|cardiac silhouette)\b|\bheart (?:is |appears )?enlarged\b|\bcardiac (?:size|silhouette) (?:is |appears )?(?:abnormal|enlarged)\b|\bheart size abnormal\b", ((r"\bcardiomegaly\b|\b(?:enlarged|enlargement of the) (?:heart|cardiac silhouette)\b|\bheart (?:is |appears )?enlarged\b|\bcardiac (?:size|silhouette) (?:is |appears )?enlarged\b", 1),)),
    ClaimSpec("cardiac_size_normal", r"\b(?:heart|cardiac|cardiomediastinal)(?: size| silhouette)?\b.{0,30}\b(?:normal|within normal limits|not enlarged)\b|\b(?:normal|within normal limits)\b.{0,30}\b(?:heart|cardiac|cardiomediastinal)(?: size| silhouette)?\b", ((r"\b(?:heart size|cardiac silhouette|cardiomediastinal silhouette)\b.{0,24}\b(?:normal|within normal limits|not enlarged)\b|\bnormal (?:heart size|cardiac silhouette)\b", 1), (r"\bcardiomegaly\b|\b(?:enlarged|enlargement of the) (?:heart|cardiac silhouette)\b|\bheart (?:is |appears )?enlarged\b", -1))),
    ClaimSpec("mediastinal_normality", r"\bmediastin(?:um|al)(?: contour| contours)?\b.{0,24}\b(?:normal|within normal limits|unremarkable)\b|\bnormal mediastinal", ((r"\bmediastinal (?:and hilar )?contours?\b.{0,20}\b(?:normal|unremarkable)\b|\bnormal mediastinal contours?\b", 1), (r"\bmediastinal (?:widening|enlargement|abnormality|mass)\b|\babnormal mediastinal contour\b", -1))),
    ClaimSpec("clear_lungs", r"\blungs?\b.{0,24}\b(?:clear|well aerated)\b|\bclear (?:lung|lungs|lung fields)\b", ((r"\b(?:lungs?|lung fields)\b.{0,20}\b(?:clear|well aerated)\b|\bclear (?:lung|lungs|lung fields)\b", 1),)),
    ClaimSpec("normal_lung_volume", r"\blung volumes?\b.{0,20}\bnormal\b", ((r"\blung volumes?\b.{0,20}\bnormal\b", 1), (r"\b(?:low lung volumes?|hypoinflat(?:ed|ion)|hyperinflat(?:ed|ion))\b", -1))),
    ClaimSpec("fracture", r"\bfractur(?:e|ed|es)\b|\bacute (?:bony|osseous) abnormalit", ((r"\bfractur(?:e|ed|es)\b|\bacute (?:bony|osseous) abnormalit(?:y|ies)\b", 1),)),
    ClaimSpec("osseous_intact", r"\b(?:bony|osseous|skeletal) structures?\b.{0,25}\b(?:intact|normal|unremarkable)\b", ((r"\b(?:bony|osseous|skeletal) structures?\b.{0,25}\b(?:intact|normal|unremarkable)\b", 1), (r"\bfractur(?:e|ed|es)\b|\bacute (?:bony|osseous) abnormalit(?:y|ies)\b", -1))),
    ClaimSpec("mass_or_nodule", r"\b(?:mass|nodule|tumou?r|lesion)\b", ((r"\b(?:mass|nodule|tumou?r|lesion)s?\b", 1),)),
    ClaimSpec("emphysema", r"\bemphysema\b|\bhyperinflat", ((r"\bemphysema\b|\bhyperinflat(?:ed|ion)\b", 1),)),
    ClaimSpec("fibrosis_or_scarring", r"\b(?:fibrosis|fibrotic|scarring|scar)\b", ((r"\b(?:fibrosis|fibrotic|scarring|scar)\b", 1),)),
    ClaimSpec("airspace_opacity", r"\b(?:airspace|parenchymal|pulmonary|lung) (?:opacity|opacities)\b", ((r"\b(?:airspace|parenchymal|pulmonary|lung) (?:opacity|opacities)\b", 1),)),
    ClaimSpec("acute_cardiopulmonary_process", r"\bacute (?:cardiopulmonary|intrathoracic|pulmonary) (?:process|abnormalit(?:y|ies)|disease)\b", ((r"\bacute (?:cardiopulmonary|intrathoracic|pulmonary) (?:process|abnormalit(?:y|ies)|disease)\b", 1),)),
)


NEG_PRE = re.compile(
    r"\b(?:no|not|without|neither|absence of|negative for|free of|clear of|fails? to (?:show|demonstrate)|does not (?:show|demonstrate|reveal))\b",
    re.I,
)
NEG_POST = re.compile(
    r"\b(?:is|are|was|were|has been|have been)?\s*(?:not (?:seen|identified|present|visualized)|absent|excluded|resolved|no longer (?:seen|present))\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|probable|probably|may|might|could|cannot exclude|can't exclude|not excluded|uncertain|equivocal|question of|concerning for|suggestive of|compatible with|likely)\b",
    re.I,
)
TOKEN = re.compile(r"\b[\w'-]+\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def decision(text: Any) -> str:
    value = normalize(text)
    match = re.match(r"^(yes|no|uncertain)\b", value)
    return match.group(1) if match else "unknown"


def target_decision(value: Any) -> str:
    value = normalize(value).rstrip(".")
    return value if value in {"yes", "no"} else "unknown"


def claim_for_question(question: str) -> ClaimSpec | None:
    lowered = normalize(question)
    for spec in CLAIMS:
        if re.search(spec.question_pattern, lowered, re.I):
            return spec
    return None


def local_clause(text: str, start: int, end: int) -> tuple[str, int, int]:
    left = max(text.rfind(".", 0, start), text.rfind(";", 0, start), text.rfind(" but ", 0, start))
    right_candidates = [value for value in (text.find(".", end), text.find(";", end), text.find(" but ", end)) if value >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    offset = left + 1
    return text[offset:right], start - offset, end - offset


def window_by_tokens(clause: str, start: int, end: int, before: int, after: int) -> tuple[str, str]:
    tokens = list(TOKEN.finditer(clause))
    before_tokens = [item for item in tokens if item.end() <= start][-before:]
    after_tokens = [item for item in tokens if item.start() >= end][:after]
    pre = clause[before_tokens[0].start():start] if before_tokens else clause[:start]
    post = clause[end:after_tokens[-1].end()] if after_tokens else clause[end:]
    return pre, post


def mention_assertion(clause: str, start: int, end: int, intrinsic: int) -> str:
    pre, post = window_by_tokens(clause, start, end, before=6, after=5)
    vicinity = f"{pre} {clause[start:end]} {post}"
    # Pseudo-negation/hedging takes precedence over ordinary negation.
    if UNCERTAIN.search(vicinity):
        return "unknown"
    negated = bool(NEG_PRE.search(pre) or NEG_POST.search(post))
    sign = intrinsic * (-1 if negated else 1)
    return "positive" if sign > 0 else "negative"


def extract_polarity(question: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    spec = claim_for_question(question)
    if spec is None:
        return {
            "finding": "unmapped",
            "polarity": "unmapped",
            "counts": {"positive": 0, "negative": 0, "unknown": 0},
            "evidence": [],
        }
    evidence: list[dict[str, Any]] = []
    counts = Counter()
    for document in documents:
        report = normalize(document.get("report", ""))
        for pattern, intrinsic in spec.mentions:
            for match in re.finditer(pattern, report, re.I):
                clause, start, end = local_clause(report, match.start(), match.end())
                assertion = mention_assertion(clause, start, end, intrinsic)
                counts[assertion] += 1
                if len(evidence) < 8:
                    evidence.append({
                        "doc_id": str(document.get("doc_id", "")),
                        "rank": document.get("rank"),
                        "assertion": assertion,
                        "mention": match.group(0),
                        "clause": clause.strip()[:400],
                    })
    asserted = {name for name in ("positive", "negative") if counts[name]}
    if len(asserted) == 2:
        polarity = "mixed"
    elif asserted:
        polarity = next(iter(asserted))
    elif counts["unknown"]:
        polarity = "unknown"
    else:
        polarity = "absent"
    return {
        "finding": spec.name,
        "polarity": polarity,
        "counts": {name: int(counts[name]) for name in ("positive", "negative", "unknown")},
        "evidence": evidence,
    }


def load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    output = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row[key])] = row
    return output


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    return load_jsonl(path, "question_id")


def compact_output(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": decision(row.get("text")),
        "text": str(row.get("text", "")),
        "tokens": row.get("metadata", {}).get("generated_token_count"),
    }


def build_rows(dataset: str) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest_path = INPUT_ROOT / f"{dataset}.json"
    retrieval_path = RAG_ROOT / dataset / "retrieval.jsonl"
    manifest = {str(row["qid"]): row for row in json.loads(manifest_path.read_text())}
    retrieval = load_jsonl(retrieval_path, "sample_id")
    answer_paths = []
    answers: dict[str, dict[str, dict[str, Any]]] = {}
    for model in ("huatuo", "hulu"):
        answers[model] = {}
        for condition in ("no_context", "rag"):
            path = SHARED_ROOT / model / dataset / condition / "answers.jsonl"
            answer_paths.append(path)
            answers[model][condition] = load_answers(path)

    qids = sorted(set(manifest) & set(retrieval))
    rows = []
    for qid in qids:
        if any(qid not in answers[model][condition] for model in answers for condition in answers[model]):
            continue
        source = manifest[qid]
        polarity = extract_polarity(source.get("source_question", source.get("question", "")), retrieval[qid].get("documents", []))
        rows.append({
            "dataset": dataset,
            "qid": qid,
            "cluster": str(source.get("patient_id") or source.get("img_name") or qid),
            "question": str(source.get("source_question", source.get("question", ""))),
            "target": target_decision(source.get("answer")),
            "finding": polarity["finding"],
            "retrieval_polarity": polarity["polarity"],
            "retrieval_counts": polarity["counts"],
            "retrieval_evidence": polarity["evidence"],
            "outputs": {
                model: {
                    "plain": compact_output(answers[model]["no_context"][qid]),
                    "rag": compact_output(answers[model]["rag"][qid]),
                }
                for model in ("huatuo", "hulu")
            },
        })
    return rows, [manifest_path, retrieval_path, *answer_paths]


def cluster_bootstrap(values: list[tuple[str, float]], seed: int) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "ci95": [None, None], "clusters": 0, "replicates": BOOTSTRAPS}
    clusters = sorted({cluster for cluster, _ in values})
    index = {cluster: position for position, cluster in enumerate(clusters)}
    sums = np.zeros(len(clusters), dtype=float)
    counts = np.zeros(len(clusters), dtype=float)
    for cluster, value in values:
        sums[index[cluster]] += value
        counts[index[cluster]] += 1
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=BOOTSTRAPS
    )
    denominator = weights @ counts
    samples = (weights @ sums) / np.maximum(1.0, denominator)
    return {
        "estimate": float(sums.sum() / counts.sum()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "clusters": len(clusters),
        "replicates": BOOTSTRAPS,
    }


def effect_summary(rows: list[dict[str, Any]], model: str, seed: int) -> dict[str, Any]:
    known = [row for row in rows if row["retrieval_polarity"] in {"positive", "negative"}]
    polarity_decision = {"positive": "yes", "negative": "no"}
    alignment_values = []
    flip_values = []
    binary_flips = []
    unknown_transitions = Counter()
    for row in known:
        wanted = polarity_decision[row["retrieval_polarity"]]
        plain = row["outputs"][model]["plain"]["decision"]
        rag = row["outputs"][model]["rag"]["decision"]
        alignment_values.append((row["cluster"], float(rag == wanted) - float(plain == wanted)))
        unknown_transitions[f"{plain}->{rag}"] += 1
        if plain in {"yes", "no"} and rag in {"yes", "no"} and plain != rag:
            direction = 1.0 if rag == wanted else -1.0
            flip_values.append((row["cluster"], direction))
            binary_flips.append(direction)
    plain_matches = sum(
        row["outputs"][model]["plain"]["decision"] == polarity_decision[row["retrieval_polarity"]]
        for row in known
    )
    rag_matches = sum(
        row["outputs"][model]["rag"]["decision"] == polarity_decision[row["retrieval_polarity"]]
        for row in known
    )
    return {
        "n_all": len(rows),
        "clusters_all": len({row["cluster"] for row in rows}),
        "known_polarity_n": len(known),
        "polarity_counts": dict(Counter(row["retrieval_polarity"] for row in rows)),
        "plain_output_counts": dict(Counter(row["outputs"][model]["plain"]["decision"] for row in rows)),
        "rag_output_counts": dict(Counter(row["outputs"][model]["rag"]["decision"] for row in rows)),
        "plain_retrieval_alignment_rate": plain_matches / len(known) if known else None,
        "rag_retrieval_alignment_rate": rag_matches / len(known) if known else None,
        "all_state_alignment_delta": cluster_bootstrap(alignment_values, seed),
        "binary_flip_n": len(binary_flips),
        "binary_flips_toward_retrieval": int(sum(value > 0 for value in binary_flips)),
        "binary_flips_away_from_retrieval": int(sum(value < 0 for value in binary_flips)),
        "binary_flip_toward_fraction": float(np.mean(np.asarray(binary_flips) > 0)) if binary_flips else None,
        "binary_flip_directional_excess": cluster_bootstrap(flip_values, seed + 1),
        "all_known_polarity_transitions": dict(unknown_transitions),
    }


def grouped_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in ("cxr_vishal", "knowledge_mimic_ce"):
        local_dataset = [row for row in rows if row["dataset"] == dataset and row["target"] in {"yes", "no"}]
        result[dataset] = {}
        for model_index, model in enumerate(("huatuo", "hulu")):
            base_seed = SEED + 1000 * model_index + (0 if dataset == "cxr_vishal" else 100)
            model_result: dict[str, Any] = {
                "overall": effect_summary(local_dataset, model, base_seed),
                "by_ground_truth": {
                    target: effect_summary([row for row in local_dataset if row["target"] == target], model, base_seed + 10 + i)
                    for i, target in enumerate(("no", "yes"))
                },
                "by_retrieval_polarity": {
                    polarity: effect_summary([row for row in local_dataset if row["retrieval_polarity"] == polarity], model, base_seed + 20 + i)
                    for i, polarity in enumerate(("positive", "negative"))
                },
                "by_truth_relation": {},
                "by_finding": {},
            }
            for i, relation in enumerate(("supports_ground_truth", "conflicts_with_ground_truth")):
                selected = []
                for row in local_dataset:
                    polarity = row["retrieval_polarity"]
                    if polarity not in {"positive", "negative"}:
                        continue
                    wanted = "yes" if polarity == "positive" else "no"
                    current = "supports_ground_truth" if wanted == row["target"] else "conflicts_with_ground_truth"
                    if current == relation:
                        selected.append(row)
                model_result["by_truth_relation"][relation] = effect_summary(selected, model, base_seed + 30 + i)
            findings = sorted({row["finding"] for row in local_dataset})
            for i, finding in enumerate(findings):
                selected = [row for row in local_dataset if row["finding"] == finding]
                known_n = sum(row["retrieval_polarity"] in {"positive", "negative"} for row in selected)
                if known_n >= 10:
                    model_result["by_finding"][finding] = effect_summary(selected, model, base_seed + 100 + i)
            result[dataset][model] = model_result
    return result


def extract_prompt_parts(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    context = prompt.split("Retrieved reports:\n", 1)[1].rsplit("\nQuestion:", 1)[0]
    question_tail = prompt.rsplit("\nQuestion:", 1)[1]
    question = question_tail.split("\nBegin the answer", 1)[0].strip()
    pieces = re.split(r"\n\[(\d+)\]\s*", "\n" + context)
    documents = []
    for index in range(1, len(pieces), 2):
        documents.append({"rank": int(pieces[index]), "doc_id": f"prompt-rank-{pieces[index]}", "report": pieces[index + 1].strip()})
    return question, documents


def build_shuffled_control() -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    original_prompt_path = CONTROL_ROOT / "t3_n200_top3/prompts/rag.json"
    shuffled_prompt_path = CONTROL_ROOT / "t3_n200_top3/controls_v1/shuffled_context.json"
    original_answer_path = CONTROL_ROOT / "ladder_v3/T3_n200/huatuo/rag/answers.jsonl"
    shuffled_answer_path = CONTROL_ROOT / "ladder_v3/causal_controls_v1/T3_n200/huatuo/shuffled_context/answers.jsonl"
    original_prompts = {str(row["qid"]): row for row in json.loads(original_prompt_path.read_text())}
    shuffled_prompts = {str(row["qid"]): row for row in json.loads(shuffled_prompt_path.read_text())}
    original_answers = load_answers(original_answer_path)
    shuffled_answers = load_answers(shuffled_answer_path)
    qids = sorted(set(original_prompts) & set(shuffled_prompts) & set(original_answers) & set(shuffled_answers), key=lambda value: int(value))
    rows = []
    for qid in qids:
        question, original_documents = extract_prompt_parts(original_prompts[qid]["question"])
        shuffled_question, shuffled_documents = extract_prompt_parts(shuffled_prompts[qid]["question"])
        if normalize(question) != normalize(shuffled_question):
            raise RuntimeError(f"shuffled control changed question at qid={qid}")
        original_polarity = extract_polarity(question, original_documents)
        shuffled_polarity = extract_polarity(question, shuffled_documents)
        img_name = str(original_prompts[qid]["img_name"])
        parts = img_name.split("/")
        cluster = parts[1] if len(parts) > 1 else img_name
        rows.append({
            "qid": qid,
            "cluster": cluster,
            "question": question,
            "target": target_decision(original_prompts[qid].get("answer")),
            "finding": original_polarity["finding"],
            "original_polarity": original_polarity["polarity"],
            "shuffled_polarity": shuffled_polarity["polarity"],
            "original_evidence": original_polarity["evidence"],
            "shuffled_evidence": shuffled_polarity["evidence"],
            "original_answer": compact_output(original_answers[qid]),
            "shuffled_answer": compact_output(shuffled_answers[qid]),
            "context_donor_qid": shuffled_prompts[qid].get("context_donor_qid"),
        })

    polarity_to_answer = {"positive": "yes", "negative": "no"}
    informative = [row for row in rows if row["original_polarity"] in polarity_to_answer and row["shuffled_polarity"] in polarity_to_answer and row["original_polarity"] != row["shuffled_polarity"]]
    new_alignment = []
    old_alignment_loss = []
    flip_direction = []
    for row in informative:
        new_wanted = polarity_to_answer[row["shuffled_polarity"]]
        old_wanted = polarity_to_answer[row["original_polarity"]]
        original_answer = row["original_answer"]["decision"]
        shuffled_answer = row["shuffled_answer"]["decision"]
        new_alignment.append((row["cluster"], float(shuffled_answer == new_wanted) - float(original_answer == new_wanted)))
        old_alignment_loss.append((row["cluster"], float(original_answer == old_wanted) - float(shuffled_answer == old_wanted)))
        if original_answer in {"yes", "no"} and shuffled_answer in {"yes", "no"} and original_answer != shuffled_answer:
            flip_direction.append((row["cluster"], 1.0 if shuffled_answer == new_wanted else -1.0))
    summary = {
        "status": "connected_existing_n200_huatuo_control",
        "n": len(rows),
        "clusters": len({row["cluster"] for row in rows}),
        "original_polarity_counts": dict(Counter(row["original_polarity"] for row in rows)),
        "shuffled_polarity_counts": dict(Counter(row["shuffled_polarity"] for row in rows)),
        "polarity_changed_known_to_opposite_n": len(informative),
        "new_context_alignment_delta": cluster_bootstrap(new_alignment, SEED + 9000),
        "old_context_alignment_loss": cluster_bootstrap(old_alignment_loss, SEED + 9001),
        "binary_answer_flip_n": len(flip_direction),
        "binary_flip_toward_new_context_n": int(sum(value > 0 for _, value in flip_direction)),
        "binary_flip_away_from_new_context_n": int(sum(value < 0 for _, value in flip_direction)),
        "binary_flip_directional_excess": cluster_bootstrap(flip_direction, SEED + 9002),
        "scope_note": "Existing shuffled-context output exists for Huatuo only; no Hulu n=200 shuffled arm was found.",
    }
    return rows, summary, [original_prompt_path, shuffled_prompt_path, original_answer_path, shuffled_answer_path]


def examples(rows: list[dict[str, Any]], limit_each: int = 3) -> list[dict[str, Any]]:
    selected = []
    counter = Counter()
    for row in rows:
        if row["retrieval_polarity"] not in {"positive", "negative"}:
            continue
        wanted = "yes" if row["retrieval_polarity"] == "positive" else "no"
        for model in ("huatuo", "hulu"):
            plain = row["outputs"][model]["plain"]["decision"]
            rag = row["outputs"][model]["rag"]["decision"]
            if plain not in {"yes", "no"} or rag not in {"yes", "no"} or plain == rag:
                continue
            direction = "toward" if rag == wanted else "away"
            key = (row["dataset"], model, direction)
            if counter[key] >= limit_each:
                continue
            counter[key] += 1
            selected.append({
                "dataset": row["dataset"], "qid": row["qid"], "model": model,
                "direction": direction, "question": row["question"], "target": row["target"],
                "finding": row["finding"], "retrieval_polarity": row["retrieval_polarity"],
                "evidence": row["retrieval_evidence"][:2],
                "plain": row["outputs"][model]["plain"], "rag": row["outputs"][model]["rag"],
            })
    return selected


def main() -> None:
    all_rows = []
    input_paths: list[Path] = []
    dataset_counts = {}
    for dataset in ("cxr_vishal", "knowledge_mimic_ce"):
        rows, paths = build_rows(dataset)
        all_rows.extend(rows)
        input_paths.extend(paths)
        dataset_counts[dataset] = {
            "all_joined": len(rows),
            "binary_ground_truth": sum(row["target"] in {"yes", "no"} for row in rows),
            "target_counts": dict(Counter(row["target"] for row in rows)),
            "finding_counts": dict(Counter(row["finding"] for row in rows)),
            "retrieval_polarity_counts": dict(Counter(row["retrieval_polarity"] for row in rows)),
        }

    control_rows, control_summary, control_paths = build_shuffled_control()
    input_paths.extend(control_paths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    CONTROL_ROWS_OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in control_rows))

    result = {
        "status": "completed_cpu_only_frozen_rule_audit",
        "version": "retrieval-polarity-transplantation-v1",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "preregistration": {
            "lexicon_frozen_before_outcome_read": True,
            "claim_specs": [asdict(spec) for spec in CLAIMS],
            "local_negation": "same local clause; 6 tokens before or 5 after mention",
            "ambiguity_precedence": "possible/probable/may/might/could/cannot exclude/not excluded/uncertain/equivocal/question of/concerning/suggestive/compatible/likely => unknown before negation",
            "document_aggregation": "positive+negative => mixed; one asserted sign => that sign; uncertain-only => unknown; no mention => absent; unmapped question => unmapped",
            "primary_statistic": "change in exact answer alignment with retrieved polarity, RAG minus plain; patient-cluster percentile bootstrap",
            "flip_statistic": "among binary Yes<->No flips, +1 when RAG equals retrieved polarity and -1 otherwise; patient-cluster bootstrap",
            "unknown_handling": "unknown outputs and unknown/mixed/absent/unmapped retrieval states are retained in rows and counts, never coerced",
            "no_llm_judge": True,
        },
        "dataset_counts": dataset_counts,
        "effects": grouped_effects(all_rows),
        "existing_n200_shuffled_context_control": control_summary,
        "examples": examples(all_rows),
        "interpretation_boundary": {
            "observational_plain_vs_rag": "Association can reflect both report polarity and other retrieved text properties; GT and finding stratification do not by themselves prove causality.",
            "shuffled_control": "The n=200 fixed-image/fixed-question context replacement is the causal transplantation check, but is available only for Huatuo and only where both contexts receive opposite known lexicon polarities.",
            "lexicon": "High precision, incomplete rule coverage; absent means no lexicon mention, not clinical absence.",
        },
        "artifacts": {"rows": str(ROWS_OUT), "shuffled_control_rows": str(CONTROL_ROWS_OUT)},
        "provenance": {
            "script": str(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
            "inputs": {str(path): sha256_file(path) for path in sorted(set(input_paths), key=str)},
        },
    }
    RESULT_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "dataset_counts": dataset_counts,
        "shuffled_control": control_summary, "result": str(RESULT_OUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
