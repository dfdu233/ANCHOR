"""Build Huatuo/Hulu tokenizer-matched neutral RAG controls on CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer, __version__ as transformers_version


ROOT = Path("corrected_runs/polarity_firewall_canary_v1")
RAW = ROOT / "raw_rag.json"
NEUTRAL = ROOT / "depolarized_rag.json"
MODELS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}

# Frozen, state-free glossary. These are not meaningless padding symbols: every
# item names a generic radiographic assessment dimension.
FILLER_GLOSSARY = {
    "anatomy": "anatomical structure under review",
    "location": "spatial localization dimension",
    "distribution": "spatial distribution dimension",
    "shape": "morphological shape dimension",
    "margin": "boundary morphology dimension",
    "size": "size assessment dimension",
    "projection": "acquisition projection consideration",
    "landmark": "anatomical landmark relation",
    "appearance": "generic visual appearance",
    "orientation": "spatial orientation dimension",
    "region": "anatomical region dimension",
    "pattern": "radiographic pattern dimension",
    "density": "radiographic density dimension",
    "contour": "contour morphology dimension",
    "view": "image-view consideration",
    "axis": "spatial axis dimension",
    "characterization": "generic visual feature characterization",
    "configuration": "generic spatial configuration",
    "visualization": "image visualization consideration",
    "interpretation": "generic image interpretation dimension",
    "measurement": "generic measurement dimension",
    "morphology": "generic morphological dimension",
    "localization": "generic spatial localization",
    "assessment": "generic image assessment",
    "relationship": "anatomical relationship dimension",
    "consideration": "generic technical consideration",
    "differentiation": "visual feature differentiation",
}
FORBIDDEN = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsent\b", r"\bpresent\b",
    r"\bidentified\b", r"\bseen\b", r"\bnoted\b", r"\bunchanged\b",
    r"\bincreased\b", r"\bdecreased\b", r"\bimproved\b", r"\bworsened\b",
    r"\bpatient\b", r"\brecommend\b", r"\bcm\b", r"\bmm\b",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parts(question: str) -> tuple[str, str, str]:
    prefix, remainder = question.split("\nRetrieved reports:\n", 1)
    context, query = remainder.rsplit("\nQuestion:\n", 1)
    return prefix, context, query


def assemble(prefix: str, context: str, query: str) -> str:
    return f"{prefix}\nRetrieved reports:\n{context}\nQuestion:\n{query}"


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_match(tokenizer, raw_question: str, neutral_question: str) -> tuple[str, dict]:
    raw_prefix, raw_context, raw_query = parts(raw_question)
    neutral_prefix, neutral_context, neutral_query = parts(neutral_question)
    if (raw_prefix, raw_query) != (neutral_prefix, neutral_query):
        raise RuntimeError("outside-context prompt drift in source neutral arm")

    target = token_count(tokenizer, raw_question)
    term_lines = [line for line in neutral_context.splitlines() if line.startswith("- ")]
    # Preserve the first (query-relevant) card, then prefer lines carrying more
    # human-readable characters per model token. This improves the secondary
    # character match without sacrificing the token constraint.
    if term_lines:
        term_lines = [term_lines[0]] + sorted(
            term_lines[1:],
            key=lambda line: -len(line) / max(1, token_count(tokenizer, "\n" + line)),
        )
    context = "[1] Neutral terminology; state unresolved from retrieval."
    prompt = assemble(raw_prefix, context, raw_query)
    if token_count(tokenizer, prompt) > target:
        context = "[1] Neutral terminology."
        prompt = assemble(raw_prefix, context, raw_query)
    if token_count(tokenizer, prompt) > target:
        raise RuntimeError("raw prompt too short for minimal neutral context")

    retained = []
    skipped = []
    for line in term_lines:
        candidate_context = context + "\n" + line
        candidate_prompt = assemble(raw_prefix, candidate_context, raw_query)
        if token_count(tokenizer, candidate_prompt) <= target:
            context, prompt = candidate_context, candidate_prompt
            retained.append(line)
        else:
            skipped.append(line)

    filler = []
    glossary = list(FILLER_GLOSSARY)
    while token_count(tokenizer, prompt) < target:
        current = token_count(tokenizer, prompt)
        remaining = target - current
        candidates = []
        # All glossary entries are meaningful. Among token-valid choices, prefer
        # the character increment closest to the remaining per-token character
        # budget; model-token equality remains the hard constraint.
        desired_char_increment = (len(raw_question) - len(prompt)) / remaining
        for order, word in enumerate(glossary):
            candidate_context = context + " " + word
            candidate_prompt = assemble(raw_prefix, candidate_context, raw_query)
            increment = token_count(tokenizer, candidate_prompt) - current
            if 0 < increment <= remaining:
                char_increment = len(candidate_prompt) - len(prompt)
                candidates.append((
                    abs(char_increment / increment - desired_char_increment),
                    filler.count(word), order, word, candidate_context, candidate_prompt,
                ))
        if not candidates:
            break
        _, _, _, word, context, prompt = min(candidates)
        filler.append(word)

    final_count = token_count(tokenizer, prompt)
    error = final_count - target
    if abs(error) > 1:
        raise RuntimeError(f"token match failed: target={target}, final={final_count}")
    context_hits = {
        pattern: len(re.findall(pattern, context, re.I))
        for pattern in FORBIDDEN if re.search(pattern, context, re.I)
    }
    return prompt, {
        "target_model_tokens": target,
        "matched_model_tokens": final_count,
        "token_error": error,
        "raw_chars": len(raw_question),
        "matched_chars": len(prompt),
        "char_delta": len(prompt) - len(raw_question),
        "raw_context_chars": len(raw_context),
        "matched_context_chars": len(context),
        "retained_term_lines": len(retained),
        "available_term_lines": len(term_lines),
        "skipped_term_lines": len(skipped),
        "filler_words": filler,
        "filler_semantics": {word: FILLER_GLOSSARY[word] for word in sorted(set(filler))},
        "forbidden_context_hits": context_hits,
        "uses_trailing_whitespace": bool(context and context[-1].isspace()),
    }


def build(model_name: str) -> None:
    model_path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    raw = json.loads(RAW.read_text())
    neutral = json.loads(NEUTRAL.read_text())
    if [str(x["qid"]) for x in raw] != [str(x["qid"]) for x in neutral]:
        raise RuntimeError("raw/neutral qid order mismatch")

    output, details = [], []
    for raw_row, neutral_row in zip(raw, neutral):
        matched, audit = make_match(tokenizer, raw_row["question"], neutral_row["question"])
        safe = {key: value for key, value in neutral_row.items() if key not in {"question", "context_condition"}}
        output.append({
            **safe,
            "context_condition": f"{model_name}_token_matched_neutral_rag",
            "question": matched,
        })
        details.append({"question_id": str(raw_row["qid"]), **audit})

    manifest_path = ROOT / f"{model_name}_token_matched_neutral_rag.json"
    manifest_path.write_text(json.dumps(output, indent=2) + "\n")
    tokenizer_files = [
        name for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json")
        if (model_path / name).is_file()
    ]
    audit = {
        "status": "completed_cpu_tokenizer_only_no_model_run",
        "model": model_name,
        "tokenizer_path": str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers_version,
        "tokenizer_file_sha256": {name: sha256(model_path / name) for name in tokenizer_files},
        "target_labels_used": False,
        "forbidden_target_field_cases": sum(
            any(key in row for key in ("answer", "gt_ans", "ground_truth", "label"))
            for row in output
        ),
        "filler_glossary": FILLER_GLOSSARY,
        "n": len(output),
        "qid_order_matches_raw": [str(x["qid"]) for x in output] == [str(x["qid"]) for x in raw],
        "outside_context_drift_count": sum(
            (parts(output[i]["question"])[0], parts(output[i]["question"])[2])
            != (parts(raw[i]["question"])[0], parts(raw[i]["question"])[2])
            for i in range(len(raw))
        ),
        "exact_token_matches": sum(x["token_error"] == 0 for x in details),
        "within_one_token_matches": sum(abs(x["token_error"]) <= 1 for x in details),
        "max_abs_token_error": max(abs(x["token_error"]) for x in details),
        "mean_raw_characters": sum(x["raw_chars"] for x in details) / len(details),
        "mean_abs_character_delta": sum(abs(x["char_delta"]) for x in details) / len(details),
        "mean_abs_character_delta_fraction": (
            sum(abs(x["char_delta"]) for x in details) / sum(x["raw_chars"] for x in details)
        ),
        "mean_character_delta": sum(x["char_delta"] for x in details) / len(details),
        "character_delta_range": [min(x["char_delta"] for x in details), max(x["char_delta"] for x in details)],
        "mean_retained_term_line_fraction": sum(
            x["retained_term_lines"] / max(1, x["available_term_lines"]) for x in details
        ) / len(details),
        "total_filler_words": sum(len(x["filler_words"]) for x in details),
        "mean_filler_words_per_prompt": sum(len(x["filler_words"]) for x in details) / len(details),
        "trailing_whitespace_cases": sum(x["uses_trailing_whitespace"] for x in details),
        "forbidden_context_hit_cases": sum(bool(x["forbidden_context_hits"]) for x in details),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "details": details,
        "limitations": [
            "Token equality is model-tokenizer specific; use only the matching manifest for each model.",
            "Generic neutral glossary words control token count but still alter discourse style.",
            "Model token count is prioritized; character length is reported rather than forced.",
        ],
    }
    audit_path = ROOT / f"{model_name}_token_matching_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: audit[key] for key in [
        "model", "n", "exact_token_matches", "within_one_token_matches",
        "max_abs_token_error", "mean_abs_character_delta", "character_delta_range",
        "mean_retained_term_line_fraction", "total_filler_words",
        "trailing_whitespace_cases", "forbidden_context_hit_cases", "manifest_sha256",
    ]}, indent=2))


def combine() -> None:
    audits = {
        model: json.loads((ROOT / f"{model}_token_matching_audit.json").read_text())
        for model in MODELS
    }
    result = {
        "status": "completed_cpu_model_token_exact_controls",
        "target_labels_used": False,
        "models": {
            model: {key: audit[key] for key in [
                "tokenizer_path", "tokenizer_class", "transformers_version",
                "tokenizer_file_sha256", "n", "qid_order_matches_raw",
                "forbidden_target_field_cases",
                "outside_context_drift_count", "exact_token_matches",
                "within_one_token_matches", "max_abs_token_error",
                "mean_abs_character_delta", "mean_character_delta",
                "mean_raw_characters", "mean_abs_character_delta_fraction",
                "character_delta_range", "mean_retained_term_line_fraction",
                "total_filler_words", "trailing_whitespace_cases",
                "mean_filler_words_per_prompt",
                "forbidden_context_hit_cases", "manifest", "manifest_sha256",
                "limitations",
            ]}
            for model, audit in audits.items()
        },
        "cross_arm_qid_order_identical": (
            [x["question_id"] for x in audits["huatuo"]["details"]]
            == [x["question_id"] for x in audits["hulu"]["details"]]
        ),
        "filler_glossary": FILLER_GLOSSARY,
    }
    (ROOT / "model_token_matching_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "combine"], required=True)
    args = parser.parse_args()
    combine() if args.model == "combine" else build(args.model)


if __name__ == "__main__":
    main()
