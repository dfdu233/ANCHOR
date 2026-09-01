#!/usr/bin/env python3
"""Build a CPU-only, fail-closed same-report polarity-twin substrate.

The source is the frozen 32-pair matched-retrieval pilot.  For each pair we
inspect the two donor reports, but never an answer, target label, or model
output.  A primary pair is retained only when one report has exactly one
simple, claim-isolated sentence about the finding.  Temporal, comparative,
anatomic, severity, measurement, uncertainty, mixed, and multi-mention cases
are excluded rather than rewritten.

The three causal twins replace that one sentence with a frozen template.  The
templates differ at exactly one whitespace token: present / absent /
uncertain.  The report prefix and suffix are byte-identical.  Original, plain,
and a deterministic non-finding-sentence edit are emitted as diagnostic
controls, not as members of the one-variable twin contrast.  No GPU is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from anchor.corrected_sgta.build_matched_retrieval_polarity_canary_v1 import (
    FINDINGS,
    report_state,
    sentence_assertions,
)
from anchor.corrected_sgta.run_target_blind_canary_v1 import (
    load_target_blind_manifest,
    preflight_inputs,
)


SOURCE = Path(
    "corrected_runs/matched_retrieval_polarity_pilot_v1/target_blind_pilot_v2.json"
)
OUT_DIR = Path("corrected_runs/same_report_polarity_twin_pilot_v1")
MANIFEST = OUT_DIR / "target_blind_same_report_twins.json"
PAIR_AUDIT = OUT_DIR / "pair_audit.jsonl"
RESULT = OUT_DIR / "result.json"
IMAGE_ROOT = Path("data/medheval/images")
MODEL_TOKENIZERS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}
PROTOCOL = "same-report-polarity-twin-pilot-v1"
SOURCE_ARMS = ("present", "absent")
ARMS = (
    "twin_present",
    "twin_absent",
    "twin_uncertain_withheld",
    "original",
    "plain",
    "random_non_target_edit",
)

DISPLAY = {
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung opacity",
    "pleural_effusion": "Pleural effusion",
    "pneumothorax": "Pneumothorax",
}
TWIN_WORDS = {
    "twin_present": "present",
    "twin_absent": "absent",
    "twin_uncertain_withheld": "uncertain",
}
RANDOM_EDIT = "A non-target report sentence is explicitly withheld in this control."

# Deliberately broad.  Hits are excluded even when a modifier might refer to a
# coordinated non-finding claim; this is a feasibility substrate, not a recall
# maximization pass.
TEMPORAL = re.compile(
    r"\b(?:compared|comparison|prior|previous|interval|new(?:ly)?|unchanged|"
    r"stable|remains?|again|persist\w*|increas\w*|decreas\w*|improv\w*|"
    r"worsen\w*|resolv\w*|now|pre-existing|progress\w*)\b",
    re.I,
)
ATTRIBUTE = re.compile(
    r"\b(?:left|right|bilateral|unilateral|apical|basal|basilar|upper|lower|"
    r"mid|focal|small(?:er|est)?|large(?:r|st)?|mild|moderate|severe|minimal|"
    r"trace|marked|significant|top)\b|\b\d+(?:\.\d+)?\s*(?:cm|mm|%)\b",
    re.I,
)
UNCERTAINTY = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|may|might|could|cannot|"
    r"can't|uncertain|equivocal|concerning|suggest\w*|compatible|suspicious|"
    r"convincing)\b",
    re.I,
)
# Claim isolation must not depend only on the supported-finding lexicon.  A
# coordinated phrase such as "no vascular congestion or pleural effusion"
# contains another clinical claim even when that claim is outside FINDINGS.
COMPOUND = re.compile(r"[,;]|\b(?:and|or)\b", re.I)
NEG_PRE = re.compile(
    r"\b(?:no|without|absence of|negative for|free of|clear of|no evidence of)\b",
    re.I,
)
NEG_POST = re.compile(
    r"\b(?:not (?:seen|identified|present|visualized)|absent|excluded|"
    r"no longer (?:seen|present))\b",
    re.I,
)
POS_POST = re.compile(
    r"\b(?:is|are|was|were)?\s*(?:present|seen|identified|noted|demonstrated)\b",
    re.I,
)

# ``FINDINGS`` requires "pleural effusion"; generic "effusion" in a
# pneumothorax coordination is still a second finding and must fail the
# claim-isolation gate.
OTHER_FINDING_PATTERNS = {
    **{key: tuple(pattern for pattern, _ in value) for key, value in FINDINGS.items()},
    "pleural_effusion": (r"\b(?:pleural )?effusions?\b", r"\bpleural fluid\b"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_parts(question: str) -> tuple[str, str, str]:
    marker = "Retrieved report:\n"
    tail = "\nQuestion:\n"
    if question.count(marker) != 1 or question.count(tail) != 1:
        raise ValueError("unexpected source prompt contract")
    before, remainder = question.split(marker, 1)
    report, after = remainder.split(tail, 1)
    return before + marker, report, tail + after


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return trimmed sentence spans while preserving original byte offsets."""
    spans: list[tuple[int, int, str]] = []
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        raw_start, raw_end = start, boundary.start()
        segment = text[raw_start:raw_end]
        left = len(segment) - len(segment.lstrip())
        right = len(segment.rstrip())
        if right > left:
            begin, end = raw_start + left, raw_start + right
            spans.append((begin, end, text[begin:end]))
        start = boundary.end()
    segment = text[start:]
    left = len(segment) - len(segment.lstrip())
    right = len(segment.rstrip())
    if right > left:
        begin, end = start + left, start + right
        spans.append((begin, end, text[begin:end]))
    return spans


def finding_mentions(sentence: str, finding: str) -> list[tuple[re.Match[str], int]]:
    matches: list[tuple[re.Match[str], int]] = []
    for pattern, intrinsic in FINDINGS[finding]:
        matches.extend((match, intrinsic) for match in re.finditer(pattern, sentence, re.I))
    return sorted(matches, key=lambda item: item[0].start())


def direct_polarity(sentence: str, finding: str) -> str | None:
    """Require a direct grammatical assertion independent of the old arm name."""
    matches = finding_mentions(sentence, finding)
    if len(matches) != 1 or UNCERTAINTY.search(sentence) or re.search(r"\bbut\b|;", sentence, re.I):
        return None
    match, intrinsic = matches[0]
    before, after = sentence[: match.start()], sentence[match.end() :]
    if intrinsic < 0:
        # E.g. "Heart size is normal".  Negating an intrinsically negative
        # phrase is not automatically inverted.
        if NEG_PRE.search(before) or NEG_POST.search(after):
            return None
        return "negative"
    if NEG_PRE.search(before) or NEG_POST.search(after):
        return "negative"
    tokens = re.findall(r"\b[\w'-]+\b", sentence)
    if len(tokens) <= 3 and sentence.strip(" .:").lower().endswith(match.group(0).lower()):
        return "positive"
    if POS_POST.search(after):
        return "positive"
    return None


def other_findings(sentence: str, finding: str) -> list[str]:
    found = []
    for other, patterns in OTHER_FINDING_PATTERNS.items():
        if other == finding or not patterns:
            continue
        if any(re.search(pattern, sentence, re.I) for pattern in patterns):
            found.append(other)
    return sorted(set(found))


def assess_report(report: str, finding: str) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    spans = sentence_spans(report)
    claim_spans = [span for span in spans if finding_mentions(span[2], finding)]
    if len(claim_spans) != 1:
        return None, [f"finding_sentence_count:{len(claim_spans)}"]
    begin, end, sentence = claim_spans[0]
    mentions = finding_mentions(sentence, finding)
    if len(mentions) != 1:
        reasons.append(f"finding_mention_count:{len(mentions)}")
    assertions = sentence_assertions(sentence, finding)
    if len(assertions) != 1:
        reasons.append(f"lexicon_assertion_count:{len(assertions)}")
    elif assertions[0] not in {"positive", "negative"}:
        reasons.append(f"lexicon_assertion:{assertions[0]}")
    if TEMPORAL.search(sentence):
        reasons.append("temporal_or_change_semantics")
    if ATTRIBUTE.search(sentence):
        reasons.append("attribute_or_measurement_semantics")
    if UNCERTAINTY.search(sentence):
        reasons.append("uncertainty_semantics")
    if COMPOUND.search(sentence):
        reasons.append("coordinated_or_compound_sentence")
    direct = direct_polarity(sentence, finding)
    if direct is None:
        reasons.append("no_independent_direct_polarity_parse")
    cofindings = other_findings(sentence, finding)
    if cofindings:
        reasons.append("non_finding_claim_in_same_sentence:" + ",".join(cofindings))
    nonclaim = [span for span in spans if span != claim_spans[0]]
    if not nonclaim:
        reasons.append("no_non_finding_sentence_for_random_edit")
    if reasons:
        return None, reasons
    return {
        "report": report,
        "sentence_begin": begin,
        "sentence_end": end,
        "sentence": sentence,
        "direct_polarity": direct,
        "lexicon_polarity": assertions[0],
        "nonclaim_spans": nonclaim,
    }, []


def twin_sentence(finding: str, state_word: str) -> str:
    return f"{DISPLAY[finding]} is explicitly {state_word} in this report."


def replace_span(text: str, begin: int, end: int, replacement: str) -> str:
    return text[:begin] + replacement + text[end:]


def make_question(source_question: str, report: str) -> str:
    before, _, after = prompt_parts(source_question)
    return before + report + after


def main() -> None:
    source_rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source_rows:
        if row["arm"] in SOURCE_ARMS:
            rows_by_pair[row["pair_id"]][row["arm"]] = row
    if len(rows_by_pair) != 32 or any(set(rows) != set(SOURCE_ARMS) for rows in rows_by_pair.values()):
        raise RuntimeError("source is not the frozen 32-pair/two-donor pilot")

    manifest_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    excluded_reasons = Counter()
    selected_by_finding = Counter()
    label_conflicts: list[dict[str, str]] = []

    tokenizer_objects = {
        name: AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True
        )
        for name, path in MODEL_TOKENIZERS.items()
    }

    for pair_id in sorted(rows_by_pair):
        arm_rows = rows_by_pair[pair_id]
        finding = arm_rows[SOURCE_ARMS[0]]["finding"]
        candidates = []
        per_arm: dict[str, Any] = {}
        for source_arm in SOURCE_ARMS:
            row = arm_rows[source_arm]
            _, report, _ = prompt_parts(row["question"])
            candidate, reasons = assess_report(report, finding)
            per_arm[source_arm] = {
                "eligible": candidate is not None,
                "exclusion_reasons": reasons,
            }
            if candidate is not None:
                candidate["source_arm"] = source_arm
                candidate["source_row"] = row
                candidates.append(candidate)

        audit: dict[str, Any] = {
            "pair_id": pair_id,
            "finding": finding,
            "source_arm_audit": per_arm,
            "eligible": bool(candidates),
        }
        if not candidates:
            pair_reasons = sorted(
                set(reason for value in per_arm.values() for reason in value["exclusion_reasons"])
            )
            audit["pair_exclusion_reasons"] = pair_reasons
            excluded_reasons.update(pair_reasons)
            audits.append(audit)
            continue

        selected = min(
            candidates,
            key=lambda item: stable_hash(
                f"{PROTOCOL}:source:{pair_id}:{item['source_arm']}:{item['sentence']}"
            ),
        )
        source_arm = selected["source_arm"]
        source_row = selected["source_row"]
        report = selected["report"]
        begin, end = selected["sentence_begin"], selected["sentence_end"]
        source_sentence = selected["sentence"]
        selected_by_finding[finding] += 1

        expected_arm_polarity = "positive" if source_arm == "present" else "negative"
        if selected["direct_polarity"] != expected_arm_polarity:
            conflict = {
                "pair_id": pair_id,
                "source_arm": source_arm,
                "arm_implied_polarity": expected_arm_polarity,
                "direct_polarity": selected["direct_polarity"],
            }
            label_conflicts.append(conflict)
            audit["legacy_arm_direct_polarity_conflict"] = conflict

        nonclaim_spans = selected["nonclaim_spans"]
        edit_span = min(
            nonclaim_spans,
            key=lambda span: stable_hash(
                f"{PROTOCOL}:non-finding-edit:{pair_id}:{span[0]}:{span[2]}"
            ),
        )
        random_report = replace_span(report, edit_span[0], edit_span[1], RANDOM_EDIT)
        if source_sentence not in random_report:
            raise RuntimeError(f"{pair_id}: random edit changed finding sentence")

        twin_reports: dict[str, str] = {}
        twin_sentences: dict[str, str] = {}
        for arm, word in TWIN_WORDS.items():
            replacement = twin_sentence(finding, word)
            twin_sentences[arm] = replacement
            twin_reports[arm] = replace_span(report, begin, end, replacement)

        # Hard one-variable audit: sentence templates have equal word count and
        # differ only at the state slot; report context outside the span is exact.
        template_tokens = {arm: sentence.split() for arm, sentence in twin_sentences.items()}
        template_lengths = {len(value) for value in template_tokens.values()}
        if template_lengths != {next(iter(template_lengths))}:
            raise RuntimeError(f"{pair_id}: template word counts differ")
        for index in range(next(iter(template_lengths))):
            values = {tokens[index] for tokens in template_tokens.values()}
            if len(values) > 1 and values != set(TWIN_WORDS.values()):
                raise RuntimeError(f"{pair_id}: non-state template token differs")
        varying_slots = [
            index
            for index in range(next(iter(template_lengths)))
            if len({tokens[index] for tokens in template_tokens.values()}) > 1
        ]
        if len(varying_slots) != 1:
            raise RuntimeError(f"{pair_id}: twins do not differ at exactly one word slot")
        prefix_hash = stable_hash(report[:begin])
        suffix_hash = stable_hash(report[end:])
        for arm, twin_report in twin_reports.items():
            replacement = twin_sentences[arm]
            if twin_report[:begin] != report[:begin] or twin_report[begin + len(replacement) :] != report[end:]:
                raise RuntimeError(f"{pair_id}:{arm}: context byte identity failed")

        desired_lexicon_states = {
            "twin_present": "present",
            "twin_absent": "absent",
            "twin_uncertain_withheld": "unresolved",
        }
        observed_states = {
            arm: report_state(value, finding)[0] for arm, value in twin_reports.items()
        }
        if observed_states != desired_lexicon_states:
            raise RuntimeError(f"{pair_id}: generated semantic state audit failed: {observed_states}")
        if report_state(random_report, finding)[0] != report_state(report, finding)[0]:
            raise RuntimeError(f"{pair_id}: random edit changed finding lexicon state")

        questions = {
            **{arm: make_question(source_row["question"], value) for arm, value in twin_reports.items()},
            "original": source_row["question"],
            "plain": make_question(source_row["question"], "[none]"),
            "random_non_target_edit": make_question(source_row["question"], random_report),
        }
        token_audit: dict[str, Any] = {}
        for model_name, tokenizer in tokenizer_objects.items():
            counts = {
                arm: len(tokenizer.encode(questions[arm], add_special_tokens=False))
                for arm in TWIN_WORDS
            }
            token_audit[model_name] = {
                "full_prompt_tokens": counts,
                "max_minus_min": max(counts.values()) - min(counts.values()),
                "exactly_matched": len(set(counts.values())) == 1,
            }

        common = {
            "dataset": source_row["dataset"],
            "finding": finding,
            "img_name": source_row["img_name"],
            "pair_id": pair_id,
            "prompt_contract": PROTOCOL,
            "question_type": "binary_target_blinded",
            "selection_uses_model_output": False,
            "selection_uses_outcome_field": False,
            "source_qid": source_row["source_qid"],
            "task": "target_blind_causal_generation",
        }
        for arm in ARMS:
            manifest_rows.append({
                **common,
                "arm": arm,
                "id": f"{pair_id}:{arm}",
                "qid": f"{pair_id}:{arm}",
                "question": questions[arm],
            })

        audit.update({
            "selected_source_arm": source_arm,
            "source_sentence": source_sentence,
            "source_direct_polarity": selected["direct_polarity"],
            "source_lexicon_polarity": selected["lexicon_polarity"],
            "sentence_span": [begin, end],
            "report_prefix_sha256": prefix_hash,
            "report_suffix_sha256": suffix_hash,
            "twin_varying_word_slot": varying_slots[0],
            "twin_state_words": TWIN_WORDS,
            "twin_lexicon_states": observed_states,
            "random_edit_source_sentence": edit_span[2],
            "random_edit_finding_sentence_byte_identical": True,
            "model_token_audit": token_audit,
        })
        audits.append(audit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    PAIR_AUDIT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audits), encoding="utf-8"
    )
    loaded = load_target_blind_manifest(MANIFEST, limit=0)
    runner_preflight = preflight_inputs(loaded, IMAGE_ROOT)

    selected_audits = [row for row in audits if row["eligible"]]
    token_summary = {}
    for model_name in MODEL_TOKENIZERS:
        gaps = [row["model_token_audit"][model_name]["max_minus_min"] for row in selected_audits]
        token_summary[model_name] = {
            "tokenizer_path": str(MODEL_TOKENIZERS[model_name]),
            "pairs": len(gaps),
            "all_exactly_matched": all(gap == 0 for gap in gaps),
            "maximum_twin_prompt_token_gap": max(gaps, default=None),
        }

    result = {
        "status": "completed_cpu_feasibility_and_semantic_length_audit_no_gpu",
        "protocol": PROTOCOL,
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "selection_contract": {
            "source_pair_count": len(rows_by_pair),
            "source_arms_inspected": list(SOURCE_ARMS),
            "outcome_or_model_output_read": False,
            "fail_closed": True,
            "required": [
                "one finding-bearing sentence in the report",
                "one finding mention and one resolved assertion",
                "independent direct polarity parse",
                "no temporal/change, attribute/location/severity/measurement, or uncertainty semantics",
                "no second supported finding in the same sentence",
                "at least one non-finding sentence for deterministic control edit",
            ],
            "candidate_choice": "minimum SHA256(protocol,pair_id,source_arm,sentence); no outcome/model field",
        },
        "counts": {
            "eligible_pairs": len(selected_audits),
            "excluded_pairs": len(audits) - len(selected_audits),
            "exclusion_rate": (len(audits) - len(selected_audits)) / len(audits),
            "eligible_by_finding": dict(sorted(selected_by_finding.items())),
            "manifest_rows": len(manifest_rows),
            "arms_per_pair": len(ARMS),
            "rows_by_arm": dict(Counter(row["arm"] for row in manifest_rows)),
        },
        "exclusion_reason_pair_incidence": dict(sorted(excluded_reasons.items())),
        "legacy_arm_direct_polarity_conflicts": label_conflicts,
        "twin_intervention_audit": {
            "causal_arms": list(TWIN_WORDS),
            "only_varying_whitespace_token": TWIN_WORDS,
            "report_context_prefix_suffix_byte_identical": True,
            "original_plain_random_are_diagnostic_controls_not_members_of_single_variable_contrast": True,
            "random_edit_preserves_finding_sentence_bytes": True,
        },
        "model_token_audit": token_summary,
        "runner_preflight_cpu_only": runner_preflight,
        "artifacts": {
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "pair_audit_jsonl": str(PAIR_AUDIT),
            "pair_audit_sha256": sha256_file(PAIR_AUDIT),
        },
        "gpu_execution": "not_run",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
