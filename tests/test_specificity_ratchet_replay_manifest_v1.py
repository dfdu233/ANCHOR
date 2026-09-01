import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from corrected_sgta.compile_specificity_ratchet_replay_manifest_v1 import (
    _case_swap_plan,
    _label_blind_case_pool,
    _require_g0_role_closure,
    build_replay_rows,
    compile_replay_manifest,
)
from corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    AdjudicationValidationError,
)


class SpaceTokenizer:
    is_fast = True

    class Encoding(dict):
        def __getattr__(self, name):
            return self[name]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        offsets = []
        ids = []
        cursor = 0
        for index, word in enumerate(text.split()):
            start = text.index(word, cursor)
            end = start + len(word)
            cursor = end
            offsets.append((start, end))
            ids.append(100 + index)
        payload = {"input_ids": ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return self.Encoding(payload)


def _candidate(index):
    side = "left" if index % 2 == 0 else "right"
    child = f"A {side} lesion is present."
    return {
        "case_id": f"CASE-{index}",
        "edge_id": f"EDGE-{index}",
        "question": "What is present?",
        "image_relpath": f"test_images/{index}.jpg",
        "answer_span": child,
        "parent_proposal": "A lesion is present.",
        "child_proposal": child,
        "added_constraint_proposal": side,
        "edge_type": "laterality",
        "modality_stratum": "XR",
        "anatomy_stratum": "thorax",
        "answer_length_stratum": "short_le_50",
        "observability_screen": "potentially_single_image_decidable",
        "prompt_requested_increment": False,
        "proposal_only": True,
    }


def _fake_inputs(tmp_path, count=8):
    pack = tmp_path / "pack"
    pack.mkdir()
    candidates = [_candidate(index) for index in range(count)]
    provenance = []
    answers = []
    finals = {}
    tokenizer = SpaceTokenizer()
    for line, candidate in enumerate(candidates, start=1):
        answer = "Clinical context. " + candidate["child_proposal"] + " End."
        qid = f"qid-{line}"
        visible = tokenizer(answer).input_ids
        answers.append(
            {
                "question_id": qid,
                "text": answer,
                "model_id": "huatuo",
                "metadata": {"generated_token_ids": visible},
            }
        )
        provenance.append(
            {
                "case_id": candidate["case_id"],
                "edge_id": candidate["edge_id"],
                "question_id": qid,
                "source_model": "huatuo",
                "source_answer_path": "answers.jsonl",
                "source_answer_line": line,
            }
        )
        finals[candidate["edge_id"]] = {
            "final_edge_entailment_admitted": "yes",
            "final_parent_visual_support": "supported",
            "final_child_visual_support": "supported" if line % 2 else "undetermined",
            "final_increment_observability": "observable_on_supplied_image",
        }
    (pack / "provenance.private.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance)
    )
    (tmp_path / "answers.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in answers)
    )
    validated = SimpleNamespace(candidates=tuple(candidates), final_rows=finals)
    return pack, validated, tokenizer


def test_replay_rows_bind_full_answer_and_two_exact_cell_swaps(tmp_path):
    pack, validated, tokenizer = _fake_inputs(tmp_path)
    rows, exclusions, _ = build_replay_rows(
        validated=validated, pack=pack, repo=tmp_path, tokenizer=tokenizer
    )
    assert not exclusions
    assert len(rows) == 8
    for row in rows:
        assert row["full_visible_answer"].startswith("Clinical context.")
        assert row["model_input_contract"] == "complete frozen visible OE answer only"
        assert row["native_generation_ids_certified"] is False
        assert len(row["matched_image_swaps"]) == 2
        assert len({swap["case_id"] for swap in row["matched_image_swaps"]}) == 2
        assert all(swap["case_id"] != row["case_id"] for swap in row["matched_image_swaps"])
        constraint = row["constraint_char_spans_in_full_answer"][0]
        assert row["full_visible_answer"][constraint["char_start"]:constraint["char_end_exclusive"]] in {"left", "right"}


def test_replay_rows_refuse_cross_model_source_or_token_identity_drift(tmp_path):
    pack, validated, tokenizer = _fake_inputs(tmp_path)
    provenance_path = pack / "provenance.private.jsonl"
    provenance = [json.loads(line) for line in provenance_path.read_text().splitlines()]
    provenance[0]["source_model"] = "hulu"
    provenance_path.write_text("".join(json.dumps(row) + "\n" for row in provenance))
    with pytest.raises(ValueError, match="non-Huatuo output"):
        build_replay_rows(
            validated=validated, pack=pack, repo=tmp_path, tokenizer=tokenizer
        )

    provenance[0]["source_model"] = "huatuo"
    provenance_path.write_text("".join(json.dumps(row) + "\n" for row in provenance))
    answers_path = tmp_path / "answers.jsonl"
    answers = [json.loads(line) for line in answers_path.read_text().splitlines()]
    answers[0]["metadata"]["generated_token_ids"] = [999]
    answers_path.write_text("".join(json.dumps(row) + "\n" for row in answers))
    with pytest.raises(ValueError, match="visible-text token provenance drift"):
        build_replay_rows(
            validated=validated, pack=pack, repo=tmp_path, tokenizer=tokenizer
        )


def test_swap_plan_never_relaxes_sparse_exact_cell():
    rows = []
    for index in range(3):
        rows.append(
            {
                "case_id": f"case-{index}",
                "image_relpath": f"{index}.jpg",
                "split": "dev",
                "modality_stratum": "XR",
                "anatomy_stratum": "thorax" if index < 2 else "neuro",
            }
        )
    plan, exclusions = _case_swap_plan(rows)
    assert plan == {}
    assert {row["case_id"] for row in exclusions} == {"case-0", "case-1", "case-2"}


def test_g0_refuses_one_sided_role_composition_before_canary():
    rows = [
        {"split": split, "scientific_role": "supported_specificity_control"}
        for split in ("dev", "test")
    ]
    with pytest.raises(ValueError, match="G0 failed"):
        _require_g0_role_closure(rows)


def test_label_blind_split_is_cell_balanced_and_outcome_independent():
    candidates = [_candidate(index) for index in range(7)]
    first = _label_blind_case_pool(candidates)
    second = _label_blind_case_pool(list(reversed(candidates)))
    assert first == second
    counts = {split: sum(row["split"] == split for row in first) for split in ("dev", "test")}
    assert sorted(counts.values()) == [3, 4]


def test_real_blank_pack_refuses_before_loading_tokenizer_or_writing(tmp_path):
    pack = Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2")
    with pytest.raises(AdjudicationValidationError):
        compile_replay_manifest(
            pack=pack,
            repo=Path.cwd(),
            tokenizer_dir=Path("/definitely/not/read"),
            output=tmp_path / "samples.jsonl",
            metadata_output=tmp_path / "metadata.json",
        )
    assert not (tmp_path / "samples.jsonl").exists()
    assert not (tmp_path / "metadata.json").exists()
