from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
import pytest

import anchor.corrected_sgta.cecd_dual_semantics_ce_adapter_v1 as adapter
from anchor.corrected_sgta.cecd_dual_semantics_kernels_v1 import (
    IMPLEMENTED_KERNEL_METHODS,
)


class FakeScorer:
    def __init__(self, model_family: str = "huatuo") -> None:
        self.model_family = model_family
        self.calls = 0

    def score(self, image, prompt):
        self.calls += 1
        base = float(image) + float(prompt)
        return {
            "logits": {
                "supported": base + 1.0,
                "refuted": -base,
                "undetermined": 0.25 * base,
            },
            "prediction": "supported",
        }

    def standard_next_token(self, image, prompt):
        return self.score(image, prompt)


def _cell_rows(offsets=None):
    offsets = offsets or {cell: 0.0 for cell in adapter.CELL_ORDER}
    values = {
        "h00": [1.0, -1.0, 0.0],
        "h10": [1.5, -1.0, -0.5],
        "h01": [1.0, -1.5, 0.5],
        "h11": [3.0, -2.0, -1.0],
    }
    return {
        cell: {
            "scores": {
                "logits": {
                    state: values[cell][index] + offsets[cell]
                    for index, state in enumerate(adapter.STATES)
                }
            }
        }
        for cell in adapter.CELL_ORDER
    }


def test_centered_ce_controls_are_gauge_invariant_and_closed() -> None:
    baseline = adapter.summarize_control_logits(_cell_rows(), 42)
    shifted = adapter.summarize_control_logits(
        _cell_rows({"h00": 100, "h10": -20, "h01": 7, "h11": 3}), 42
    )
    assert list(baseline) == list(IMPLEMENTED_KERNEL_METHODS)
    for method in IMPLEMENTED_KERNEL_METHODS:
        assert baseline[method]["prediction"] == shifted[method]["prediction"]
        np.testing.assert_allclose(
            list(baseline[method]["centered_logits"].values()),
            list(shifted[method]["centered_logits"].values()),
            atol=1e-12,
        )
        assert sum(baseline[method]["centered_logits"].values()) == pytest.approx(
            0.0, abs=1e-12
        )
        assert sum(baseline[method]["probabilities"].values()) == pytest.approx(1.0)
    orbit = {
        cell: adapter.centered_logit_vector(_cell_rows()[cell]["scores"])
        for cell in adapter.CELL_ORDER
    }
    parts = adapter.factorial_components(orbit)
    additive = parts.grand + parts.render + parts.prompt
    randomized = np.asarray(
        list(baseline["random_norm"]["centered_logits"].values())
    )
    assert np.linalg.norm(randomized - additive) == pytest.approx(
        np.linalg.norm(parts.interaction), abs=1e-12
    )


def test_atomic_cells_resume_only_valid_completed_shards(tmp_path: Path) -> None:
    scorer = FakeScorer()
    images = {cell: index + 1 for index, cell in enumerate(adapter.CELL_ORDER)}
    prompts = {cell: str(index + 1) for index, cell in enumerate(adapter.CELL_ORDER)}
    rows, newly = adapter.score_atomic_cells(
        scorer=scorer,
        images=images,
        prompts=prompts,
        output_dir=tmp_path,
        config_fingerprint="f" * 64,
    )
    assert newly == 4
    assert scorer.calls == 4
    assert set(rows) == set(adapter.CELL_ORDER)
    rows, newly = adapter.score_atomic_cells(
        scorer=scorer,
        images=images,
        prompts=prompts,
        output_dir=tmp_path,
        config_fingerprint="f" * 64,
    )
    assert newly == 0
    assert scorer.calls == 4
    (tmp_path / "cells/h10.json").write_text("corrupt")
    _, newly = adapter.score_atomic_cells(
        scorer=scorer,
        images=images,
        prompts=prompts,
        output_dir=tmp_path,
        config_fingerprint="f" * 64,
    )
    assert newly == 1
    assert scorer.calls == 5


def test_atomic_cell_rejects_nonfinite_or_incomplete_logits(tmp_path: Path) -> None:
    class Bad(FakeScorer):
        def score(self, image, prompt):
            return {"logits": {"supported": float("nan")}}

    with pytest.raises(adapter.CEAdapterError, match="exactly three"):
        adapter.score_atomic_cells(
            scorer=Bad(),
            images={cell: 1 for cell in adapter.CELL_ORDER},
            prompts={cell: "1" for cell in adapter.CELL_ORDER},
            output_dir=tmp_path,
            config_fingerprint="a" * 64,
        )
    assert not list(tmp_path.glob("cells/*.json"))


def test_config_is_write_once_and_resume_rejects_drift(tmp_path: Path) -> None:
    candidate = {"version": adapter.VERSION, "created_at": "t1", "frozen": 1}
    first = adapter.freeze_or_resume_config(
        output_dir=tmp_path, candidate=candidate, resume=False
    )
    replay = adapter.freeze_or_resume_config(
        output_dir=tmp_path,
        candidate={**candidate, "created_at": "t2"},
        resume=True,
    )
    assert replay == first
    with pytest.raises(adapter.CEAdapterError, match="config drift"):
        adapter.freeze_or_resume_config(
            output_dir=tmp_path,
            candidate={**candidate, "frozen": 2},
            resume=True,
        )


def test_gpu_flock_is_nonblocking_and_fail_closed(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    lock.touch()
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(adapter.CEAdapterError, match="GPU lock is busy"):
            with adapter.gpu_flock(lock):
                raise AssertionError("unreachable")


def test_engineering_smoke_never_becomes_formal_or_oe(
    tmp_path: Path, monkeypatch
) -> None:
    images = {cell: index + 1 for index, cell in enumerate(adapter.CELL_ORDER)}
    prompts = {cell: str(index + 1) for index, cell in enumerate(adapter.CELL_ORDER)}
    preflight = {
        "version": adapter.VERSION,
        "status": "cpu_preflight_passed_no_model_or_cuda",
        "scientific_status": "engineering_only_no_scientific_authorization",
        "orbit": {"record_key": "one"},
        "fingerprint": "p" * 64,
    }
    monkeypatch.setattr(
        adapter, "cpu_preflight", lambda **_: (preflight, images, prompts)
    )
    scorer = FakeScorer()
    monkeypatch.setattr(adapter, "build_real_scorer", lambda *_: scorer)
    result = adapter.run_engineering_smoke(
        family="huatuo",
        manifest=tmp_path / "unused.jsonl",
        image_root=tmp_path,
        model_dir=tmp_path,
        output_dir=tmp_path / "out",
        record_key=None,
        render_names=["r0", "r1"],
        prompt_names=["p0", "p1"],
        seed=42,
        gpu_lock_path=tmp_path / "gpu.lock",
        resume=False,
    )
    assert result["status"] == "engineering_smoke_complete"
    assert result["formal_method_output_authorized"] is False
    assert result["oe_adapter_implemented"] is False
    assert result["cecd_hidden_intervention_implemented"] is False
    assert result["treble_variants_implemented"] is False
    assert result["paper_claim_authorized"] is False
    assert scorer.calls == 4
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary == result


def test_formal_ce_seven_controls_share_one_four_cell_cache_and_resume(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(30):
        image = image_root / f"image_{index}.dcm"
        image.write_bytes(b"synthetic-dicom-placeholder")
        row = {
            "cluster_id": f"cluster_{index}",
            "image_id": f"image_{index}",
            "finding": "pleural_effusion",
            "image_path": str(image),
        }
        row["record_key"] = adapter._record_key(row)
        rows.append(row)

    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    record_keys = tmp_path / "record_keys.json"
    record_keys.write_text(
        json.dumps(
            {
                "schema_version": "cecd-dual-semantics-record-keys-v1",
                "record_keys": [row["record_key"] for row in rows],
            }
        ),
        encoding="utf-8",
    )
    claim_contract = tmp_path / "claim_contract.json"
    claim_contract.write_text(
        json.dumps(
            {
                "schema_version": adapter.FORMAL_CE_CLAIM_CONTRACT_SCHEMA,
                "task": "fixed_claim_single_token_ce",
                "render_names": ["render_a", "render_b"],
                "prompt_names": ["prompt_a", "prompt_b"],
                "seed": 42,
                "image_root": str(image_root),
                "minimum_clusters": 30,
            }
        ),
        encoding="utf-8",
    )
    worker_path = Path(adapter.__file__).resolve()
    descriptor = {"model_family": "hulu", "runtime": "fake-scorer-v1"}
    run_contract = {
        "fingerprint": "f" * 64,
        "models": {"hulu": {"model_id": "hulu:fake"}},
        "runtime_descriptors": {"hulu": descriptor},
        "source_files": {"worker": adapter.file_record(worker_path)},
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter.py").write_text("# fake source closure\n", encoding="utf-8")

    scorer = FakeScorer("hulu")
    factory_calls = []

    def scorer_factory(family: str, path: Path) -> FakeScorer:
        factory_calls.append((family, path))
        return scorer

    def orbit_provider(**kwargs):
        index = int(str(kwargs["row"]["image_id"]).split("_")[-1])
        images = {cell: index + offset for offset, cell in enumerate(adapter.CELL_ORDER)}
        prompts = {
            cell: str(offset + 1) for offset, cell in enumerate(adapter.CELL_ORDER)
        }
        return images, prompts, {"fake": True}

    shared_cache = tmp_path / "shared_cache"
    completions = []
    for method in IMPLEMENTED_KERNEL_METHODS:
        completions.append(
            adapter.run_formal_ce_method(
                family="hulu",
                method=method,
                model_dir=model_dir,
                evaluation_manifest=evaluation,
                record_keys_path=record_keys,
                claim_contract_path=claim_contract,
                run_contract=run_contract,
                shared_cache_root=shared_cache,
                output_dir=tmp_path / "arms" / method,
                scorer_factory=scorer_factory,
                orbit_provider=orbit_provider,
            )
        )

    assert factory_calls == [("hulu", model_dir)]
    assert scorer.calls == 30 * 4
    assert len({row["raw_cache_manifest"]["sha256"] for row in completions}) == 1
    assert all(row["rows"] == 30 and row["clusters"] == 30 for row in completions)
    assert all(row["oe_implemented"] is False for row in completions)
    assert all(row["hidden_intervention_implemented"] is False for row in completions)
    assert all(row["paper_native_treble_claimed"] is False for row in completions)
    assert len(list(shared_cache.glob("records/*/cells/*.json"))) == 30 * 4
    cached_cell = json.loads(next(shared_cache.glob("records/*/cells/*.json")).read_text())
    assert cached_cell["scientific_status"] == "formal_ce_raw_logit_cache_only"

    replay = adapter.run_formal_ce_method(
        family="hulu",
        method=IMPLEMENTED_KERNEL_METHODS[0],
        model_dir=model_dir,
        evaluation_manifest=evaluation,
        record_keys_path=record_keys,
        claim_contract_path=claim_contract,
        run_contract=run_contract,
        shared_cache_root=shared_cache,
        output_dir=tmp_path / "arms" / IMPLEMENTED_KERNEL_METHODS[0],
        scorer_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("resume must not reload the scorer")
        ),
        orbit_provider=lambda **_: (_ for _ in ()).throw(
            AssertionError("resume must not rebuild the orbit")
        ),
    )
    assert replay == completions[0]
    assert scorer.calls == 30 * 4
