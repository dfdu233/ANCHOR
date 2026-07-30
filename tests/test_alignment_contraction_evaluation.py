import torch

from anchor.corrected_sgta.evaluate_alignment_contraction import (
    cluster_bootstrap_mean_interval,
    per_sample_sequence_nll,
)


def test_per_sample_sequence_nll_masks_prompt_and_padding() -> None:
    logits = torch.zeros(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2],
            [-100, 0, -100, -100],
        ]
    )
    logits[0, 1, 1] = 4.0
    logits[0, 2, 2] = 4.0
    logits[1, 0, 0] = 4.0
    nll, counts = per_sample_sequence_nll(logits, labels)
    assert counts.tolist() == [2, 1]
    assert torch.all(nll < 0.04)


def test_per_sample_sequence_nll_ignores_unsupervised_logits() -> None:
    logits = torch.zeros(1, 3, 2)
    labels = torch.tensor([[-100, 1, -100]])
    baseline, _ = per_sample_sequence_nll(logits, labels)
    logits[0, 1, 0] = 100.0
    changed, _ = per_sample_sequence_nll(logits, labels)
    assert torch.equal(baseline, changed)


def test_cluster_bootstrap_keeps_constant_difference() -> None:
    interval = cluster_bootstrap_mean_interval(
        torch.tensor([0.5, 0.5, 0.5]).numpy(),
        ["a", "a", "b"],
        draws=100,
    )
    assert interval == [0.5, 0.5]
