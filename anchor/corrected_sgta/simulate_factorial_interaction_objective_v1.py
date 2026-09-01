"""Synthetic sanity check for interaction-only versus full consistency loss.

The simulation encodes a case-level label observed under a 2x2 factorial
design.  Both factors legitimately change evidence clarity (main effects),
while one training-only shortcut appears only in the joint (1,1) cell and
reverses out of domain.  FIN should remove the joint shortcut without forcing
all four clarity levels to be identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


OUT = Path("corrected_runs/factorial_interaction_simulation_v1/result.json")


def make_data(
    n: int, seed: int, reverse_shortcut: bool, clarity_main_strength: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    sign = 2 * y - 1
    rows, soft = [], []
    for case in range(n):
        cells, targets = [], []
        base_visual = sign[case] + rng.normal(0, 1.0)
        base_knowledge = sign[case] + rng.normal(0, 1.0)
        for a in (0, 1):
            for b in (0, 1):
                visual = base_visual + a * sign[case] * 0.65 + rng.normal(0, 0.35)
                knowledge = base_knowledge + b * sign[case] * 0.65 + rng.normal(0, 0.35)
                joint = 0.0
                if a == 1 and b == 1:
                    joint = sign[case] * (2.5 if not reverse_shortcut else -2.5) + rng.normal(0, 0.2)
                cells.append([visual, knowledge, float(a), float(b), joint])
                clarity = 0.25 + clarity_main_strength * a + clarity_main_strength * b
                targets.append(1.0 / (1.0 + np.exp(-sign[case] * clarity)))
        rows.append(cells)
        soft.append(targets)
    return (
        torch.tensor(np.asarray(rows), dtype=torch.float32),
        torch.tensor(np.asarray(soft), dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def metrics(logits: torch.Tensor, y: torch.Tensor, soft: torch.Tensor) -> dict:
    probability = logits.sigmoid()
    pred = (probability >= 0.5).float()
    hard = y[:, None].expand_as(pred)
    positive = hard == 1
    negative = ~positive
    bacc = 0.5 * ((pred[positive] == 1).float().mean() + (pred[negative] == 0).float().mean())
    return {
        "balanced_accuracy": float(bacc),
        "brier_reader_clarity": float(((probability - soft) ** 2).mean()),
        "mixed_logit_magnitude": float(
            (logits[:, 3] - logits[:, 2] - logits[:, 1] + logits[:, 0]).abs().mean()
        ),
    }


def fit(method: str, seed: int, clarity_main_strength: float) -> dict:
    torch.manual_seed(seed)
    train_x, train_soft, train_y = make_data(3000, 1000 + seed, False, clarity_main_strength)
    test_x, test_soft, test_y = make_data(2000, 2000 + seed, True, clarity_main_strength)
    model = MLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(250):
        logits = model(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_soft)
        if method == "full_consistency":
            loss = loss + 0.5 * ((logits - logits.mean(dim=1, keepdim=True)) ** 2).mean()
        elif method == "fin":
            mixed = logits[:, 3] - logits[:, 2] - logits[:, 1] + logits[:, 0]
            loss = loss + 0.5 * (mixed**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        return metrics(model(test_x), test_y, test_soft)


def main() -> None:
    methods = ("erm", "full_consistency", "fin")
    strengths = (0.0, 0.5, 1.0, 2.0)
    runs, summary = {}, {}
    for strength in strengths:
        strength_key = str(strength)
        runs[strength_key] = {
            method: [fit(method, seed, strength) for seed in range(5)] for method in methods
        }
        summary[strength_key] = {}
        for method, values in runs[strength_key].items():
            summary[strength_key][method] = {
                key: {
                    "mean": float(np.mean([row[key] for row in values])),
                    "std": float(np.std([row[key] for row in values], ddof=1)),
                }
                for key in values[0]
            }
    interpretation = (
        "A strength sweep is required because FIN has no reason to dominate full consistency "
        "when factor main effects do not encode legitimate reader-clarity differences."
    )
    result = {
        "status": "synthetic_objective_sanity_check_not_empirical_evidence",
        "setting": "clarity main-effect sweep plus a joint shortcut that reverses OOD",
        "interpretation": interpretation,
        "runs": runs,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
