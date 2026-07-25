from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass


@dataclass
class JudgeResult:
    score: float
    raw: str
    backend: str = "opencode"


def judge(prompt: str, *, mock: bool = False) -> JudgeResult:
    """Run an optional opencode-style judge.

    Set ``OPENCODE_JUDGE_CMD`` to a command that accepts the prompt on stdin and
    returns either JSON with ``score`` or free text. In smoke/mock mode this
    returns a neutral score without external calls.
    """
    if mock or os.environ.get("OPENCODE_JUDGE_MOCK") == "1":
        return JudgeResult(score=0.0, raw="mock")
    command = os.environ.get("OPENCODE_JUDGE_CMD")
    if not command:
        raise RuntimeError("OPENCODE_JUDGE_CMD is not configured")
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        shell=True,
        check=False,
        capture_output=True,
    )
    raw = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or raw)
    try:
        payload = json.loads(raw)
        score = float(payload.get("score", 0.0))
    except Exception:
        score = 0.0
    return JudgeResult(score=score, raw=raw)
