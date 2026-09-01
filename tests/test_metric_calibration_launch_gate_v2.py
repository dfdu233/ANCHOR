import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_launcher_fails_closed_on_stop_after_n8(tmp_path):
    analyses = {}
    for name in ("qwen", "huatuo"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "runtime_admissible": True,
                    "runtime": {
                        "structured_json_valid_rate": 1.0,
                        "structured_nonempty_rate": 1.0,
                        "structured_cap_hit_rate": 0.0,
                    },
                }
            )
        )
        analyses[name] = path
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "decision": "STOP_AFTER_N8",
                "n97_authorized": False,
                "gpu_authorized": False,
                "models": {
                    "qwen_parent": {"analysis_sha256": digest(analyses["qwen"])},
                    "huatuo_medical": {"analysis_sha256": digest(analyses["huatuo"])},
                },
            }
        )
    )
    decision = tmp_path / "launch.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/launch_metric_calibration_full_after_pilot_v2.py"),
            "--qwen",
            str(analyses["qwen"]),
            "--huatuo",
            str(analyses["huatuo"]),
            "--pilot-decision",
            str(pilot),
            "--decision",
            str(decision),
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 2
    value = json.loads(decision.read_text())
    assert value["decision"] == "STOP_FAIL_CLOSED_GATE"
    assert value["pilot_pass"] is False
    assert value["launch_admitted"] is False
