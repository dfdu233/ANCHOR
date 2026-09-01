import json
from pathlib import Path

from anchor.medeval.diagnose_llava_port import main


def test_port_summary_separates_content_from_function_fragments(tmp_path: Path, monkeypatch):
    canonical = tmp_path / "canonical.jsonl"
    port = tmp_path / "port.jsonl"
    canonical.write_text("".join(
        json.dumps({"text": value, "metadata": {"generated_token_count": 2}}) + "\n"
        for value in ("right side", "not visible")
    ))
    port.write_text("".join(
        json.dumps({"text": value, "metadata": {"decoded_sequence_token_count": 1}}) + "\n"
        for value in ("The", "In")
    ))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        ["diagnose", "summarize", "--answers", f"canonical={canonical}",
         f"port_test={port}", "--output", str(output)],
    )
    main()
    payload = json.loads(output.read_text())
    assert payload["port_failure_confirmed"] is True
