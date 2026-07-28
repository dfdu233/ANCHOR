import json
import tempfile
import unittest
from pathlib import Path

from anchor.corrected_sgta.run_anchor_let_report import (
    build_payload,
    load_records,
)


class LetReportProtocolTest(unittest.TestCase):
    def test_manifest_selection_is_dataset_balanced_and_reference_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"not-decoded-by-loader")
            records = []
            for dataset in ("iuxray", "mimic"):
                for index in range(3):
                    records.append(
                        {
                            "dataset": dataset,
                            "id": f"{dataset}-{index}",
                            "image": str(image),
                            "reference": f"reference {index}",
                        }
                    )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"fingerprint": "x", "records": records}))
            _, selected = load_records(manifest, ("iuxray", "mimic"), 2)
            self.assertEqual(
                [row["dataset"] for row in selected],
                ["iuxray", "iuxray", "mimic", "mimic"],
            )

    def test_payload_keeps_ce_interface_disabled(self):
        row = {
            "dataset": "mimic",
            "id": "1",
            "image": "image.png",
            "ground_truth": "reference",
            "prompt": "report prompt",
            "baseline_text": "baseline",
            "let_text": "let",
            "baseline_normal_template": False,
            "let_normal_template": False,
            "baseline_token_count": 10,
            "let_token_count": 11,
        }
        payload = build_payload([row], "fingerprint", {"ce_parser_used": False})
        self.assertFalse(payload["ground_truth_used_for_generation_or_selection"])
        self.assertEqual(payload["records"][0]["candidates"]["let"], "let")


if __name__ == "__main__":
    unittest.main()
