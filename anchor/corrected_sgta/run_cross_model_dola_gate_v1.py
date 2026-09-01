"""Run token-identity and functional DoLa gates on native medical VLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .cross_model_dola import generate_dola
from .models_oe import load_oe_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("huatuo", "hulu", "qwen"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    rows = [row for row in rows if (args.image_root / row["img_name"]).is_file()][: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adapter = load_oe_adapter(args.model)
    exact = changed = 0
    with args.output.open("w") as handle:
        for index, row in enumerate(rows, start=1):
            with Image.open(args.image_root / row["img_name"]) as source:
                image = source.convert("RGB")
            prompt = str(row["question"]).strip() + " Answer the question directly."
            native = adapter.generate_control(
                image, prompt, do_sample=False, temperature=0.7, top_p=0.9,
                num_beams=1, max_new_tokens=args.max_new_tokens, seed=args.seed,
            )
            off, off_audit = generate_dola(
                adapter, image, prompt, max_new_tokens=args.max_new_tokens,
                seed=args.seed, dola_layers=None,
            )
            dola, audit = generate_dola(
                adapter, image, prompt, max_new_tokens=args.max_new_tokens,
                seed=args.seed, dola_layers="low",
            )
            record = {
                "protocol_version": "cross-model-dola-gate-v1",
                "model": args.model,
                "qid": str(row.get("qid", row.get("id", index))),
                "question": row["question"],
                "answer": row.get("answer"),
                "native": {"text": native.text, "token_ids": list(native.token_ids)},
                "off": {"text": off.text, "token_ids": list(off.token_ids), "audit": off_audit},
                "dola": {"text": dola.text, "token_ids": list(dola.token_ids), "audit": audit},
                "off_token_exact": native.token_ids == off.token_ids,
                "dola_changed": native.token_ids != dola.token_ids,
            }
            exact += int(record["off_token_exact"])
            changed += int(record["dola_changed"])
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(rows)}] {record['qid']} identity={record['off_token_exact']} "
                f"changed={record['dola_changed']}", flush=True,
            )
    print(json.dumps({"model": args.model, "n": len(rows), "off_exact": exact, "changed": changed}), flush=True)
    adapter.close()


if __name__ == "__main__":
    main()
