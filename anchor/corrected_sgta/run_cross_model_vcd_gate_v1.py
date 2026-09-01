"""Run token-identity and functional VCD gates on native medical VLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .cross_model_vcd import generate_vcd
from .models_oe import load_oe_adapter
from .protocol_v2 import build_prompt


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("huatuo", "hulu", "qwen"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text())
    rows = [
        row
        for row in rows
        if (args.image_root / str(row.get("img_name", ""))).is_file()
    ][: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.output.is_file():
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["qid"])] = row
    adapter = load_oe_adapter(args.model)
    exact = changed = 0
    with args.output.open("a") as handle:
        for index, row in enumerate(rows, start=1):
            qid = str(row.get("qid", row.get("id", index)))
            if qid in done:
                result = done[qid]
            else:
                with Image.open(args.image_root / row["img_name"]) as source:
                    image = source.convert("RGB")
                prompt = build_prompt(row)
                native = adapter.generate_control(
                    image,
                    prompt,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                off, off_audit = generate_vcd(
                    adapter,
                    image,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    alpha=0.0,
                    beta=1.0,
                    sample=False,
                )
                vcd, vcd_audit = generate_vcd(
                    adapter,
                    image,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    alpha=1.0,
                    beta=0.1,
                    noise_step=500,
                    sample=False,
                )
                result = {
                    "version": "cross-model-vcd-gate-v1",
                    "model": args.model,
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": _sha(args.manifest),
                    "qid": qid,
                    "img_name": row["img_name"],
                    "question": row["question"],
                    "answer": row.get("answer"),
                    "native": {
                        "text": native.text,
                        "token_ids": list(native.token_ids),
                        "token_count": native.token_count,
                    },
                    "off": {
                        "text": off.text,
                        "token_ids": list(off.token_ids),
                        "token_count": off.token_count,
                        "audit": off_audit,
                    },
                    "vcd": {
                        "text": vcd.text,
                        "token_ids": list(vcd.token_ids),
                        "token_count": vcd.token_count,
                        "audit": vcd_audit,
                    },
                    "off_token_exact": native.token_ids == off.token_ids,
                    "vcd_changed": native.token_ids != vcd.token_ids,
                }
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
            exact += int(result["off_token_exact"])
            changed += int(result["vcd_changed"])
            print(
                f"[{index}/{len(rows)}] {qid} identity={result['off_token_exact']} "
                f"changed={result['vcd_changed']} native={result['native']['token_count']} "
                f"vcd={result['vcd']['token_count']}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "model": args.model,
                "n": len(rows),
                "off_token_exact": exact,
                "vcd_changed": changed,
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    adapter.close()


if __name__ == "__main__":
    main()
