#!/usr/bin/env python3
"""Fatal canary for query-conditioned visual-token topology.

Decoder-only VLMs commonly serialize image tokens before the user question.
Under a causal mask this makes every visual-token hidden state structurally
independent of the question.  This probe changes only the placeholder order:

    native:      <image> question -> answer
    query-first: question <image> -> answer

No weights, pixels, verbalizers, or answer budget change.  A failure closes
the reserialization branch before any architectural story is built around it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    IGNORE_INDEX,
    dicom_to_pil,
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    prompt_for,
)


VERSION = "huatuo-query-first-topology-probe-v1"
ORDERS = ("image_first", "query_first")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def stable_key(seed: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}:{row['record_key']}".encode()).hexdigest()


def load_cases(path: Path, per_bin: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    output: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for votes in (0, 3):
        cell = sorted(
            [row for row in rows if int(row["positive_votes"]) == votes],
            key=lambda row: stable_key(seed, row),
        )
        chosen = []
        for row in cell:
            if row["image_id"] in used_images:
                continue
            chosen.append(row)
            used_images.add(row["image_id"])
            if len(chosen) == per_bin:
                break
        if len(chosen) != per_bin:
            raise ValueError(f"vote bin {votes} has only {len(chosen)} image-disjoint rows")
        output.extend(chosen)
    return sorted(output, key=lambda row: stable_key(seed + 1, row))


def query_first_embeddings(
    bot: Any, prompt: str, image_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    # Keep the placeholder inside the same user turn, but move it after all
    # question tokens.  The assistant marker remains after both modalities.
    text = f"{prompt}\n<image>"
    input_ids = bot.preprocess(bot.get_conv_without_history(text), return_tensors="pt").to(
        bot.model.device
    )
    image_positions = torch.where(input_ids < 0)[0]
    if image_positions.numel() != 1:
        raise RuntimeError("query-first prompt must contain one image placeholder")
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    _, positions, expanded_attention, _, embeddings, _ = (
        bot.model.prepare_inputs_labels_for_multimodal_new(
            [input_ids], None, [attention], None, [labels], image_tensor
        )
    )
    if embeddings is None:
        raise RuntimeError("multimodal expansion returned no embeddings")
    start = int(image_positions.item())
    count = int(embeddings.shape[1] - input_ids.numel() + 1)
    return embeddings, expanded_attention, positions, (start, start + count)


@torch.inference_mode()
def topology_mask(
    embeddings: torch.Tensor,
    visual_span: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    length = embeddings.shape[1]
    if mode == "visual_read":
        # Preserve the native causal graph everywhere except visual-token query
        # rows, which may read the whole already-observed multimodal prompt.
        # CUDA does not implement ``tril`` for bfloat16 on the deployed
        # PyTorch build.  Construct the binary topology in float32 and only
        # then cast it to the model dtype.
        mask = torch.tril(torch.ones((length, length), device=embeddings.device, dtype=torch.float32))
        start, end = visual_span
        mask[start:end, :] = 1
    elif mode == "full_prefix":
        mask = torch.ones((length, length), device=embeddings.device, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported topology mask: {mode}")
    return mask.to(dtype=embeddings.dtype).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def score(bot: Any, ids: dict[str, int], embeddings: torch.Tensor, attention: torch.Tensor, positions: torch.Tensor | None) -> dict[str, Any]:
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    logits = layer_logits(bot, hidden, (), ids)[len(hidden) - 1]
    values = np.asarray([logits["supported"], logits["refuted"], logits["undetermined"]])
    probs = np.exp(values - values.max()); probs /= probs.sum()
    margin = float(logits["supported"] - logits["refuted"])
    return {
        "logits": logits,
        "yes_minus_no": margin,
        "prediction": int(margin > 0),
        "probabilities": {"yes": float(probs[0]), "no": float(probs[1]), "maybe": float(probs[2])},
    }


@torch.inference_mode()
def score_topology(
    bot: Any,
    ids: dict[str, int],
    embeddings: torch.Tensor,
    positions: torch.Tensor | None,
    visual_span: tuple[int, int],
    mode: str,
) -> dict[str, Any]:
    mask = topology_mask(embeddings, visual_span, mode)
    return score(bot, ids, embeddings, mask, positions)


def metrics(rows: list[dict[str, Any]], order: str) -> dict[str, Any]:
    selected = [row for row in rows if row["status"] == "ok" and row["order"] == order]
    truth = np.asarray([int(row["label"]) for row in selected])
    pred = np.asarray([int(row["score"]["prediction"]) for row in selected])
    tp = int(np.sum((truth == 1) & (pred == 1))); tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1))); fn = int(np.sum((truth == 1) & (pred == 0)))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {"n": len(selected), "accuracy": (tp + tn) / len(selected), "balanced_accuracy": (tpr + tnr) / 2, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def bootstrap_delta(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, float]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["status"] == "ok": grouped[row["record_key"]][row["order"]] = row
    pairs = [cell for cell in grouped.values() if set(cell) == set(ORDERS)]
    y = np.asarray([int(cell["image_first"]["label"]) for cell in pairs])
    native = np.asarray([int(cell["image_first"]["score"]["prediction"]) for cell in pairs])
    query = np.asarray([int(cell["query_first"]["score"]["prediction"]) for cell in pairs])
    def bacc(indices: np.ndarray, pred: np.ndarray) -> float:
        yy=y[indices]; pp=pred[indices]
        return float((np.mean(pp[yy == 1] == 1) + np.mean(pp[yy == 0] == 0)) / 2)
    rng=np.random.default_rng(seed); values=[]; all_idx=np.arange(len(y))
    # Stratified image bootstrap preserves the frozen balanced design.
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    for _ in range(draws):
        idx=np.concatenate((rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)))
        values.append(bacc(idx,query)-bacc(idx,native))
    point=bacc(all_idx,query)-bacc(all_idx,native)
    return {"estimate": point, "ci_low": float(np.quantile(values,.025)), "ci_high": float(np.quantile(values,.975))}


def analyze(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    cells={order:metrics(rows,order) for order in ORDERS}
    delta=bootstrap_delta(rows,draws,seed)
    passed=bool(delta["estimate"]>=.02 and delta["ci_low"]>0 and cells["query_first"]["fp"]<=cells["image_first"]["fp"] and cells["query_first"]["fn"]<=cells["image_first"]["fn"])
    return {
        "version":VERSION,
        "status":"GO_QUERY_FIRST_TOPOLOGY" if passed else "NO_GO_QUERY_FIRST_TOPOLOGY",
        "orders":cells,
        "query_first_minus_image_first_bacc":delta,
        "gate_passed":passed,
        "gate":"BAcc +2pp with CI low>0; neither FP nor FN may increase",
        "structural_fact":"under image-first causal serialization, visual-token hidden states have exactly zero derivative with respect to later question tokens",
        "boundary":"a pass is only a topology phenomenon; cross-model OE correction and collision clearance remain required",
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,default=Path("corrected_runs/c3_guard/vindr_claim_common_mode_canary_v1/manifest.jsonl"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--per-bin",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20260813)
    parser.add_argument("--bootstrap-draws",type=int,default=5000)
    parser.add_argument("--huatuo-root",type=Path,default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir",type=Path,default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    cases=load_cases(args.manifest,args.per_bin,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config={"version":VERSION,"created_at":now(),"model":"HuatuoGPT-Vision-7B","seed":args.seed,"per_bin":args.per_bin,"cases":cases,"intervention":"move one image placeholder after the unchanged question inside the same user turn","orders":list(ORDERS),"research_role":"fatal topology canary"}
    config_path=args.output_dir/"config.json"
    if config_path.exists():
        if not args.resume: raise FileExistsError("output exists; use --resume")
        old=json.loads(config_path.read_text())
        for key in ("version","model","seed","per_bin","cases","intervention","orders"):
            if old[key]!=config[key]: raise RuntimeError(f"resume config drift: {key}")
    else: atomic_json(config_path,config)
    raw_path=args.output_dir/"raw.jsonl"; completed=set()
    if raw_path.exists() and args.resume:
        completed={(json.loads(line)["record_key"],json.loads(line)["order"]) for line in raw_path.read_text().splitlines() if line.strip()}
    HuatuoChatbot=import_huatuo(args.huatuo_root); bot=HuatuoChatbot(str(args.model_dir),device="cuda:0"); bot.debug=False; ids=label_ids(bot)
    total=len(cases)*2
    for case in cases:
        image=dicom_to_pil(Path(case["image_path"])); tensor=torch.stack(bot.get_image_tensors([image])).to(device=bot.model.device,dtype=torch.bfloat16)
        for order in ORDERS:
            key=(case["record_key"],order)
            if key in completed: continue
            row={"version":VERSION,"record_key":case["record_key"],"image_id":case["image_id"],"image_path":case["image_path"],"finding":case["finding"],"positive_votes":case["positive_votes"],"label":int(case["positive_votes"]==3),"order":order,"status":"error"}
            try:
                builder=prepared_embeddings if order=="image_first" else query_first_embeddings
                emb,att,pos,span=builder(bot,prompt_for(case["finding"]),tensor)
                row.update({"status":"ok","visual_span":list(span),"sequence_tokens":int(emb.shape[1]),"score":score(bot,ids,emb,att,pos),"completed_at":now()})
            except Exception as error: row.update({"error":repr(error),"traceback":traceback.format_exc(),"completed_at":now()})
            append_jsonl(raw_path,row); completed.add(key); print(f"[{len(completed)}/{total}] {case['record_key']} {order} {row['status']} pred={row.get('score',{}).get('prediction')}",flush=True)
    rows=[json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    atomic_json(args.output_dir/"analysis.json",analyze(rows,args.bootstrap_draws,args.seed))


if __name__=="__main__": main()
