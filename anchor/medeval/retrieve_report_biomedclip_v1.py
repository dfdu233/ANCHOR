#!/usr/bin/env python3
"""Retrieve train-only reports with BioMedCLIP image similarity.

This is deliberately an image-to-image retriever.  The query report/reference is
never encoded, and same-patient/same-study candidates are excluded before top-k.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.nn import functional as F

from .hashing import sha256_file

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "shared-report-rag-biomedclip-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--dataset", choices=("iuxray", "mimic"), required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--corpus-dataset", choices=("iuxray", "mimic"))
    p.add_argument("--corpus-image-root", type=Path)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load_json(path: Path) -> Any:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_biomedclip(root: Path):
    from open_clip import create_model_and_transforms
    from open_clip.factory import _MODEL_CONFIGS

    config = json.loads((root / "open_clip_config.json").read_text())
    text_encoder = root / "text_encoder"
    if not (text_encoder / "config.json").is_file():
        raise FileNotFoundError(text_encoder / "config.json")
    # OpenCLIP only needs the architecture config here: the complete text-tower
    # weights are already inside the BioMedCLIP checkpoint. Binding this to a
    # local path prevents an otherwise hidden Hub lookup during image-only
    # retrieval.
    config["model_cfg"]["text_cfg"]["hf_model_name"] = str(text_encoder.resolve())
    config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(text_encoder.resolve())
    _MODEL_CONFIGS["anchor_biomedclip_local"] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        "anchor_biomedclip_local",
        pretrained=str(root / "open_clip_pytorch_model.bin"),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    return model, preprocess


def ids_for(dataset: str, row: dict[str, Any], *, query: bool) -> tuple[str | None, str | None]:
    if dataset == "iuxray":
        study = str(row.get("qid") or row.get("study_id") or row.get("id") or row.get("doc_id"))
        return None, study.removeprefix("iuxray:")
    if query:
        rel = str(row["img_name"])
        parts = Path(rel).parts
        patient = next((x[1:] for x in parts if x.startswith("p") and x[1:].isdigit()), None)
        study = next((x[1:] for x in parts if x.startswith("s") and x[1:].isdigit()), None)
        return patient, study
    patient = row.get("patient_id")
    study = row.get("study_id")
    return (str(patient) if patient is not None else None, str(study) if study is not None else None)


def image_path(dataset: str, root: Path, row: dict[str, Any], *, query: bool) -> Path:
    if query:
        return root / str(row["img_name"])
    values = row.get("image_paths") or [row.get("image_id")]
    rel = str(values[0])
    return root / rel


@torch.inference_mode()
def encode_images(model, preprocess, paths: list[Path], batch_size: int, device: torch.device) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start : start + batch_size]:
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as handle:
                tensors.append(preprocess(handle.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        features.append(F.normalize(model.encode_image(batch), dim=-1).float().cpu())
        print(f"encoded {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(features)


def main() -> None:
    args = parse_args()
    corpus_dataset = args.corpus_dataset or args.dataset
    corpus_image_root = args.corpus_image_root or args.image_root
    queries = load_json(args.queries)
    corpus_all = load_json(args.corpus)
    corpus = [row for row in corpus_all if row.get("dataset") == corpus_dataset]
    if not queries or not corpus:
        raise RuntimeError("query or dataset-specific corpus is empty")

    weights = args.model_root / "open_clip_pytorch_model.bin"
    text_config = args.model_root / "text_encoder/config.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fingerprint_payload = {
        "version": VERSION,
        "queries_sha256": sha256_file(args.queries),
        "corpus_sha256": sha256_file(args.corpus),
        "weights_sha256": sha256_file(weights),
        "text_config_sha256": sha256_file(text_config),
        "dataset": args.dataset,
        "corpus_dataset": corpus_dataset,
        "corpus_image_root": str(corpus_image_root.resolve()),
        "top_k": args.top_k,
        "query_signal": "image_only",
        "exclusions": "same_patient_or_same_study",
        "embedding_device": device.type,
    }
    fingerprint = stable_hash(fingerprint_payload)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / f"{args.dataset}.{fingerprint}.pt"
    corpus_paths = [image_path(corpus_dataset, corpus_image_root, row, query=False) for row in corpus]
    query_paths = [image_path(args.dataset, args.image_root, row, query=True) for row in queries]

    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        corpus_features = payload["corpus_features"]
        query_features = payload["query_features"]
        print(f"reused {cache}", flush=True)
    else:
        model, preprocess = load_biomedclip(args.model_root)
        model = model.to(device).eval()
        try:
            corpus_features = encode_images(model, preprocess, corpus_paths, args.batch_size, device)
            query_features = encode_images(model, preprocess, query_paths, args.batch_size, device)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        torch.save({"corpus_features": corpus_features.half(), "query_features": query_features.half()}, cache)

    similarities = query_features.float() @ corpus_features.float().T
    records = []
    for index, query in enumerate(queries):
        query_patient, query_study = ids_for(args.dataset, query, query=True)
        allowed = []
        for candidate_index, candidate in enumerate(corpus):
            patient, study = ids_for(corpus_dataset, candidate, query=False)
            same_domain = args.dataset == corpus_dataset
            same_patient = same_domain and query_patient is not None and patient == query_patient
            same_study = same_domain and query_study is not None and study == query_study
            if not same_patient and not same_study:
                allowed.append(candidate_index)
        if len(allowed) < args.top_k:
            raise RuntimeError(f"insufficient non-overlapping candidates for query {index}")
        scores = similarities[index, allowed]
        offsets = torch.topk(scores, args.top_k).indices.tolist()
        docs = []
        for rank, offset in enumerate(offsets, 1):
            candidate_index = allowed[offset]
            candidate = corpus[candidate_index]
            docs.append({
                "rank": rank,
                "doc_id": candidate["doc_id"],
                "report": candidate["report"],
                "similarity": float(similarities[index, candidate_index]),
                "patient_id": candidate.get("patient_id"),
                "study_id": candidate.get("study_id"),
                "image_id": candidate.get("image_id"),
            })
        qid = str(query.get("qid", query.get("id", index)))
        records.append({"sample_id": qid, "documents": docs})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    manifest = {
        "protocol": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "rows": len(records),
        "corpus_rows": len(corpus),
        "query_dataset": args.dataset,
        "corpus_dataset": corpus_dataset,
        "cross_dataset_corpus": args.dataset != corpus_dataset,
        "top_k": args.top_k,
        "reference_used_for_retrieval": False,
        "query_signal": "image_only",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "feature_cache": str(cache.resolve()),
    }
    (args.output.parent / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
