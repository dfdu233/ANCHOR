from __future__ import annotations

import json
from pathlib import Path

from .registry import REPO_ROOT, dataset_config, method_config


def _require(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path.relative_to(REPO_ROOT)}")


def _check_jsonl(path: Path, errors: list[str]) -> int:
    if not path.exists():
        errors.append(f"missing jsonl: {path.relative_to(REPO_ROOT)}")
        return 0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def _check_manifest(path: Path, errors: list[str], sample: int = 4) -> int:
    if not path.exists():
        errors.append(f"missing manifest: {path.relative_to(REPO_ROOT)}")
        return 0
    count = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if count <= sample:
                storage = row.get("storage_path") or row.get("relative_path")
                if storage:
                    candidate = (path.parent / str(storage)).resolve()
                    if not candidate.exists():
                        # Manifests may intentionally point outside their folder.
                        alt = (REPO_ROOT / "data/medheval/images" / str(row.get("relative_path", ""))).resolve()
                        if not alt.exists():
                            errors.append(f"manifest target missing: {storage}")
    return count


def main() -> None:
    errors: list[str] = []
    datasets = dataset_config()
    methods = method_config()
    if not datasets.get("default"):
        errors.append("configs/datasets.yaml has no default datasets")
    if not methods.get("methods"):
        errors.append("configs/methods.yaml has no methods")

    for name in datasets.get("default", []):
        cfg = datasets["datasets"][name]
        root = REPO_ROOT / cfg["data_root"]
        _require(root, errors)
        if "questions" in cfg:
            count = _check_jsonl(REPO_ROOT / cfg["questions"], errors)
            if count == 0:
                errors.append(f"{name}: empty questions")
        if "manifest" in cfg:
            _require(REPO_ROOT / cfg["manifest"], errors)
        if "image_manifest" in cfg:
            count = _check_manifest(REPO_ROOT / cfg["image_manifest"], errors)
            if count == 0:
                errors.append(f"{name}: empty image manifest")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    print("ANCHOR smoke passed: configs and default dataset paths are valid.")


if __name__ == "__main__":
    main()
