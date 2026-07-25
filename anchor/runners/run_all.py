from __future__ import annotations

import json

from .registry import (
    build_arg_parser,
    dataset_config,
    default_datasets,
    ensure_run_dir,
    expand_methods,
    method_config,
    write_run_metadata,
)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    datasets_cfg = dataset_config()
    methods_cfg = method_config()
    datasets = args.datasets.split(",") if args.datasets else default_datasets()
    methods = expand_methods(args.methods.split(","))

    skipped = []
    planned = []
    for dataset in datasets:
        dcfg = datasets_cfg["datasets"].get(dataset)
        if not dcfg:
            skipped.append({"dataset": dataset, "reason": "unknown dataset"})
            continue
        task = dcfg["task"]
        for method in methods:
            mcfg = methods_cfg["methods"].get(method)
            if not mcfg:
                skipped.append({"dataset": dataset, "method": method, "reason": "unknown method"})
                continue
            if task not in mcfg["tasks"] and not (
                task == "vqa_and_report" and {"vqa_binary", "report_generation"} & set(mcfg["tasks"])
            ):
                skipped.append({"dataset": dataset, "method": method, "reason": f"unsupported task {task}"})
                continue
            run_dir = ensure_run_dir(dataset, args.model, method)
            payload = {
                "dataset": dataset,
                "dataset_config": dcfg,
                "method": method,
                "method_config": mcfg,
                "model": args.model,
                "max_samples": args.max_samples,
                "execute": bool(args.execute),
            }
            write_run_metadata(run_dir, payload)
            summary = {
                "status": "planned" if not args.execute else "dispatch_pending",
                "dataset": dataset,
                "method": method,
                "task": task,
                "run_dir": str(run_dir),
                "note": "Use maintained corrected_sgta modules for full GPU inference.",
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            planned.append(summary)

    print(json.dumps({"planned": planned, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
