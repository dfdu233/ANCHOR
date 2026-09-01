"""Command-line entry points for config validation and legacy triage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .legacy import audit_legacy_answers
from .queue import JobQueue
from .task import load_records, load_task_spec, materialize_samples


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m anchor.medeval.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-task")
    validate.add_argument("config", type=Path)
    audit = commands.add_parser("audit-legacy")
    audit.add_argument("--answers", type=Path, required=True)
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--id-field", default="question_id")
    audit.add_argument("--output", type=Path)
    status = commands.add_parser("queue-status")
    status.add_argument("database", type=Path)
    args = parser.parse_args()

    if args.command == "validate-task":
        spec = load_task_spec(args.config)
        samples = materialize_samples(spec)
        result = {"valid": True, "task": spec.name, "samples": len(samples)}
    elif args.command == "audit-legacy":
        records = load_records(args.dataset)
        expected = [str(row.get(args.id_field, index)) for index, row in enumerate(records)]
        result = audit_legacy_answers(args.answers, expected)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = {"jobs": JobQueue(args.database).status()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
