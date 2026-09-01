#!/usr/bin/env python3
"""Extract prediction-side structured claims without a reference-derived ontology."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from anchor.corrected_sgta.clinical_claims import ClinicalClaim, normalize_term
from anchor.corrected_sgta.radgraph_claims import (
    ANATOMY_PREFIX,
    OBSERVATION_PREFIX,
    _ordered_unique_entity_ids,
    _recursive_incoming,
    _relation_map,
    _state_from_labels,
)
from anchor.medeval.hashing import sha256_file


VERSION = "radgraph-surface-clinical-claims-v1"


def claims_from_surface_graph(annotation: Mapping[str, object]) -> tuple[list[ClinicalClaim], dict[str, Any]]:
    raw_entities = annotation.get("entities")
    if not isinstance(raw_entities, Mapping):
        raise ValueError("RadGraph annotation is missing entities")
    entities = {str(key): value for key, value in raw_entities.items()}
    if not all(isinstance(value, Mapping) for value in entities.values()):
        raise ValueError("every RadGraph entity must be an object")

    observations = {
        key for key, entity in entities.items()
        if str(entity.get("label", "")).startswith(OBSERVATION_PREFIX)
    }
    observation_modifiers = _relation_map(entities, OBSERVATION_PREFIX, "modify")
    anatomy_modifiers = _relation_map(entities, ANATOMY_PREFIX, "modify")
    outgoing_modifiers = {
        source for source in observations
        if any(
            len(relation) == 2 and str(relation[0]) == "modify"
            for relation in entities[source].get("relations", [])
        )
    }
    roots = sorted(observations - outgoing_modifiers)
    suggested_targets: dict[str, list[str]] = {}
    for source in observations:
        for relation in entities[source].get("relations", []):
            if len(relation) == 2 and str(relation[0]) == "suggestive_of":
                suggested_targets.setdefault(str(relation[1]), []).append(source)

    claims: list[ClinicalClaim] = []
    records: list[dict[str, Any]] = []
    for root in roots:
        component, cycle = _recursive_incoming(root, observation_modifiers)
        component = _ordered_unique_entity_ids(component, entities)
        finding = normalize_term(str(entities[root].get("tokens", "")))
        if not finding:
            continue
        attributes = tuple(
            normalize_term(str(entities[key].get("tokens", "")))
            for key in component
            if key != root and str(entities[key].get("tokens", "")).strip()
        )
        anatomy_ids: list[str] = []
        for observation_id in component:
            for relation in entities[observation_id].get("relations", []):
                if len(relation) == 2 and str(relation[0]) == "located_at":
                    anatomy_ids.append(str(relation[1]))
        anatomy_parts: list[str] = []
        for anatomy_id in anatomy_ids:
            if anatomy_id not in entities:
                continue
            anatomy_component, anatomy_cycle = _recursive_incoming(anatomy_id, anatomy_modifiers)
            cycle = cycle or anatomy_cycle
            anatomy_component = _ordered_unique_entity_ids(anatomy_component, entities)
            phrase = " ".join(str(entities[key].get("tokens", "")) for key in anatomy_component)
            if phrase.strip():
                anatomy_parts.append(normalize_term(phrase))
        anatomy_parts = list(dict.fromkeys(anatomy_parts))
        labels = [str(entities[key].get("label", "")) for key in component]
        polarity, uncertainty, polarity_conflict = _state_from_labels(labels)
        incoming_suggestions = suggested_targets.get(root, [])
        claim = ClinicalClaim(
            finding=finding,
            polarity=polarity,
            uncertainty=uncertainty,
            anatomy="+".join(anatomy_parts) if anatomy_parts else None,
            attributes=attributes,
            provenance="knowledge" if incoming_suggestions else "image_grounded",
        )
        claims.append(claim)
        records.append(
            {
                "root_entity_id": root,
                "component_entity_ids": component,
                "claim": claim.to_dict(),
                "polarity_conflict": polarity_conflict,
                "incoming_suggestive_of": incoming_suggestions,
                "relation_cycle_detected": cycle,
            }
        )
    audit = {
        "version": VERSION,
        "text": str(annotation.get("text", "")),
        "n_entities": len(entities),
        "n_observation_roots": len(roots),
        "n_claims": len(claims),
        "records": records,
        "radgraph_entities": entities,
        "unparsed_as_no_structured_claim": len(claims) == 0,
        "warning": "prediction-side structure only; this extractor never supplies clinical truth",
    }
    return claims, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--model-cache-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-cache-dir", required=True, type=Path)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    from radgraph import RadGraph

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    extractor = RadGraph(
        batch_size=1,
        cuda=args.cuda,
        model_type=args.model_type,
        model_cache_dir=str(args.model_cache_dir),
        tokenizer_cache_dir=str(args.tokenizer_cache_dir),
    )
    annotations = extractor([str(row["report"]) for row in rows])
    reports = []
    for index, row in enumerate(rows):
        claims, audit = claims_from_surface_graph(annotations[str(index)])
        reports.append(
            {
                "id": row["id"],
                "report": row["report"],
                "source": row.get("source", {}),
                "claims": [claim.to_dict() for claim in claims],
                "audit": audit,
            }
        )
    weights = args.model_cache_dir / args.model_type / "weights.th"
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "model_type": args.model_type,
        "model_weights_sha256": sha256_file(weights),
        "radgraph_package_version": importlib.metadata.version("radgraph"),
        "code_sha256": sha256_file(Path(__file__)),
        "cuda": args.cuda,
        "n_reports": len(reports),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    payload = {"config": config, "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
