#!/usr/bin/env python3
"""Auditable prediction-side RadGraph to clinical-claim conversion.

RadGraph proposes entities and relations in generated text.  It never supplies
reader support or reference truth.  Unknown and ambiguous ontology mappings are
returned in the audit trail instead of being guessed into a known finding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from corrected_sgta.clinical_claims import ClinicalClaim, normalize_term


VERSION = "missing-third-state-radgraph-claims-v2"
OBSERVATION_PREFIX = "Observation::"
ANATOMY_PREFIX = "Anatomy::"


def _words(value: object) -> tuple[str, ...]:
    text = str(value).strip().lower().replace("/", " ").replace("-", " ")
    return tuple(part for part in text.split() if part)


def load_ontology_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    """Load a frozen canonical-to-alias mapping from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "findings" in payload:
        payload = payload["findings"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("ontology must be a non-empty object or contain 'findings'")
    aliases: dict[str, tuple[str, ...]] = {}
    for canonical, raw_aliases in payload.items():
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        if not isinstance(raw_aliases, Sequence):
            raise ValueError(f"aliases for {canonical!r} must be a string or list")
        canonical_name = normalize_term(str(canonical))
        values = [str(canonical).replace("_", " "), *(str(x) for x in raw_aliases)]
        aliases[canonical_name] = tuple(dict.fromkeys(x.strip().lower() for x in values if x))
    return aliases


def _ontology_matches(
    phrase_words: Sequence[str],
    ontology_aliases: Mapping[str, Iterable[str]],
) -> list[tuple[int, str, tuple[str, ...], int]]:
    matches: list[tuple[int, str, tuple[str, ...], int]] = []
    phrase = tuple(phrase_words)
    for canonical, aliases in ontology_aliases.items():
        for alias in aliases:
            alias_words = _words(alias)
            if not alias_words or len(alias_words) > len(phrase):
                continue
            for start in range(len(phrase) - len(alias_words) + 1):
                if phrase[start : start + len(alias_words)] == alias_words:
                    matches.append((len(alias_words), normalize_term(canonical), alias_words, start))
    return sorted(matches, key=lambda item: (-item[0], item[1], item[3]))


def _relation_map(
    entities: Mapping[str, Mapping[str, object]],
    source_prefix: str,
    relation_name: str,
) -> dict[str, list[str]]:
    incoming: dict[str, list[str]] = defaultdict(list)
    for entity_id, entity in entities.items():
        if not str(entity.get("label", "")).startswith(source_prefix):
            continue
        for relation in entity.get("relations", []):
            if len(relation) == 2 and str(relation[0]) == relation_name:
                incoming[str(relation[1])].append(str(entity_id))
    return incoming


def _recursive_incoming(
    root: str,
    incoming: Mapping[str, Sequence[str]],
    visited: set[str] | None = None,
) -> tuple[list[str], bool]:
    visited = set() if visited is None else visited
    if root in visited:
        return [], True
    visited.add(root)
    result: list[str] = []
    cycle = False
    for source in incoming.get(root, []):
        nested, nested_cycle = _recursive_incoming(source, incoming, visited)
        result.extend(nested)
        cycle = cycle or nested_cycle
    result.append(root)
    return result, cycle


def _ordered_unique_entity_ids(
    entity_ids: Iterable[str], entities: Mapping[str, Mapping[str, object]]
) -> list[str]:
    unique = list(dict.fromkeys(entity_ids))
    return sorted(unique, key=lambda key: (int(entities[key].get("start_ix", 0)), key))


def _state_from_labels(labels: Sequence[str]) -> tuple[str, str, bool]:
    absent = any("definitely absent" in label for label in labels)
    uncertain = any("uncertain" in label for label in labels)
    present = any("definitely present" in label for label in labels)
    conflict = absent and present
    if uncertain or conflict:
        return "present", "uncertain", conflict
    if absent:
        return "absent", "definite", False
    return "present", "definite", False


def claims_from_radgraph(
    annotation: Mapping[str, object],
    ontology_aliases: Mapping[str, Iterable[str]],
) -> tuple[list[ClinicalClaim], dict[str, object]]:
    """Convert one RadGraph annotation into claims plus a lossless audit.

    ``annotation`` is one item returned by ``RadGraph`` (it contains ``text``
    and ``entities``), not the outer batch dictionary.
    """

    raw_entities = annotation.get("entities")
    if not isinstance(raw_entities, Mapping):
        raise ValueError("RadGraph annotation is missing an entities mapping")
    entities = {str(key): value for key, value in raw_entities.items()}
    if not all(isinstance(value, Mapping) for value in entities.values()):
        raise ValueError("every RadGraph entity must be an object")

    observation_ids = {
        key
        for key, entity in entities.items()
        if str(entity.get("label", "")).startswith(OBSERVATION_PREFIX)
    }
    observation_modifiers = _relation_map(entities, OBSERVATION_PREFIX, "modify")
    anatomy_modifiers = _relation_map(entities, ANATOMY_PREFIX, "modify")
    outgoing_modify = {
        source
        for source in observation_ids
        if any(
            len(relation) == 2 and str(relation[0]) == "modify"
            for relation in entities[source].get("relations", [])
        )
    }
    roots = sorted(observation_ids - outgoing_modify)
    suggested_targets: dict[str, list[str]] = defaultdict(list)
    for source in observation_ids:
        for relation in entities[source].get("relations", []):
            if len(relation) == 2 and str(relation[0]) == "suggestive_of":
                suggested_targets[str(relation[1])].append(source)

    claims: list[ClinicalClaim] = []
    records: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    for root in roots:
        component_ids, cycle = _recursive_incoming(root, observation_modifiers)
        component_ids = _ordered_unique_entity_ids(component_ids, entities)
        phrase_words: list[str] = []
        for entity_id in component_ids:
            phrase_words.extend(_words(entities[entity_id].get("tokens", "")))
        phrase = " ".join(phrase_words)
        matches = _ontology_matches(phrase_words, ontology_aliases)
        if not matches:
            unmatched.append(
                {"root_entity_id": root, "phrase": phrase, "reason": "no_ontology_match"}
            )
            continue
        best_length = matches[0][0]
        best_canonicals = sorted({item[1] for item in matches if item[0] == best_length})
        if len(best_canonicals) != 1:
            unmatched.append(
                {
                    "root_entity_id": root,
                    "phrase": phrase,
                    "reason": "ambiguous_ontology_match",
                    "candidates": best_canonicals,
                }
            )
            continue
        _, canonical, matched_words, match_start = next(
            item for item in matches if item[0] == best_length and item[1] == best_canonicals[0]
        )
        remaining_words = (
            phrase_words[:match_start]
            + phrase_words[match_start + len(matched_words) :]
        )
        attributes = tuple([" ".join(remaining_words)]) if remaining_words else ()

        anatomy_ids: list[str] = []
        for observation_id in component_ids:
            for relation in entities[observation_id].get("relations", []):
                if len(relation) == 2 and str(relation[0]) == "located_at":
                    anatomy_ids.append(str(relation[1]))
        anatomy_phrases: list[str] = []
        for anatomy_id in anatomy_ids:
            if anatomy_id not in entities:
                continue
            anatomy_component, anatomy_cycle = _recursive_incoming(
                anatomy_id, anatomy_modifiers
            )
            cycle = cycle or anatomy_cycle
            anatomy_component = _ordered_unique_entity_ids(anatomy_component, entities)
            anatomy_phrases.append(
                " ".join(str(entities[key].get("tokens", "")) for key in anatomy_component)
            )
        anatomy_phrases = list(dict.fromkeys(normalize_term(x) for x in anatomy_phrases if x))
        anatomy = "+".join(anatomy_phrases) if anatomy_phrases else None

        labels = [str(entities[key].get("label", "")) for key in component_ids]
        polarity, uncertainty, polarity_conflict = _state_from_labels(labels)
        incoming_suggestions = suggested_targets.get(root, [])
        provenance = "knowledge" if incoming_suggestions else "image_grounded"
        claim = ClinicalClaim(
            finding=canonical,
            polarity=polarity,
            uncertainty=uncertainty,
            anatomy=anatomy,
            attributes=attributes,
            provenance=provenance,
        )
        claims.append(claim)
        records.append(
            {
                "root_entity_id": root,
                "component_entity_ids": component_ids,
                "phrase": phrase,
                "matched_alias": " ".join(matched_words),
                "claim": claim.to_dict(),
                "polarity_conflict": polarity_conflict,
                "incoming_suggestive_of": incoming_suggestions,
                "relation_cycle_detected": cycle,
            }
        )

    duplicates: dict[tuple[str, str | None, tuple[str, ...]], list[int]] = defaultdict(list)
    for index, claim in enumerate(claims):
        duplicates[claim.key].append(index)
    duplicate_records = [
        {
            "claim_key": [key[0], key[1], list(key[2])],
            "indices": indices,
            "states": [claims[index].state for index in indices],
            "state_conflict": len({claims[index].state for index in indices}) > 1,
        }
        for key, indices in duplicates.items()
        if len(indices) > 1
    ]
    linked_anatomy_ids = {
        str(relation[1])
        for entity in entities.values()
        if str(entity.get("label", "")).startswith(OBSERVATION_PREFIX)
        for relation in entity.get("relations", [])
        if len(relation) == 2 and str(relation[0]) == "located_at"
    }
    orphan_anatomy = [
        {
            "entity_id": entity_id,
            "tokens": str(entity.get("tokens", "")),
            "start_ix": int(entity.get("start_ix", 0)),
            "reason": "anatomy_not_linked_to_observation",
        }
        for entity_id, entity in entities.items()
        if str(entity.get("label", "")).startswith(ANATOMY_PREFIX)
        and entity_id not in linked_anatomy_ids
    ]
    audit = {
        "version": VERSION,
        "text": str(annotation.get("text", "")),
        "n_entities": len(entities),
        "n_observation_roots": len(roots),
        "n_claims": len(claims),
        "records": records,
        "unmatched_observations": unmatched,
        "orphan_anatomy_entities": orphan_anatomy,
        "duplicate_claims": duplicate_records,
        "radgraph_entities": entities,
        "warning": (
            "Prediction-side structure only: reader/expert references remain the sole truth. "
            "RadGraph alone does not resolve history, temporal comparison, or observability."
        ),
    }
    return claims, audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id/report fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="report")
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-cache-dir", type=Path, required=True)
    parser.add_argument("--cuda", type=int, default=-1)
    args = parser.parse_args()

    from radgraph import RadGraph

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [str(row[args.text_field]) for row in rows]
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    args.tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
    extractor = RadGraph(
        batch_size=1,
        cuda=args.cuda,
        model_type=args.model_type,
        model_cache_dir=str(args.model_cache_dir),
        tokenizer_cache_dir=str(args.tokenizer_cache_dir),
    )
    annotations = extractor(texts)
    aliases = load_ontology_aliases(args.ontology)
    output_rows = []
    for index, row in enumerate(rows):
        claims, audit = claims_from_radgraph(annotations[str(index)], aliases)
        output_rows.append(
            {
                "id": row[args.id_field],
                "report": row[args.text_field],
                "claims": [claim.to_dict() for claim in claims],
                "audit": audit,
            }
        )
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "model_type": args.model_type,
        "model_weights_sha256": sha256_file(
            args.model_cache_dir / args.model_type / "weights.th"
        ),
        "radgraph_package_version": importlib.metadata.version("radgraph"),
        "transformers_package_version": importlib.metadata.version("transformers"),
        "code_sha256": sha256_file(Path(__file__)),
        "cuda": args.cuda,
        "n_reports": len(rows),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    payload = {"config": config, "reports": output_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
