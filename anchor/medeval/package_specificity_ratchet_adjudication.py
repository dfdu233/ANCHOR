#!/usr/bin/env python3
"""Package a deterministic, provenance-blinded adjudication archive.

The input CSV must already contain two independently returned reviewer sheets
copied by ``merge_specificity_ratchet_reviews_v1``.  This module never fills a
final clinical field.  It only joins reviewer-visible candidates and images to
an offline form whose mutable surface is restricted to adjudication fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import REVIEW_FIELDS
from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    FINAL_FIELDS,
    PROTOCOL_ID,
)
from anchor.medeval.package_physician_oe_deliveries import (
    _write_archive,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


VERSION = "specificity-ratchet-adjudication-delivery-v1"
MUTABLE_FIELDS = tuple(
    [f"final_{field}" for field in FINAL_FIELDS]
    + ["adjudicator_id", "disagreement_reason", "adjudication_rationale"]
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instructions(bundle_id: str) -> bytes:
    return f"""# Blinded physician adjudication

Bundle: `{bundle_id}`

1. Work only from this extracted directory. Model identity, automatic scores,
   benchmark answers, and private provenance are intentionally absent.
2. Open `ADJUDICATION_FORM.html`. Inspect the supplied image, question, answer
   span, proposal edge, and both independently frozen reviews.
3. Enter one stable adjudicator ID and complete every `final_*` field plus an
   adjudication rationale. A disagreement reason is mandatory whenever either
   reviewer's categorical fields differ.
4. Do not change copied reviewer fields, row order, case IDs, or edge IDs. Do
   not begin free text with `=`, `+`, `-`, or `@`.
5. Check the physician and provenance-blinding attestations only if true, then
   export both `adjudication.completed.csv` and
   `adjudicator.attestation.json`.
6. Return only those two files. The coordinator may combine separately signed
   attestations but may not sign or infer a clinical value for you.

Browser validation is ergonomic only. The repository's fail-closed Python
validator remains authoritative.
""".encode("utf-8")


def _form(
    *,
    header: list[str],
    rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    schema: dict[str, Any],
    bundle_id: str,
) -> bytes:
    candidate_by_edge = {str(row["edge_id"]): row for row in candidates}
    payload = {
        "header": header,
        "rows": rows,
        "candidates": candidate_by_edge,
        "schema": schema,
        "review_fields": list(REVIEW_FIELDS),
        "final_fields": list(FINAL_FIELDS),
        "mutable_fields": list(MUTABLE_FIELDS),
    }
    embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Specificity Ratchet blinded adjudication</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#17202a}header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #c8d0d8;padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.wrap{max-width:1250px;margin:auto;padding:16px}.panel,.edge{background:#fff;border:1px solid #c8d0d8;border-radius:8px;padding:14px;margin:12px 0}.edge{border-left:5px solid #8c5ac7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:10px}.reviews{display:grid;grid-template-columns:1fr 1fr;gap:10px}.review{background:#edf1f4;border-radius:6px;padding:9px;white-space:pre-wrap}label{display:flex;flex-direction:column;gap:4px;font-size:13px;font-weight:650}select,input,textarea,button{font:inherit}select,input,textarea{padding:7px;border:1px solid #aab5c0;border-radius:5px;background:#fff}textarea{min-height:70px}button{padding:7px 11px;border:1px solid #68409b;border-radius:5px;background:#68409b;color:white;cursor:pointer}button.secondary{background:white;color:#273142}.image{display:block;max-width:100%;max-height:650px;margin:auto;background:#111}.fixed{white-space:pre-wrap;background:#edf1f4;border-radius:6px;padding:9px}.muted{color:#65727e;font-size:13px}.pill{background:#e7ebef;border-radius:18px;padding:4px 9px}.error{color:#a40000;font-weight:700}.ok{color:#176b32;font-weight:700}.wide{grid-column:1/-1}@media(max-width:760px){.reviews{grid-template-columns:1fr}}
</style></head><body><header><strong>Specificity Ratchet — blinded adjudicator</strong><span id="position" class="pill"></span><span id="progress" class="pill"></span><label>Stable adjudicator ID<input id="adjudicator-id" autocomplete="off"></label><button id="prev" class="secondary">Previous image</button><button id="next" class="secondary">Next image</button><button id="validate" class="secondary">Validate all</button><button id="export">Export completed CSV</button><button id="export-attestation">Export attestation JSON</button><label>Import CSV<input id="import" type="file" accept=".csv"></label><label><input id="attest-physician" type="checkbox">I am a physician</label><label><input id="attest-blinded" type="checkbox">I remained blinded to private provenance</label><span id="message"></span></header><main class="wrap"><section class="panel"><p class="muted">Bundle __BUNDLE__. Reviewer fields are read-only; private provenance and model identity are absent.</p><h2 id="question"></h2><img id="image" class="image"><p><strong>Observed answer span</strong></p><div id="answer" class="fixed"></div></section><section id="edges"></section></main><script id="seed" type="application/json">__DATA__</script><script>
"use strict";
const pack=JSON.parse(document.getElementById('seed').textContent),seed=pack.rows,header=pack.header,schema=pack.schema,reviewFields=pack.review_fields,finalFields=pack.final_fields,mutableFields=new Set(pack.mutable_fields),bundle='__BUNDLE__',storageKey=`specificity-ratchet-adjudication-v1:${bundle}`;let rows=JSON.parse(JSON.stringify(seed)),groupIndex=0;try{const saved=localStorage.getItem(storageKey);if(saved)rows=JSON.parse(saved)}catch(e){}
const $=id=>document.getElementById(id),esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function groups(){const out=[],by=new Map();rows.forEach((row,i)=>{if(!by.has(row.case_id)){const g={case_id:row.case_id,indices:[]};by.set(row.case_id,g);out.push(g)}by.get(row.case_id).indices.push(i)});return out}
function disagrees(row){return reviewFields.some(f=>row[`r1_${f}`]!==row[`r2_${f}`])}
function save(){localStorage.setItem(storageKey,JSON.stringify(rows));progress()}
function textOK(value){const s=String(value||'');return !!s.trim()&&!/^[=+\-@]/.test(s.trimStart())}
function complete(row){const id=row.adjudicator_id.trim();return !!id&&finalFields.every(f=>(schema.fields[f]||[]).includes(row[`final_${f}`]))&&textOK(row.adjudication_rationale)&&(!disagrees(row)||textOK(row.disagreement_reason))}
function progress(){$('progress').textContent=`${rows.filter(complete).length}/${rows.length} edges complete`}
function selectField(row,field){const key=`final_${field}`,l=document.createElement('label');l.textContent=key.replaceAll('_',' ');const s=document.createElement('select');s.innerHTML='<option value="">— select —</option>'+schema.fields[field].map(v=>`<option value="${esc(v)}" ${row[key]===v?'selected':''}>${esc(v)}</option>`).join('');s.onchange=()=>{row[key]=s.value;save()};l.appendChild(s);return l}
function reviewBlock(row,n){return `<div class="review"><strong>Reviewer ${n}</strong>${reviewFields.map(f=>`<div><b>${esc(f)}:</b> ${esc(row[`r${n}_${f}`])}</div>`).join('')}<div><b>rationale:</b> ${esc(row[`r${n}_rationale`])}</div></div>`}
function render(){const gs=groups(),group=gs[groupIndex%gs.length],row0=rows[group.indices[0]],candidate0=pack.candidates[row0.edge_id];$('position').textContent=`${groupIndex+1}/${gs.length} images · ${group.indices.length} edge${group.indices.length===1?'':'s'}`;$('question').textContent=candidate0.question;$('answer').textContent=candidate0.answer_span;$('image').src=candidate0.image_relpath;const ids=new Set(rows.map(r=>r.adjudicator_id).filter(Boolean));$('adjudicator-id').value=ids.size===1?[...ids][0]:'';const out=$('edges');out.innerHTML='';group.indices.forEach((ri,local)=>{const row=rows[ri],candidate=pack.candidates[row.edge_id],card=document.createElement('section');card.className='edge';card.innerHTML=`<h3>Edge ${local+1}: ${esc(row.edge_id)}</h3><div class="grid"><div class="wide"><strong>Parent proposal</strong><div class="fixed">${esc(candidate.parent_proposal)}</div></div><div class="wide"><strong>Child proposal</strong><div class="fixed">${esc(candidate.child_proposal)}</div></div><div><strong>Added constraint</strong><div class="fixed">${esc(candidate.added_constraint_proposal)}</div></div><div><strong>Edge type</strong><div class="fixed">${esc(candidate.edge_type)}</div></div></div><div class="reviews">${reviewBlock(row,1)}${reviewBlock(row,2)}</div>`;const form=document.createElement('div');form.className='grid';finalFields.forEach(f=>form.appendChild(selectField(row,f)));const disagreement=document.createElement('label');disagreement.className='wide';disagreement.textContent=`Disagreement reason ${disagrees(row)?'(required)':'(optional)'}`;const d=document.createElement('textarea');d.value=row.disagreement_reason;d.oninput=()=>{row.disagreement_reason=d.value;save()};disagreement.appendChild(d);form.appendChild(disagreement);const rationale=document.createElement('label');rationale.className='wide';rationale.textContent='Adjudication rationale (required)';const a=document.createElement('textarea');a.value=row.adjudication_rationale;a.oninput=()=>{row.adjudication_rationale=a.value;save()};rationale.appendChild(a);form.appendChild(rationale);card.appendChild(form);out.appendChild(card)});progress();window.scrollTo(0,0)}
$('adjudicator-id').oninput=e=>{rows.forEach(r=>r.adjudicator_id=e.target.value);save()};$('prev').onclick=()=>{const n=groups().length;groupIndex=(groupIndex+n-1)%n;render()};$('next').onclick=()=>{groupIndex=(groupIndex+1)%groups().length;render()};
function validate(){const errors=[],ids=new Set(rows.map(r=>r.adjudicator_id.trim()));if(ids.size!==1||ids.has(''))errors.push('adjudicator ID must be one stable nonempty value');rows.forEach((row,i)=>{finalFields.forEach(f=>{if(!(schema.fields[f]||[]).includes(row[`final_${f}`]))errors.push(`row ${i+1}: invalid final_${f}`)});if(!textOK(row.adjudication_rationale))errors.push(`row ${i+1}: blank or formula-like adjudication rationale`);if(disagrees(row)&&!textOK(row.disagreement_reason))errors.push(`row ${i+1}: disagreement reason required`);const admitted=row.final_edge_entailment_admitted,parent=row.final_parent_visual_support,child=row.final_child_visual_support,source=row.final_increment_observability,scope=row.final_logical_scope_preserved;if(admitted==='yes'&&!['yes','not_applicable'].includes(scope))errors.push(`row ${i+1}: admitted edge must preserve scope`);if(admitted==='yes'&&child==='supported'&&parent!=='supported')errors.push(`row ${i+1}: supported child requires supported parent`);if(admitted==='yes'&&['requires_other_view_or_sequence','requires_history_lab_pathology_or_prior','fundamentally_nonvisual_knowledge'].includes(source)&&child!=='unobservable')errors.push(`row ${i+1}: unavailable source requires child unobservable`);if(admitted==='yes'&&source==='observable_on_supplied_image'&&child==='unobservable')errors.push(`row ${i+1}: observable increment conflicts with unobservable child`)});$('message').className=errors.length?'error':'ok';$('message').textContent=errors.length?`${errors.length} errors; first: ${errors[0]}`:'Browser checks pass; coordinator must run the Python validator.';return errors}
function parseCSV(text){const table=[],row=[];let field='',quoted=false,i=0;function pushField(){row.push(field);field=''}function pushRow(){if(row.length||field){pushField();table.push(row.splice(0))}}while(i<text.length){const c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){field+='"';i+=2;continue}if(c==='"'){quoted=false;i++;continue}field+=c;i++;continue}if(c==='"'){quoted=true;i++;continue}if(c===','){pushField();i++;continue}if(c==='\r'&&text[i+1]==='\n'){pushRow();i+=2;continue}if(c==='\n'||c==='\r'){pushRow();i++;continue}field+=c;i++}if(quoted)throw Error('unterminated quoted CSV field');if(field||row.length)pushRow();if(!table.length)throw Error('empty CSV');const h=table.shift();if(JSON.stringify(h)!==JSON.stringify(header))throw Error('CSV header changed');return table.filter(r=>r.some(x=>x!=='')).map((values,ri)=>{if(values.length!==header.length)throw Error(`row ${ri+2} column count changed`);return Object.fromEntries(header.map((h,j)=>[h,values[j]]))})}
function csvCell(value){const s=String(value==null?'':value);return '"'+s.replaceAll('"','""')+'"'}function stringifyCSV(data){return header.map(csvCell).join(',')+'\n'+data.map(r=>header.map(h=>csvCell(r[h])).join(',')).join('\n')+'\n'}
function immutable(data){const fields=header.filter(f=>!mutableFields.has(f));return JSON.stringify(data.map(r=>fields.map(f=>r[f])))}
$('validate').onclick=validate;$('import').onchange=async e=>{try{const file=e.target.files[0];if(!file)throw Error('no file selected');const parsed=parseCSV(await file.text());if(parsed.length!==seed.length)throw Error('row count changed');if(immutable(parsed)!==immutable(seed))throw Error('immutable copied review content changed');rows=parsed;save();groupIndex=0;render();$('message').className='ok';$('message').textContent='Imported without immutable-content changes.'}catch(err){$('message').className='error';$('message').textContent=`Import rejected: ${err.message}`}};
$('export').onclick=()=>{if(validate().length)return;const blob=new Blob([stringifyCSV(rows)],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='adjudication.completed.csv';a.click();URL.revokeObjectURL(a.href)};
$('export-attestation').onclick=()=>{if(validate().length)return;const missing=[['attest-physician','physician role'],['attest-blinded','private-provenance blinding']].filter(([id])=>!$(id).checked).map(x=>x[1]);if(missing.length){$('message').className='error';$('message').textContent=`Attestation not exported; confirm if true: ${missing.join(', ')}`;return}const id=rows[0].adjudicator_id.trim(),payload={protocol_id:schema.protocol_id,adjudicator:{adjudicator_id:id,role:'physician',blinded_to_private_provenance:true,completed_at_utc:new Date().toISOString()}},blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='adjudicator.attestation.json';a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>'''
    return template.replace("__DATA__", embedded).replace("__BUNDLE__", bundle_id).encode("utf-8")


def package_adjudication(
    *,
    pack_dir: Path,
    merged_csv: Path,
    image_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    schema = json.loads((pack_dir / "annotation_schema.json").read_text(encoding="utf-8"))
    if schema.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("source annotation protocol mismatch")
    candidates_path = pack_dir / "candidates.blinded.jsonl"
    candidates = _read_jsonl(candidates_path)
    header, rows = _read_csv(merged_csv)
    if [row.get("edge_id") for row in rows] != [str(row["edge_id"]) for row in candidates]:
        raise ValueError("merged adjudication rows differ from frozen candidates")
    missing = [field for field in MUTABLE_FIELDS if field not in header]
    if missing:
        raise ValueError(f"merged adjudication CSV missing columns: {missing}")
    if any(row.get(field, "") for row in rows for field in MUTABLE_FIELDS):
        raise ValueError("merged adjudication CSV already contains final values")
    bundle_id = sha256_file(merged_csv)[:24]
    root = f"specificity_ratchet_adjudicator_{bundle_id[:12]}_v1"
    images: dict[str, Path] = {}
    inventory: list[str] = []
    for relative in sorted({str(row["image_relpath"]) for row in candidates}):
        source = image_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        if Path(relative).stem != digest:
            raise ValueError(f"image filename/hash mismatch: {relative}")
        images[f"{root}/{relative}"] = source
        inventory.append(f"{digest}  {relative}\n")
    csv_bytes = merged_csv.read_bytes()
    schema_bytes = canonical_json(schema)
    inventory_bytes = "".join(inventory).encode("utf-8")
    instructions = _instructions(bundle_id)
    form = _form(
        header=header,
        rows=rows,
        candidates=candidates,
        schema=schema,
        bundle_id=bundle_id,
    )
    manifest = {
        "version": VERSION,
        "source_protocol": PROTOCOL_ID,
        "bundle_id": bundle_id,
        "merged_csv_sha256": sha256_bytes(csv_bytes),
        "candidate_sha256": sha256_file(candidates_path),
        "rows": len(rows),
        "cases": len({row["case_id"] for row in candidates}),
        "images": {"count": len(images), "inventory_sha256": sha256_bytes(inventory_bytes)},
        "form_sha256": sha256_bytes(form),
        "private_provenance_in_archive": False,
        "model_identity_in_archive": False,
        "clinical_labels_created": False,
        "adjudicator_attestation_export_required": True,
        "archive_member_file_count": len(images) + 6,
    }
    byte_members = {
        f"{root}/INSTRUCTIONS.md": instructions,
        f"{root}/ADJUDICATION_FORM.html": form,
        f"{root}/adjudication.with_reviews.csv": csv_bytes,
        f"{root}/annotation_schema.json": schema_bytes,
        f"{root}/IMAGE_SHA256SUMS": inventory_bytes,
        f"{root}/DELIVERY_MANIFEST.json": canonical_json(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root}.tar.gz"
    _write_archive(archive, byte_members, images)
    result = {
        **manifest,
        "root": root,
        "archive": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
    }
    index = output_dir / "adjudicator_delivery.json"
    payload = canonical_json(result)
    if index.exists() and index.read_bytes() != payload:
        raise FileExistsError(f"adjudicator delivery collision: {index}")
    index.write_bytes(payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--merged-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_adjudication(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
