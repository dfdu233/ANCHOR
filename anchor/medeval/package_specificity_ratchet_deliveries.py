#!/usr/bin/env python3
"""Build deterministic, role-isolated Specificity Ratchet review archives v3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import (
    COPY_FIELDS,
    REVIEW_FIELDS,
)
from anchor.medeval.package_physician_oe_deliveries import (
    _write_archive,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


VERSION = "specificity-ratchet-review-delivery-v3"
SOURCE_PROTOCOL = "specificity-ratchet-physician-pack-v2"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instructions(role: int, csv_name: str, bundle_id: str) -> bytes:
    return f"""# Independent blinded physician review — role {role}

Bundle: `{bundle_id}`

Every parent-to-child edge is a linguistic proposal, never clinical truth.
Inspect the supplied image and text independently. Do not infer correctness
from fluency, repetition, or presumed model identity.

1. Work independently and keep this directory private from the other reviewer.
2. Use `REVIEW_FORM.html` rather than spreadsheet software. It groups all edges
   from one image, protects frozen fields, autosaves locally, and exports the
   required UTF-8 CSV.
3. Enter one stable, nonempty reviewer ID and complete all categorical fields
   plus a rationale for every edge.
4. Judge edge entailment first, then parent and child visual support. Preserve
   alternatives such as `A OR B` as one uncertain set.
5. `undetermined` means image-observable in principle but insufficient here;
   `unobservable` means another view/sequence, history, laboratory, pathology,
   prior study, or nonvisual knowledge is required.
6. A five-image calibration may be done on a disposable extraction. Discard it
   and re-extract the archive before formal independent review.
7. Check the three attestation boxes only if true, then export both
   `{csv_name}` and `reviewer_{role}.attestation.json`. The coordinator may
   validate and combine these statements but may not sign them for you.
8. The coordinator's Python merger is authoritative; a browser validation pass
   is not clinical or schema proof.

Never edit the exported CSV in Excel. Five frozen proposal cells begin with a
literal hyphen and must remain text. Never add patient identifiers to reviewer
IDs or rationales.
""".encode("utf-8")


def _review_form(
    *,
    header: list[str],
    rows: list[dict[str, str]],
    schema: dict[str, Any],
    role: int,
    bundle_id: str,
) -> bytes:
    payload = {
        "header": header,
        "rows": rows,
        "schema": schema,
        "review_fields": list(REVIEW_FIELDS),
        "copy_fields": list(COPY_FIELDS),
    }
    embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Specificity Ratchet blinded review</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#17202a}header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #c8d0d8;padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.wrap{max-width:1200px;margin:auto;padding:16px}.panel,.edge{background:#fff;border:1px solid #c8d0d8;border-radius:8px;padding:14px;margin:12px 0}.edge{border-left:5px solid #526bc5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:10px}label{display:flex;flex-direction:column;gap:4px;font-size:13px;font-weight:650}select,input,textarea,button{font:inherit}select,input,textarea{padding:7px;border:1px solid #aab5c0;border-radius:5px;background:#fff}textarea{min-height:70px}button{padding:7px 11px;border:1px solid #526bc5;border-radius:5px;background:#526bc5;color:white;cursor:pointer}button.secondary{background:white;color:#273142}.image{display:block;max-width:100%;max-height:650px;margin:auto;background:#111}.fixed{white-space:pre-wrap;background:#edf1f4;border-radius:6px;padding:9px}.muted{color:#65727e;font-size:13px}.pill{background:#e7ebef;border-radius:18px;padding:4px 9px}.error{color:#a40000;font-weight:700}.ok{color:#176b32;font-weight:700}.wide{grid-column:1/-1}.proposal{border-left:3px solid #9aa9b7;padding-left:9px}.edge h3{margin-top:0}
</style></head><body><header><strong>Specificity Ratchet — reviewer __ROLE__</strong><span id="position" class="pill"></span><span id="progress" class="pill"></span><label>Stable reviewer ID<input id="reviewer-id" autocomplete="off"></label><button id="prev" class="secondary">Previous image</button><button id="next" class="secondary">Next image</button><button id="validate" class="secondary">Validate all</button><button id="export">Export completed CSV</button><button id="export-attestation">Export attestation JSON</button><label>Import CSV<input id="import" type="file" accept=".csv"></label><label><input id="attest-physician" type="checkbox">I am a physician</label><label><input id="attest-independent" type="checkbox">I reviewed independently</label><label><input id="attest-blinded" type="checkbox">I remained blinded to private provenance</label><span id="message"></span></header><main class="wrap"><section class="panel"><p class="muted">Bundle __BUNDLE__. Every edge is a proposal, not truth. Work independently; model identity and private provenance are absent.</p><h2 id="question"></h2><img id="image" class="image"><p><strong>Observed answer span</strong></p><div id="answer" class="fixed"></div></section><section id="edges"></section></main><script id="seed" type="application/json">__DATA__</script><script>
"use strict";
const pack=JSON.parse(document.getElementById('seed').textContent),seed=pack.rows,header=pack.header,schema=pack.schema,reviewFields=pack.review_fields,copyFields=pack.copy_fields,role='__ROLE__',bundle='__BUNDLE__',storageKey=`specificity-ratchet-v3:${bundle}:${role}`;
const immutableFields=header.filter(x=>!['reviewer_id',...copyFields].includes(x));let rows=JSON.parse(JSON.stringify(seed)),groupIndex=0;try{const saved=localStorage.getItem(storageKey);if(saved)rows=JSON.parse(saved)}catch(e){}
const $=id=>document.getElementById(id), esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function groups(){const out=[],by=new Map();rows.forEach((row,i)=>{if(!by.has(row.case_id)){const g={case_id:row.case_id,indices:[]};by.set(row.case_id,g);out.push(g)}by.get(row.case_id).indices.push(i)});return out}
function save(){localStorage.setItem(storageKey,JSON.stringify(rows));progress()}
function complete(row){const id=row.reviewer_id.trim();return !!id&&reviewFields.every(f=>(schema.fields[f]||[]).includes(row[f]))&&row.rationale.trim()&&!/^[=+\-@]/.test(row.rationale.trimStart())}
function progress(){const done=rows.filter(complete).length;$('progress').textContent=`${done}/${rows.length} edges complete`}
function selectField(row,field){const l=document.createElement('label');l.textContent=field.replaceAll('_',' ');const s=document.createElement('select');s.innerHTML='<option value="">— select —</option>'+schema.fields[field].map(v=>`<option value="${esc(v)}" ${row[field]===v?'selected':''}>${esc(v)}</option>`).join('');s.onchange=()=>{row[field]=s.value;save()};l.appendChild(s);return l}
function render(){const gs=groups(),group=gs[groupIndex%gs.length],first=rows[group.indices[0]];$('position').textContent=`${groupIndex+1}/${gs.length} images · ${group.indices.length} edge${group.indices.length===1?'':'s'}`;$('question').textContent=first.question;$('answer').textContent=first.answer_span;$('image').src=first.image_relpath;const ids=new Set(rows.map(r=>r.reviewer_id).filter(Boolean));$('reviewer-id').value=ids.size===1?[...ids][0]:'';const out=$('edges');out.innerHTML='';group.indices.forEach((ri,local)=>{const row=rows[ri],card=document.createElement('section');card.className='edge';card.innerHTML=`<h3>Edge ${local+1}: ${esc(row.edge_id)}</h3><div class="grid"><div class="proposal wide"><strong>Parent proposal</strong><div class="fixed">${esc(row.parent_proposal)}</div></div><div class="proposal wide"><strong>Child proposal</strong><div class="fixed">${esc(row.child_proposal)}</div></div><div><strong>Added constraint</strong><div class="fixed">${esc(row.added_constraint_proposal)}</div></div><div><strong>Edge type</strong><div class="fixed">${esc(row.edge_type)}</div></div></div>`;const form=document.createElement('div');form.className='grid';reviewFields.forEach(f=>form.appendChild(selectField(row,f)));const rationale=document.createElement('label');rationale.className='wide';rationale.textContent='Rationale (required; may not begin = + - @)';const area=document.createElement('textarea');area.value=row.rationale;area.oninput=()=>{row.rationale=area.value;save()};rationale.appendChild(area);form.appendChild(rationale);card.appendChild(form);out.appendChild(card)});progress();window.scrollTo(0,0)}
$('reviewer-id').oninput=e=>{rows.forEach(r=>r.reviewer_id=e.target.value);save()};$('prev').onclick=()=>{const n=groups().length;groupIndex=(groupIndex+n-1)%n;render()};$('next').onclick=()=>{groupIndex=(groupIndex+1)%groups().length;render()};
function validate(){const errors=[],ids=new Set(rows.map(r=>r.reviewer_id.trim()));if(ids.size!==1||ids.has(''))errors.push('reviewer ID must be one stable nonempty value');rows.forEach((row,i)=>{reviewFields.forEach(f=>{if(!(schema.fields[f]||[]).includes(row[f]))errors.push(`row ${i+1}: invalid ${f}`)});if(!row.rationale.trim())errors.push(`row ${i+1}: blank rationale`);if(/^[=+\-@]/.test(row.rationale.trimStart()))errors.push(`row ${i+1}: formula-like rationale prefix`) });$('message').className=errors.length?'error':'ok';$('message').textContent=errors.length?`${errors.length} errors; first: ${errors[0]}`:'Browser checks pass; coordinator must run the Python merger.';return errors}
function parseCSV(text){const table=[],row=[];let field='',quoted=false,i=0;function pushField(){row.push(field);field=''}function pushRow(){if(row.length||field){pushField();table.push(row.splice(0))}}while(i<text.length){const c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){field+='"';i+=2;continue}if(c==='"'){quoted=false;i++;continue}field+=c;i++;continue}if(c==='"'){quoted=true;i++;continue}if(c===','){pushField();i++;continue}if(c==='\r'&&text[i+1]==='\n'){pushRow();i+=2;continue}if(c==='\n'||c==='\r'){pushRow();i++;continue}field+=c;i++}if(quoted)throw Error('unterminated quoted CSV field');if(field||row.length)pushRow();if(!table.length)throw Error('empty CSV');const h=table.shift();if(JSON.stringify(h)!==JSON.stringify(header))throw Error('CSV header changed');return table.filter(r=>r.some(x=>x!=='')).map((values,ri)=>{if(values.length!==header.length)throw Error(`row ${ri+2} column count changed`);return Object.fromEntries(header.map((h,j)=>[h,values[j]]))})}
function csvCell(value){const s=String(value==null?'':value);return '"'+s.replaceAll('"','""')+'"'}function stringifyCSV(data){return header.map(csvCell).join(',')+'\n'+data.map(r=>header.map(h=>csvCell(r[h])).join(',')).join('\n')+'\n'}
function immutable(data){return JSON.stringify(data.map(r=>immutableFields.map(f=>r[f])))}
$('validate').onclick=validate;$('import').onchange=async e=>{try{const file=e.target.files[0];if(!file)throw Error('no file selected');const parsed=parseCSV(await file.text());if(parsed.length!==seed.length)throw Error('row count changed');if(immutable(parsed)!==immutable(seed))throw Error('immutable reviewer-visible content changed');rows=parsed;save();groupIndex=0;render();$('message').className='ok';$('message').textContent='Imported without immutable-content changes.'}catch(err){$('message').className='error';$('message').textContent=`Import rejected: ${err.message}`}};
$('export').onclick=()=>{if(validate().length)return;const blob=new Blob([stringifyCSV(rows)],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`annotations.reviewer_${role}.completed.csv`;a.click();URL.revokeObjectURL(a.href)};
$('export-attestation').onclick=()=>{if(validate().length)return;const checks=[['attest-physician','physician role'],['attest-independent','independent review'],['attest-blinded','private-provenance blinding']],missing=checks.filter(([id])=>!$(id).checked).map(x=>x[1]);if(missing.length){$('message').className='error';$('message').textContent=`Attestation not exported; confirm if true: ${missing.join(', ')}`;return}const reviewerId=rows[0].reviewer_id.trim(),payload={protocol_id:schema.protocol_id,reviewer:{reviewer_id:reviewerId,role:'physician',independent_review:true,blinded_to_private_provenance:true,completed_at_utc:new Date().toISOString()}},blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`reviewer_${role}.attestation.json`;a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>'''
    return (
        template.replace("__DATA__", embedded)
        .replace("__ROLE__", str(role))
        .replace("__BUNDLE__", bundle_id)
        .encode("utf-8")
    )


def build_role_archive(
    *,
    pack_dir: Path,
    image_root: Path,
    output_dir: Path,
    role: int,
    bundle_id: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    csv_path = pack_dir / f"annotations.reviewer_{role}.csv"
    header, rows = read_csv(csv_path)
    candidates = read_jsonl(pack_dir / "candidates.blinded.jsonl")
    if len(rows) != len(candidates) or [r["edge_id"] for r in rows] != [
        str(r["edge_id"]) for r in candidates
    ]:
        raise ValueError(f"reviewer {role}: CSV differs from frozen candidates")
    if any(row.get("reviewer_id") or any(row.get(f) for f in COPY_FIELDS) for row in rows):
        raise ValueError(f"reviewer {role}: source sheet is not blank")
    if set(schema.get("fields", {})) != set(REVIEW_FIELDS):
        raise ValueError("schema and reviewer fields differ")

    root = f"specificity_ratchet_reviewer_{role}_{bundle_id[:12]}_v3"
    image_members: dict[str, Path] = {}
    inventory = []
    for relative in sorted({row["image_relpath"] for row in rows}):
        source = image_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        if Path(relative).stem != digest:
            raise ValueError(f"image filename/hash mismatch: {relative}")
        image_members[f"{root}/{relative}"] = source
        inventory.append(f"{digest}  {relative}\n")
    inventory_bytes = "".join(inventory).encode("utf-8")
    csv_bytes = csv_path.read_bytes()
    schema_bytes = canonical_json(schema)
    instructions = _instructions(role, csv_path.name, bundle_id)
    form = _review_form(
        header=header,
        rows=rows,
        schema=schema,
        role=role,
        bundle_id=bundle_id,
    )
    manifest = {
        "version": VERSION,
        "source_protocol": SOURCE_PROTOCOL,
        "bundle_id": bundle_id,
        "reviewer_role": role,
        "csv": {
            "filename": csv_path.name,
            "sha256": sha256_bytes(csv_bytes),
            "header": header,
            "rows": len(rows),
        },
        "cases": len({row["case_id"] for row in rows}),
        "images": {
            "count": len(image_members),
            "inventory_sha256": sha256_bytes(inventory_bytes),
        },
        "schema_sha256": sha256_bytes(schema_bytes),
        "instructions_sha256": sha256_bytes(instructions),
        "review_form": {
            "sha256": sha256_bytes(form),
            "offline": True,
            "browser_export_requires_python_merge": True,
            "reviewer_attestation_export_required": True,
        },
        "private_provenance_in_archive": False,
        "model_identity_in_archive": False,
        "clinical_labels_created": False,
        "archive_member_file_count": len(image_members) + 6,
    }
    byte_members = {
        f"{root}/INSTRUCTIONS.md": instructions,
        f"{root}/REVIEW_FORM.html": form,
        f"{root}/{csv_path.name}": csv_bytes,
        f"{root}/annotation_schema.json": schema_bytes,
        f"{root}/IMAGE_SHA256SUMS": inventory_bytes,
        f"{root}/DELIVERY_MANIFEST.json": canonical_json(manifest),
    }
    output = output_dir / f"{root}.tar.gz"
    _write_archive(output, byte_members, image_members)
    return {
        "role": role,
        "root": root,
        "archive": output.name,
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_member_file_count": manifest["archive_member_file_count"],
    }


def package_deliveries(
    *, pack_dir: Path, image_root: Path, output_dir: Path
) -> dict[str, Any]:
    schema = json.loads((pack_dir / "annotation_schema.json").read_text(encoding="utf-8"))
    if schema.get("protocol_id") != SOURCE_PROTOCOL:
        raise ValueError("source annotation protocol mismatch")
    candidates_path = pack_dir / "candidates.blinded.jsonl"
    bundle_id = sha256_file(candidates_path)[:24]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        build_role_archive(
            pack_dir=pack_dir,
            image_root=image_root,
            output_dir=output_dir,
            role=role,
            bundle_id=bundle_id,
            schema=schema,
        )
        for role in (1, 2)
    ]
    index = {
        "version": VERSION,
        "source_protocol": SOURCE_PROTOCOL,
        "bundle_id": bundle_id,
        "source_candidates_sha256": sha256_file(candidates_path),
        "archives": records,
    }
    (output_dir / "delivery_index.json").write_bytes(canonical_json(index))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package_deliveries(
                pack_dir=args.pack_dir,
                image_root=args.image_root,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
