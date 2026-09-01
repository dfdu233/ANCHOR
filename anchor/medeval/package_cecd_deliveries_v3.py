#!/usr/bin/env python3
"""Build deterministic CECD v3 reviewer archives with offline forms.

The frozen CECD v2 admission pack is never modified.  Each output archive is
role-isolated, contains exactly one blank source sheet, and exports a separate
explicit human attestation.  No clinical or language decision is synthesized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    CLINICAL_FIELDS,
    LANGUAGE_FIELDS,
    ROLES,
    SOURCE_VERSION,
    _source_role,
    _write_deterministic_tar_gz,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


VERSION = "cecd-blinded-reviewer-delivery-v3"
PROFESSIONAL_ROLE = {
    "clinical_reviewer_1": "physician",
    "clinical_reviewer_2": "physician",
    "clinical_template_reviewer": "physician",
    "language_reviewer": "language_expert",
}
ALLOWED = {
    "support_state_same_supported_refuted_undetermined": ["yes", "no", "unable"],
    "lesion_visibility": ["unchanged", "A_clearer", "B_clearer", "unable"],
    "clinically_interchangeable": ["yes", "no", "unable"],
    "unable_to_judge": ["yes", "no"],
    "same_clinical_proposition": ["yes", "no", "unable"],
    "same_speech_act": ["yes", "no", "unable"],
    "same_certainty_demand": ["yes", "no", "unable"],
    "same_answer_space": ["yes", "no", "unable"],
}


def v3_root(role: str) -> str:
    return f"cecd_{role}_v3"


def _instructions(role: str, sheet: str, kind: str) -> bytes:
    professional = PROFESSIONAL_ROLE[role].replace("_", " ")
    completed_sheet = sheet.replace(".csv", ".completed.csv")
    if kind == "clinical":
        task = """Inspect image A and image B for the named finding. Complete the three primary fields and `unable_to_judge`; if any primary field is `unable`, `unable_to_judge` must be `yes`, otherwise it must be `no`."""
    else:
        task = """Compare the two wordings without answering the medical question. Judge clinical proposition, speech act, certainty demand, and answer space separately."""
    return f"""# Independent blinded CECD review — {role}

Extract the entire assigned archive into one folder, keep the `images/` folder
next to `REVIEW_FORM.html`, then open `REVIEW_FORM.html` in Chrome or Edge.
Do not open the form inside an archive preview and do not edit the CSV in
spreadsheet software.

{task}

1. Work independently and do not consult another reviewer.
2. Do not infer why a pair was included or attempt to recover the sealed
   mapping, source image identity, transform, baseline side, votes, or model
   outputs.
3. Use one stable non-identifying reviewer ID. Complete every required field;
   comments are optional and may not begin with `=`, `+`, `-`, or `@`.
4. The form protects frozen fields, autosaves locally, rejects changed source
   content, and exports `{completed_sheet}`.
5. Check the attestations only if true, then export both the completed CSV and
   `{role}.attestation.json`. The coordinator may validate the statement but
   may not sign it for you.
6. Return only those two files under the exact names shown by the form.

Expected professional role for this slot: `{professional}`. Browser validation
is ergonomic only; the fail-closed Python validator remains authoritative.
""".encode("utf-8")


def _form(
    *,
    role: str,
    kind: str,
    sheet: str,
    header: list[str],
    rows: list[dict[str, str]],
    decision_fields: tuple[str, ...],
) -> bytes:
    schema = {
        "protocol_id": SOURCE_VERSION,
        "delivery_version": VERSION,
        "role": role,
        "kind": kind,
        "professional_role": PROFESSIONAL_ROLE[role],
        "sheet": sheet,
        "header": header,
        "rows": rows,
        "decision_fields": list(decision_fields),
        "allowed": {field: ALLOWED[field] for field in decision_fields if field != "comments"},
    }
    embedded = json.dumps(schema, ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CECD blinded review</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f3f6f8;color:#17202a}header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #c8d0d8;padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.wrap{max-width:1320px;margin:auto;padding:16px}.panel{background:#fff;border:1px solid #c8d0d8;border-radius:8px;padding:14px;margin:12px 0}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.image{display:block;width:100%;max-height:720px;object-fit:contain;background:#111}.wording{white-space:pre-wrap;background:#edf1f4;border-radius:6px;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}label{display:flex;flex-direction:column;gap:4px;font-size:13px;font-weight:650}select,input,textarea,button{font:inherit}select,input,textarea{padding:7px;border:1px solid #aab5c0;border-radius:5px;background:#fff}textarea{min-height:80px}button{padding:7px 11px;border:1px solid #286c7e;border-radius:5px;background:#286c7e;color:#fff;cursor:pointer}button.secondary{background:#fff;color:#273142}.muted{color:#65727e;font-size:13px}.pill{background:#e7ebef;border-radius:18px;padding:4px 9px}.error{color:#a40000;font-weight:700}.ok{color:#176b32;font-weight:700}.wide{grid-column:1/-1}@media(max-width:800px){.images{grid-template-columns:1fr}}
</style></head><body><header><strong>CECD blinded review — __ROLE__</strong><span id="position" class="pill"></span><span id="progress" class="pill"></span><label>Stable reviewer ID<input id="reviewer-id" autocomplete="off"></label><button id="prev" class="secondary">Previous</button><button id="next" class="secondary">Next</button><button id="validate" class="secondary">Validate all</button><button id="export">Export completed CSV</button><button id="export-attestation">Export attestation JSON</button><label>Import CSV<input id="import" type="file" accept=".csv"></label><label><input id="attest-qualified" type="checkbox">I hold the stated professional role</label><label><input id="attest-independent" type="checkbox">I reviewed independently</label><label><input id="attest-blinded" type="checkbox">I remained blinded to sealed mapping</label><span id="message"></span></header><main class="wrap"><section class="panel"><p class="muted">Every pair is a blinded equivalence proposal, not truth. Hidden transform, votes, baseline side, source IDs and model outputs are absent.</p><h2 id="title"></h2><div id="stimulus"></div></section><section class="panel"><div id="fields" class="grid"></div></section></main><script id="seed" type="application/json">__DATA__</script><script>
"use strict";const pack=JSON.parse(document.getElementById('seed').textContent),seed=pack.rows,header=pack.header,decisionFields=pack.decision_fields,allowed=pack.allowed,role=pack.role,kind=pack.kind,storageKey=`cecd-v3:${pack.delivery_version}:${role}`,idKey=`${storageKey}:reviewer-id`;let rows=JSON.parse(JSON.stringify(seed)),index=0;try{const saved=localStorage.getItem(storageKey);if(saved)rows=JSON.parse(saved)}catch(e){}const $=id=>document.getElementById(id),esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function commentOK(v){return !v.trim()||!/^[=+\-@]/.test(v.trimStart())}function rowErrors(row,i){const errors=[];decisionFields.filter(f=>f!=='comments').forEach(f=>{if(!(allowed[f]||[]).includes(row[f]))errors.push(`row ${i+1}: invalid ${f}`)});if(!commentOK(row.comments||''))errors.push(`row ${i+1}: formula-like comments prefix`);if(kind==='clinical'){const anyUnable=['support_state_same_supported_refuted_undetermined','lesion_visibility','clinically_interchangeable'].some(f=>row[f]==='unable');if(anyUnable!==(row.unable_to_judge==='yes'))errors.push(`row ${i+1}: unable_to_judge is inconsistent`)}return errors}function complete(row,i){return rowErrors(row,i).length===0}function save(){localStorage.setItem(storageKey,JSON.stringify(rows));progress()}function progress(){$('progress').textContent=`${rows.filter(complete).length}/${rows.length} complete`}
function render(){const row=rows[index];$('position').textContent=`${index+1}/${rows.length}`;$('title').textContent=kind==='clinical'?`Finding: ${row.finding}`:'Compare task semantics';const stimulus=$('stimulus');if(kind==='clinical'){stimulus.innerHTML=`<div class="images"><div><strong>Image A</strong><img class="image" id="image-a" src="${esc(row.image_A)}"></div><div><strong>Image B</strong><img class="image" id="image-b" src="${esc(row.image_B)}"></div></div>`}else{stimulus.innerHTML=`<div class="grid"><div><strong>Wording A</strong><div class="wording">${esc(row.wording_A)}</div></div><div><strong>Wording B</strong><div class="wording">${esc(row.wording_B)}</div></div></div>`}const fields=$('fields');fields.innerHTML='';decisionFields.filter(f=>f!=='comments').forEach(field=>{const label=document.createElement('label');label.textContent=field.replaceAll('_',' ');const select=document.createElement('select');select.innerHTML='<option value="">— select —</option>'+allowed[field].map(v=>`<option value="${esc(v)}" ${row[field]===v?'selected':''}>${esc(v)}</option>`).join('');select.onchange=()=>{row[field]=select.value;save()};label.appendChild(select);fields.appendChild(label)});const comments=document.createElement('label');comments.className='wide';comments.textContent='Comments (optional; may not begin = + - @)';const area=document.createElement('textarea');area.value=row.comments||'';area.oninput=()=>{row.comments=area.value;save()};comments.appendChild(area);fields.appendChild(comments);progress();window.scrollTo(0,0)}
$('reviewer-id').value=localStorage.getItem(idKey)||'';$('reviewer-id').oninput=e=>localStorage.setItem(idKey,e.target.value);$('prev').onclick=()=>{index=(index+rows.length-1)%rows.length;render()};$('next').onclick=()=>{index=(index+1)%rows.length;render()};
function validate(){const errors=[];if(!$('reviewer-id').value.trim())errors.push('stable reviewer ID is required');rows.forEach((row,i)=>errors.push(...rowErrors(row,i)));$('message').className=errors.length?'error':'ok';$('message').textContent=errors.length?`${errors.length} errors; first: ${errors[0]}`:'Browser checks pass; coordinator must run the Python validator.';return errors}
function parseCSV(text){const table=[],row=[];let field='',quoted=false,i=0;function pushField(){row.push(field);field=''}function pushRow(){if(row.length||field){pushField();table.push(row.splice(0))}}while(i<text.length){const c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){field+='"';i+=2;continue}if(c==='"'){quoted=false;i++;continue}field+=c;i++;continue}if(c==='"'){quoted=true;i++;continue}if(c===','){pushField();i++;continue}if(c==='\r'&&text[i+1]==='\n'){pushRow();i+=2;continue}if(c==='\n'||c==='\r'){pushRow();i++;continue}field+=c;i++}if(quoted)throw Error('unterminated quoted CSV field');if(field||row.length)pushRow();if(!table.length)throw Error('empty CSV');const h=table.shift();if(JSON.stringify(h)!==JSON.stringify(header))throw Error('CSV header changed');return table.filter(r=>r.some(x=>x!=='')).map((values,ri)=>{if(values.length!==header.length)throw Error(`row ${ri+2} column count changed`);return Object.fromEntries(header.map((h,j)=>[h,values[j]]))})}function csvCell(v){const s=String(v==null?'':v);return '"'+s.replaceAll('"','""')+'"'}function stringifyCSV(data){return header.map(csvCell).join(',')+'\n'+data.map(r=>header.map(h=>csvCell(r[h])).join(',')).join('\n')+'\n'}const mutable=new Set(decisionFields);function immutable(data){const fields=header.filter(f=>!mutable.has(f));return JSON.stringify(data.map(r=>fields.map(f=>r[f])))}
$('validate').onclick=validate;$('import').onchange=async e=>{try{const file=e.target.files[0];if(!file)throw Error('no file selected');const parsed=parseCSV(await file.text());if(parsed.length!==seed.length)throw Error('row count changed');if(immutable(parsed)!==immutable(seed))throw Error('immutable reviewer-visible content changed');rows=parsed;save();index=0;render();$('message').className='ok';$('message').textContent='Imported without immutable-content changes.'}catch(err){$('message').className='error';$('message').textContent=`Import rejected: ${err.message}`}};
$('export').onclick=()=>{if(validate().length)return;const blob=new Blob([stringifyCSV(rows)],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=pack.sheet.replace('.csv','.completed.csv');a.click();URL.revokeObjectURL(a.href)};
$('export-attestation').onclick=()=>{if(validate().length)return;const missing=[['attest-qualified','professional role'],['attest-independent','independent review'],['attest-blinded','sealed-mapping blinding']].filter(([id])=>!$(id).checked).map(x=>x[1]);if(missing.length){$('message').className='error';$('message').textContent=`Attestation not exported; confirm if true: ${missing.join(', ')}`;return}const payload={protocol_id:pack.protocol_id,review_role:role,reviewer:{reviewer_id:$('reviewer-id').value.trim(),professional_role:pack.professional_role,independent_review:true,blinded_to_sealed_mapping:true,completed_at_utc:new Date().toISOString()}},blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${role}.attestation.json`;a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>'''
    return template.replace("__DATA__", embedded).replace("__ROLE__", role).encode("utf-8")


def build_role_v3(
    pack_dir: Path,
    output_dir: Path,
    role: str,
    image_hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    source = _source_role(pack_dir, role)
    spec = source["spec"]
    root = v3_root(role)
    decision_fields = CLINICAL_FIELDS if spec["kind"] == "clinical" else LANGUAGE_FIELDS
    header = list(source["rows"][0])
    instructions = _instructions(role, spec["sheet"], spec["kind"])
    form = _form(
        role=role,
        kind=spec["kind"],
        sheet=spec["sheet"],
        header=header,
        rows=source["rows"],
        decision_fields=decision_fields,
    )
    review_schema = {
        "protocol_id": SOURCE_VERSION,
        "delivery_version": VERSION,
        "role": role,
        "kind": spec["kind"],
        "professional_role": PROFESSIONAL_ROLE[role],
        "sheet": spec["sheet"],
        "header": header,
        "decision_fields": list(decision_fields),
        "allowed": {field: ALLOWED[field] for field in decision_fields if field != "comments"},
    }
    schema_bytes = canonical_json(review_schema)
    cache = image_hash_cache if image_hash_cache is not None else {}
    image_members: dict[str, Path] = {}
    inventory: list[str] = []
    for relative in source["images"]:
        path = pack_dir / relative
        digest = cache.get(path)
        if digest is None:
            digest = sha256_file(path)
            cache[path] = digest
        image_members[f"{root}/{relative}"] = path
        inventory.append(f"{digest}  {relative}\n")
    inventory_bytes = "".join(inventory).encode("utf-8")
    manifest = {
        "version": VERSION,
        "source_version": SOURCE_VERSION,
        "role": role,
        "kind": spec["kind"],
        "professional_role": PROFESSIONAL_ROLE[role],
        "review_sheet": {
            "filename": spec["sheet"],
            "rows": len(source["rows"]),
            "sha256": sha256_bytes(source["sheet_bytes"]),
            "header": header,
        },
        "images": {
            "included": bool(source["images"]),
            "count": len(image_members),
            "inventory_sha256": sha256_bytes(inventory_bytes),
        },
        "instructions_sha256": sha256_bytes(instructions),
        "form_sha256": sha256_bytes(form),
        "schema_sha256": sha256_bytes(schema_bytes),
        "attestation_export_required": True,
        "blinded_role_isolation": True,
        "sealed_mapping_in_archive": False,
        "model_outputs_in_archive": False,
        "clinical_or_language_labels_created": False,
        "archive_member_file_count": len(image_members) + 6,
    }
    byte_members = {
        f"{root}/INSTRUCTIONS.md": instructions,
        f"{root}/REVIEW_FORM.html": form,
        f"{root}/{spec['sheet']}": source["sheet_bytes"],
        f"{root}/REVIEW_SCHEMA.json": schema_bytes,
        f"{root}/IMAGE_SHA256SUMS": inventory_bytes,
        f"{root}/DELIVERY_MANIFEST.json": canonical_json(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root}.tar.gz"
    _write_deterministic_tar_gz(archive, byte_members, image_members)
    return {
        "role": role,
        "root": root,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_member_file_count": manifest["archive_member_file_count"],
    }


def package_deliveries(pack_dir: Path, output_dir: Path) -> dict[str, Any]:
    cache: dict[Path, str] = {}
    records = [
        build_role_v3(pack_dir, output_dir, role, cache)
        for role in ROLES
    ]
    index = {
        "version": VERSION,
        "source_version": SOURCE_VERSION,
        "source_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "archives": records,
        "clinical_or_language_labels_created": False,
    }
    (output_dir / "delivery_index.json").write_bytes(canonical_json(index))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_deliveries(args.pack_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
