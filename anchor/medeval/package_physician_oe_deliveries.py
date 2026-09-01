#!/usr/bin/env python3
"""Package frozen physician-OE sheets into deterministic role-isolated archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any


VERSION = "anchor-physician-oe-review-archive-v2"
SOURCE_VERSION = "anchor-physician-oe-review-deliveries-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(
    output: Path,
    byte_members: dict[str, bytes],
    file_members: dict[str, Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as raw:
        temporary = Path(raw.name)
        try:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    for name in sorted(set(byte_members) | set(file_members)):
                        if name in byte_members:
                            data = byte_members[name]
                            archive.addfile(_tar_info(name, len(data)), io.BytesIO(data))
                        else:
                            path = file_members[name]
                            with path.open("rb") as handle:
                                archive.addfile(_tar_info(name, path.stat().st_size), handle)
            raw.flush()
            os.fsync(raw.fileno())
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _instructions(
    role: str, review_filename: str, bundle_id: str, calibration_groups: int
) -> bytes:
    return f"""# Independent blinded physician review

Bundle: `{bundle_id}`  
Reviewer role: `{role}`

1. Extract this archive and keep its directory private from the other reviewer.
2. Open images only from `images/`; each JSONL row names the matching image by
   its SHA-256 filename.
3. Complete a copy of `{review_filename}` without changing bundle, group,
   question, image, answer, ordering, phase, or reviewer-slot fields.
4. Follow `PHYSICIAN_OE_REVIEW_RUNBOOK.md`. Inspect the image and question
   before reading candidate answers. Preserve alternative scope: `A or B` is
   not two definite diagnoses.
5. Independently complete the first {calibration_groups} `calibration` groups. Do not inspect a
   model name, method name, score, or private mapping. The coordinator will
   freeze a shared clarification log before the remaining `double_review`
   groups are finalized.
6. Return only the completed JSONL and the frozen clarification log. Do not
   rename the source JSONL inside this archive or place patient identifiers in
   free-text rationales.

`REVIEW_FORM.html` is the recommended editor. It protects reviewer-visible
source text, autosaves locally in the browser, and exports the required JSONL.
The Python validator remains authoritative; a successful browser export is not
itself a valid review.

The benchmark answer is context, not automatic truth. Knowledge, unavailable
history, missing comparisons, laboratory information, and other unobservable
claims must not be scored as visual hallucinations.
""".encode("utf-8")


def _review_form(rows: list[dict[str, Any]], role: str, bundle_id: str) -> bytes:
    embedded = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded physician OE review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#17202a}header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd4dc;padding:10px 18px;z-index:3;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.wrap{max-width:1180px;margin:auto;padding:18px}.panel,.answer,.claim{background:#fff;border:1px solid #ccd4dc;border-radius:8px;padding:14px;margin:12px 0}.answer{border-left:5px solid #6574cd}.claim{background:#f8fafc}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}label{display:flex;flex-direction:column;gap:4px;font-size:13px;font-weight:600}select,input,textarea,button{font:inherit}select,input,textarea{padding:7px;border:1px solid #aeb8c2;border-radius:5px;background:#fff}textarea{min-height:58px}button{padding:7px 11px;border:1px solid #6574cd;border-radius:5px;background:#6574cd;color:white;cursor:pointer}button.secondary{background:white;color:#334}button.danger{background:#a33;border-color:#a33}.image{max-width:100%;max-height:650px;display:block;margin:auto;background:#111}.answer-text{white-space:pre-wrap;background:#eef2f5;padding:10px;border-radius:6px}.muted{color:#61707d;font-size:13px}.error{color:#a40000;font-weight:700}.ok{color:#176b32;font-weight:700}.nav{display:flex;gap:8px}.wide{grid-column:1/-1}.claim-head{display:flex;justify-content:space-between;align-items:center}.pill{background:#e7ebef;border-radius:20px;padding:4px 9px}.hidden{display:none}
</style></head><body>
<header><strong>Blinded review — role __ROLE__</strong><span id="position" class="pill"></span><span id="progress" class="pill"></span><div class="nav"><button id="prev" class="secondary">Previous</button><button id="next" class="secondary">Next</button></div><button id="validate" class="secondary">Validate all</button><button id="export">Export completed JSONL</button><label class="secondary">Import JSONL<input id="import" type="file" accept=".jsonl" style="max-width:220px"></label><span id="message"></span></header>
<main class="wrap"><section class="panel"><div class="muted">Bundle __BUNDLE__. Work independently. Model and method identities are intentionally absent. Final acceptance requires the packaged Python validator.</div><h2 id="question"></h2><div><strong>Benchmark reference:</strong> <span id="reference"></span></div><p class="muted">Reference is context, not automatic truth. Inspect the image first.</p><img id="image" class="image"></section><section id="reference-form" class="panel"></section><section id="answers"></section></main>
<script id="seed" type="application/json">__DATA__</script>
<script>
"use strict";
const seed=JSON.parse(document.getElementById("seed").textContent),role="__ROLE__",bundle="__BUNDLE__",storageKey=`anchor-oe-review:${bundle}:${role}`;
let rows=seed,index=0;try{const saved=localStorage.getItem(storageKey);if(saved)rows=JSON.parse(saved)}catch(e){}
const $=id=>document.getElementById(id), opts=(values,value,blank=true)=>`${blank?'<option value="">— select —</option>':''}${values.map(x=>`<option ${x===value?'selected':''}>${x}</option>`).join('')}`;
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const attrString=a=>(a||[]).join(', '), attrList=s=>s.split(',').map(x=>x.trim()).filter(Boolean);
function save(){localStorage.setItem(storageKey,JSON.stringify(rows));progress()}
function selectField(label,values,value,onchange){const l=document.createElement('label');l.textContent=label;const s=document.createElement('select');s.innerHTML=opts(values,value);s.onchange=()=>{onchange(s.value||null);save()};l.appendChild(s);return l}
function inputField(label,value,onchange,wide=false){const l=document.createElement('label');if(wide)l.className='wide';l.textContent=label;const x=document.createElement(wide?'textarea':'input');x.value=value==null?'':value;x.oninput=()=>{onchange(x.value);save()};l.appendChild(x);return l}
function claimEditor(parent,claims,atomic,answerText){parent.innerHTML='';claims.forEach((claim,i)=>{const card=document.createElement('div');card.className='claim';const head=document.createElement('div');head.className='claim-head';head.innerHTML=`<strong>${atomic?'Atomic':'Required'} claim ${i+1}</strong>`;const remove=document.createElement('button');remove.className='danger';remove.textContent='Remove';remove.onclick=()=>{claims.splice(i,1);save();claimEditor(parent,claims,atomic,answerText)};head.appendChild(remove);card.appendChild(head);const g=document.createElement('div');g.className='grid';const n=claim.normalized_claim||(claim.normalized_claim={finding:'',polarity:'present',uncertainty:'definite',anatomy:null,attributes:[]});g.append(inputField('Claim ID',claim.claim_id,v=>claim.claim_id=v));if(atomic)g.append(inputField('Exact text span',claim.text_span,v=>claim.text_span=v,true));g.append(inputField('Finding',n.finding,v=>n.finding=v));g.append(selectField('Polarity',['present','absent'],n.polarity,v=>n.polarity=v));g.append(selectField('Uncertainty',['definite','uncertain','unknown'],n.uncertainty,v=>{n.uncertainty=v;claim.commitment=v}));g.append(inputField('Anatomy (blank = null)',n.anatomy||'',v=>n.anatomy=v.trim()?v:null));g.append(inputField('Attributes (comma separated)',attrString(n.attributes),v=>n.attributes=attrList(v),true));if(atomic){g.append(selectField('Claim type',['visual','knowledge','unobservable'],claim.claim_type,v=>claim.claim_type=v));g.append(selectField('Visual support',['supported','refuted','undetermined','not_applicable'],claim.visual_support,v=>claim.visual_support=v));g.append(selectField('Relevance',['required','optional','out_of_scope'],claim.relevance,v=>claim.relevance=v));g.append(selectField('Error type',['none','fabricated','false_negation','location','attribute','inappropriate_certainty','indeterminate'],claim.error_type,v=>claim.error_type=v));const hint=document.createElement('div');hint.className='wide muted';hint.textContent=`Span must occur verbatim in: ${answerText}`;g.appendChild(hint)}card.appendChild(g);parent.appendChild(card)})}
function addClaim(claims,atomic){const id=`${atomic?'c':'r'}${claims.length+1}`;const c={claim_id:id,normalized_claim:{finding:'',polarity:'present',uncertainty:'definite',anatomy:null,attributes:[]}};if(atomic)Object.assign(c,{text_span:'',claim_type:'visual',visual_support:'supported',commitment:'definite',relevance:'optional',error_type:'none'});claims.push(c);save();render()}
function render(){const row=rows[index];$('position').textContent=`${index+1}/${rows.length} · ${row.review_phase}`;$('question').textContent=row.question;$('reference').textContent=row.benchmark_reference;$('image').src=`images/${row.image.relative_path}`;
const rf=$('reference-form'),r=row.reference_annotation;rf.innerHTML='<h3>Reference-side adjudication</h3>';const rg=document.createElement('div');rg.className='grid';rg.append(selectField('Visual observability',['observable','partially_observable','unobservable','indeterminate'],r.visual_observability,v=>r.visual_observability=v));rg.append(selectField('Benchmark reference correctness',['correct','partially_correct','incorrect','indeterminate'],r.benchmark_reference_correctness,v=>r.benchmark_reference_correctness=v));rg.append(inputField('Notes',r.notes,v=>r.notes=v,true));rf.appendChild(rg);const rc=document.createElement('div');claimEditor(rc,r.required_answer_claims,false,'');rf.appendChild(rc);const rb=document.createElement('button');rb.textContent='Add required claim';rb.onclick=()=>addClaim(r.required_answer_claims,false);rf.appendChild(rb);
const out=$('answers');out.innerHTML='';row.candidate_answers.forEach((candidate,ai)=>{const a=candidate.annotation,card=document.createElement('section');card.className='answer';card.innerHTML=`<h3>Candidate ${ai+1} <span class="muted">${esc(candidate.answer_id)}</span></h3><div class="answer-text">${esc(candidate.answer_text)}</div>`;const g=document.createElement('div');g.className='grid';g.append(selectField('Direct answer correctness',['correct','partially_correct','incorrect','indeterminate'],a.direct_answer_correctness,v=>a.direct_answer_correctness=v));g.append(selectField('Direct answer state',['supported','refuted','undetermined','unobservable'],a.direct_answer_state,v=>a.direct_answer_state=v));const no=document.createElement('label');no.textContent='No clinical claims';const cb=document.createElement('input');cb.type='checkbox';cb.checked=a.no_clinical_claims===true;cb.onchange=()=>{a.no_clinical_claims=cb.checked;if(cb.checked)a.atomic_claims=[];save();render()};no.appendChild(cb);g.appendChild(no);g.append(selectField('Overall clinically harmful',['no','possibly','yes','indeterminate'],a.overall_clinically_harmful,v=>a.overall_clinically_harmful=v));g.append(selectField('Reviewer confidence',['1','2','3','4','5'],a.reviewer_confidence==null?null:String(a.reviewer_confidence),v=>a.reviewer_confidence=v?Number(v):null));g.append(inputField('Omitted required claim IDs (comma separated)',(a.omitted_required_claim_ids||[]).join(', '),v=>a.omitted_required_claim_ids=attrList(v),true));g.append(inputField('Rationale',a.rationale,v=>a.rationale=v,true));card.appendChild(g);const claims=document.createElement('div');claimEditor(claims,a.atomic_claims,true,candidate.answer_text);card.appendChild(claims);if(!a.no_clinical_claims){const b=document.createElement('button');b.textContent='Add atomic claim';b.onclick=()=>{a.no_clinical_claims=false;addClaim(a.atomic_claims,true)};card.appendChild(b)}out.appendChild(card)});progress();window.scrollTo(0,0)}
function groupComplete(row){const r=row.reference_annotation;if(!r.visual_observability||!r.benchmark_reference_correctness)return false;if(['observable','partially_observable'].includes(r.visual_observability)&&!r.required_answer_claims.length)return false;return row.candidate_answers.every(c=>{const a=c.annotation;return a.direct_answer_correctness&&a.direct_answer_state&&typeof a.no_clinical_claims==='boolean'&&a.no_clinical_claims===(a.atomic_claims.length===0)&&a.overall_clinically_harmful&&Number.isInteger(a.reviewer_confidence)})}
function progress(){const done=rows.filter(groupComplete).length;$('progress').textContent=`${done}/${rows.length} groups structurally complete`}
function validate(){const errors=[];rows.forEach((row,gi)=>{if(!groupComplete(row))errors.push(`Group ${gi+1}: required fields incomplete`);const ids=new Set(row.reference_annotation.required_answer_claims.map(c=>c.claim_id));row.candidate_answers.forEach((c,ai)=>{const a=c.annotation;if((a.overall_clinically_harmful!=='no'||a.direct_answer_correctness==='indeterminate')&&!a.rationale.trim())errors.push(`Group ${gi+1}, candidate ${ai+1}: rationale required`);if(a.omitted_required_claim_ids.some(x=>!ids.has(x)))errors.push(`Group ${gi+1}, candidate ${ai+1}: unknown omission ID`);a.atomic_claims.forEach((cl,ci)=>{if(!cl.text_span||!c.answer_text.includes(cl.text_span))errors.push(`Group ${gi+1}, candidate ${ai+1}, claim ${ci+1}: span not verbatim`);if(cl.commitment!==cl.normalized_claim.uncertainty)errors.push(`Group ${gi+1}, candidate ${ai+1}, claim ${ci+1}: uncertainty mismatch`);if(cl.claim_type==='visual'&&cl.visual_support==='not_applicable')errors.push(`Group ${gi+1}, candidate ${ai+1}, claim ${ci+1}: visual support missing`);if(cl.claim_type!=='visual'&&cl.visual_support!=='not_applicable')errors.push(`Group ${gi+1}, candidate ${ai+1}, claim ${ci+1}: nonvisual support must be not_applicable`)})})});$('message').className=errors.length?'error':'ok';$('message').textContent=errors.length?`${errors.length} errors; first: ${errors[0]}`:'Browser checks pass; run Python validator after export.';return errors}
$('prev').onclick=()=>{index=(index+rows.length-1)%rows.length;render()};$('next').onclick=()=>{index=(index+1)%rows.length;render()};$('validate').onclick=validate;$('export').onclick=()=>{if(validate().length)return;const blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`reviewer_${role}.completed.jsonl`;a.click();URL.revokeObjectURL(a.href)};
$('import').onchange=async e=>{try{const parsed=e.target.files[0]?(await e.target.files[0].text()).trim().split(/\n/).filter(Boolean).map(JSON.parse):null;if(!parsed||parsed.length!==seed.length)throw Error('group count mismatch');const imm=x=>JSON.stringify(x.map(r=>[r.bundle_id,r.group_id,r.review_order,r.review_phase,r.reviewer_slot,r.image,r.question,r.benchmark_reference,r.candidate_answers.map(c=>[c.answer_id,c.answer_text])]));if(imm(parsed)!==imm(seed))throw Error('immutable reviewer-visible content changed');rows=parsed;save();render();$('message').className='ok';$('message').textContent='Imported without immutable-content changes.'}catch(err){$('message').className='error';$('message').textContent=`Import rejected: ${err.message}`}};
render();
</script></body></html>"""
    return (
        template.replace("__DATA__", embedded)
        .replace("__ROLE__", role)
        .replace("__BUNDLE__", bundle_id)
        .encode("utf-8")
    )


def _validate_rows(rows: list[dict[str, Any]], role: str, bundle_id: str) -> list[str]:
    if not rows:
        raise RuntimeError(f"reviewer {role}: empty delivery")
    expected_slot = role
    group_ids: set[str] = set()
    answer_ids: set[str] = set()
    images: set[str] = set()
    for row in rows:
        if row.get("bundle_id") != bundle_id:
            raise RuntimeError(f"reviewer {role}: bundle ID mismatch")
        if row.get("reviewer_slot") != expected_slot:
            raise RuntimeError(f"reviewer {role}: reviewer-slot mismatch")
        group_id = str(row.get("group_id", ""))
        if not group_id or group_id in group_ids:
            raise RuntimeError(f"reviewer {role}: invalid or duplicate group ID")
        group_ids.add(group_id)
        image = row.get("image") or {}
        relative = str(image.get("relative_path", ""))
        digest = str(image.get("sha256", ""))
        if relative != f"{digest}.jpg" or len(digest) != 64:
            raise RuntimeError(f"reviewer {role}: non-hash-bound image reference")
        images.add(relative)
        for answer in row.get("candidate_answers", []):
            answer_id = str(answer.get("answer_id", ""))
            if not answer_id or answer_id in answer_ids:
                raise RuntimeError(f"reviewer {role}: invalid or duplicate answer ID")
            answer_ids.add(answer_id)
    return sorted(images)


def build_role_archive(
    *,
    delivery_dir: Path,
    metadata: dict[str, Any],
    delivery_manifest: dict[str, Any],
    runbook: Path,
    output_dir: Path,
    role: str,
) -> dict[str, Any]:
    reviewer_record = delivery_manifest["reviewers"][role]
    review_path = delivery_dir / f"reviewer_{role}.blinded.jsonl"
    review_bytes = review_path.read_bytes()
    if sha256_bytes(review_bytes) != reviewer_record["sha256"]:
        raise RuntimeError(f"reviewer {role}: frozen JSONL hash mismatch")
    rows = load_jsonl_bytes(review_bytes)
    image_names = _validate_rows(rows, role, str(metadata["bundle_id"]))

    image_root = Path(metadata["image_root"])
    image_members: dict[str, Path] = {}
    inventory: list[tuple[str, str]] = []
    root = f"physician_oe_reviewer_{role}_{str(metadata['bundle_id'])[:12]}"
    for name in image_names:
        source = image_root / name
        if not source.is_file():
            raise FileNotFoundError(f"reviewer {role}: missing image {source}")
        expected = name.removesuffix(".jpg")
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(f"reviewer {role}: image hash mismatch for {name}")
        archive_name = f"{root}/images/{name}"
        image_members[archive_name] = source
        inventory.append((name, actual))

    instructions = _instructions(
        role,
        review_path.name,
        str(metadata["bundle_id"]),
        int(delivery_manifest["calibration_groups"]),
    )
    review_form = _review_form(rows, role, str(metadata["bundle_id"]))
    runbook_bytes = runbook.read_bytes()
    clarification_path = delivery_dir / "clarification_log.template.md"
    clarification_bytes = clarification_path.read_bytes()
    inventory_bytes = "".join(
        f"{digest}  images/{name}\n" for name, digest in sorted(inventory)
    ).encode("utf-8")
    role_manifest = {
        "version": VERSION,
        "bundle_id": metadata["bundle_id"],
        "reviewer_role": role,
        "review_jsonl": {
            "filename": review_path.name,
            "sha256": sha256_bytes(review_bytes),
            "groups": len(rows),
            "answer_units": sum(len(row["candidate_answers"]) for row in rows),
        },
        "images": {
            "count": len(image_names),
            "inventory_sha256": sha256_bytes(inventory_bytes),
        },
        "runbook_sha256": sha256_bytes(runbook_bytes),
        "instructions_sha256": sha256_bytes(instructions),
        "review_form": {
            "filename": "REVIEW_FORM.html",
            "sha256": sha256_bytes(review_form),
            "offline": True,
            "browser_export_requires_python_validation": True,
        },
        "clarification_template_sha256": sha256_bytes(clarification_bytes),
        "private_mapping_in_archive": False,
        "method_identity_in_archive": False,
        "unblinding_authorized": False,
        "archive_member_file_count": len(image_members) + 7,
    }
    byte_members = {
        f"{root}/INSTRUCTIONS.md": instructions,
        f"{root}/REVIEW_FORM.html": review_form,
        f"{root}/PHYSICIAN_OE_REVIEW_RUNBOOK.md": runbook_bytes,
        f"{root}/{review_path.name}": review_bytes,
        f"{root}/clarification_log.template.md": clarification_bytes,
        f"{root}/IMAGE_SHA256SUMS": inventory_bytes,
        f"{root}/DELIVERY_MANIFEST.json": canonical_json(role_manifest),
    }
    output = output_dir / f"{root}.tar.gz"
    _write_archive(output, byte_members, image_members)
    return {
        "role": role,
        "archive": output.name,
        "root": root,
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_member_file_count": role_manifest["archive_member_file_count"],
    }


def package_deliveries(
    *, delivery_dir: Path, metadata_path: Path, runbook: Path, output_dir: Path
) -> dict[str, Any]:
    delivery_manifest = json.loads(
        (delivery_dir / "delivery_manifest.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if delivery_manifest.get("version") != SOURCE_VERSION:
        raise RuntimeError("wrong physician-OE delivery version")
    if delivery_manifest.get("bundle_id") != metadata.get("bundle_id"):
        raise RuntimeError("delivery and metadata bundle IDs differ")
    if metadata.get("bundle_sha256") != delivery_manifest.get("source_template_sha256"):
        raise RuntimeError("delivery source template is not the frozen metadata bundle")
    if set(delivery_manifest.get("reviewers", {})) != {"A", "B"}:
        raise RuntimeError("expected exactly reviewer roles A and B")
    if delivery_manifest.get("private_mapping_in_delivery") is not False:
        raise RuntimeError("source delivery is not certified blinded")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        build_role_archive(
            delivery_dir=delivery_dir,
            metadata=metadata,
            delivery_manifest=delivery_manifest,
            runbook=runbook,
            output_dir=output_dir,
            role=role,
        )
        for role in ("A", "B")
    ]
    index = {
        "version": VERSION,
        "source_delivery_version": SOURCE_VERSION,
        "bundle_id": metadata["bundle_id"],
        "archives": records,
    }
    index_path = output_dir / "delivery_index.json"
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json(index))
    os.replace(temporary, index_path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package_deliveries(
                delivery_dir=args.delivery_dir,
                metadata_path=args.metadata,
                runbook=args.runbook,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
