#!/usr/bin/env bash
# ------------------------------------------------------------------
# Download raw images for the MedHEval benchmark.
#
#   benchmark_data/   <- annotations only (already in repo)
#   images/           <- created by this script; raw images go here
#
# Image sources (verified 2026-07-02):
#   - Slake      Google Drive file 1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U
#                   (linked from https://www.med-vqa.com/slake/)
#   - VQA-RAD    OSF node 89kps, folder "VQA_RAD Image Folder" (315 jpgs),
#                   per-file download via https://osf.io/download/<guid>/
#   - IU-Xray    Google Drive file 1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg
#                   (from R2GenGPT README; linked in MedHEval readme.md)
#   - MIMIC-CXR  https://physionet.org/content/mimic-cxr-jpg/2.0.0/
#                   -> credentialed; s3://physionet-data/mimic-cxr-jpg/2.0.0/
#                   (NOT physionet-open / NOT --no-sign-request)
#                 -> requires PhysioNet credentialing (CITI training +
#                    signed DUA). Cannot be downloaded unauthenticated.
#
# Usage:
#   ./download_images.sh            # download all public sources
#   ./download_images.sh slake      # only one source
#   ./download_images.sh iu_xray vqarad
#
# Valid source names: slake vqarad iu_xray mimic_cxr
# ------------------------------------------------------------------
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMG_ROOT="images"          # all raw images live under here
mkdir -p "$IMG_ROOT"

# Color log helpers.
log()  { printf '\033[1;34m[download]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Source: Slake
#   img_name pattern: xmlab###/source.jpg  (under Slake's images/ dir)
#   Used by: Visual_Misinterpretation/.../fine-grained/slake_qa_pairs.json
#            Visual_Misinterpretation/.../close-ended/MM-VisHal.json
#   The download button on https://www.med-vqa.com/slake/ points to a Google
#   Drive file (id 1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U). We fetch via gdown.
# ---------------------------------------------------------------------------
download_slake() {
  local dest="$IMG_ROOT/Slake"
  if [[ -d "$dest" && $(find "$dest" -type f | head -1 | wc -l) -gt 0 ]]; then
    log "Slake: already present at $dest, skipping."
    return
  fi
  log "Slake: downloading from Google Drive ..."
  mkdir -p "$dest"

  local file_id="1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U"
  local tmp; tmp="$(mktemp -d)"

  if ! have gdown; then
    warn "gdown not found; installing: pip install gdown"
    pip install gdown >/dev/null 2>&1 || true
  fi
  if ! have gdown; then
    err "gdown unavailable. Install with: pip install gdown"
    rm -rf "$tmp"; return 1
  fi

  gdown "$file_id" -O "$tmp/Slake.zip" || { err "Slake: gdown failed."; rm -rf "$tmp"; return 1; }
  if have unzip; then
    unzip -q -o "$tmp/Slake.zip" -d "$tmp/extracted"
  else
    err "unzip not found."; rm -rf "$tmp"; return 1
  fi

  # Slake's zip lays out as Slake/images/xmlab###/source.jpg (and labels/).
  # Move the images dir into place so that $dest/xmlab###/source.jpg resolves.
  local found
  found="$(find "$tmp/extracted" -type d -name images | head -1)"
  if [[ -z "$found" ]]; then
    # Fall back: if the zip already contains xmlab### dirs at top level, use them directly.
    if find "$tmp/extracted" -maxdepth 2 -type d -name 'xmlab*' | head -1 | grep -q .; then
      found="$(dirname "$(find "$tmp/extracted" -maxdepth 2 -type d -name 'xmlab*' | head -1)")"
    else
      err "Slake: could not find 'images/' or 'xmlab###' inside the zip. Inspect $tmp/extracted."
      return 1
    fi
  fi
  cp -rn "$found/." "$dest/"   # merge so xmlab### lands at $dest/xmlab###
  rm -rf "$tmp"
  log "Slake: done -> $dest ($(find "$dest" -type f | wc -l) files)"
}

# ---------------------------------------------------------------------------
# Source: VQA-RAD
#   img_name pattern: synpic#####.jpg
#   Used by: Visual_Misinterpretation/.../fine-grained/rad_vqa_pairs.json
#   OSF node 89kps stores the 315 images as individual files inside the
#   folder "VQA_RAD Image Folder" (id 5b21453986d8510011c277bc). OSF has no
#   reliable anonymous whole-folder zip, so we page the JSON:API (10 per page,
#   32 pages) and download each file via its https://osf.io/download/<guid>/
#   link. Verified 2026-07-02: API reachable, per-file downloads valid JPGs.
#
#   CAVEAT (verified 2026-07-02): the OSF image folder contains 315 jpgs but
#   only 300 of the 314 images referenced by rad_vqa_pairs.json are present;
#   14 referenced synpic#####.jpg files are NOT on OSF, and 15 OSF files are
#   not referenced. This is a data-source discrepancy, not a script bug --
#   those 14 images simply cannot be obtained from the OSF mirror. Coverage
#   will therefore cap at ~95% for VQA-RAD.
# ---------------------------------------------------------------------------
VQARAD_NODE="89kps"
VQARAD_IMG_FOLDER="5b21453986d8510011c277bc"

download_vqarad() {
  local dest="$IMG_ROOT/VQA-RAD"
  # NOTE: no coarse "already present" skip here -- the per-file loop below is
  # idempotent (re-downloads only missing files), so re-running resumes.
  log "VQA-RAD: enumerating OSF image folder via JSON:API ..."
  mkdir -p "$dest"

  if ! have curl; then err "curl required for VQA-RAD download."; return 1; fi
  if ! have jq;  then err "jq required for VQA-RAD download.";   return 1; fi

  local tmp; tmp="$(mktemp -d)"
  local page=1 next="https://api.osf.io/v2/nodes/$VQARAD_NODE/files/osfstorage/$VQARAD_IMG_FOLDER/"
  # Per-file manifest: list of "<guid>\t<name>" lines.
  local manifest="$tmp/manifest.tsv"
  : > "$manifest"
  while [[ -n "$next" && "$next" != "null" ]]; do
    local body="$tmp/page_$page.json"
    curl -sL --max-time 40 -A "Mozilla/5.0" "$next" -o "$body" || { err "VQA-RAD: API page $page failed."; rm -rf "$tmp"; return 1; }
    jq -r '.data[] | select(.attributes.kind=="file") | "\(.attributes.guid)\t\(.attributes.name)"' "$body" 2>/dev/null >> "$manifest"
    next="$(jq -r '.links.next // empty' "$body" 2>/dev/null)"
    page=$((page+1))
    [[ $page -gt 50 ]] && break   # safety: there are 32 pages
  done
  local total; total=$(wc -l < "$manifest")
  log "VQA-RAD: $total files enumerated. Downloading (concurrency ${VQARAD_JOBS:-8}) ..."

  # Concurrent fetch with a bounded job pool. Skip files already on disk
  # (idempotent re-runs after interruption).
  local jobs="${VQARAD_JOBS:-8}"
  local done=0
  while IFS=$'\t' read -r guid name; do
    [[ -z "$guid" || -z "$name" ]] && continue
    if [[ -s "$dest/$name" ]]; then   # idempotent: already downloaded
      done=$((done+1)); printf '\r\033[K  %d/%d (cached)' "$done" "$total"; continue
    fi
    (
      curl -sfL --max-time 40 -A "Mozilla/5.0" "https://osf.io/download/$guid/" -o "$dest/$name" \
        || rm -f "$dest/$name"
    ) &
    # bound concurrency
    while (( $(jobs -rp | wc -l) >= jobs )); do wait -n 2>/dev/null || sleep 0.1; done
    done=$((done+1)); printf '\r\033[K  %d/%d' "$done" "$total"
  done < "$manifest"
  wait
  printf '\n'
  rm -rf "$tmp"
  local got; got=$(find "$dest" -type f -name '*.jpg' | wc -l)
  log "VQA-RAD: done -> $dest ($got/$total files)"
  [[ "$got" -lt "$total" ]] && warn "VQA-RAD: $((total-got)) files failed; re-run to retry."
}

# ---------------------------------------------------------------------------
# Source: IU-Xray  (R2GenGPT-packaged version)
#   img_name pattern: CXR####_IM-####/0.png
#   Used by: Visual_Misinterpretation/.../close-ended/CXR-VisHal.json
#            Visual_Misinterpretation/.../fine-grained/xray_closed_pairs.json
# ---------------------------------------------------------------------------
download_iu_xray() {
  local dest="$IMG_ROOT/IU-Xray"
  if [[ -d "$dest" && $(find "$dest" -type f | head -1 | wc -l) -gt 0 ]]; then
    log "IU-Xray: already present at $dest, skipping."
    return
  fi
  log "IU-Xray: downloading from Google Drive (R2GenGPT) ..."
  mkdir -p "$dest"

  local file_id="1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg"
  local tmp; tmp="$(mktemp -d)"

  if ! have gdown; then
    warn "gdown not found; installing: pip install gdown"
    pip install gdown >/dev/null 2>&1 || true
  fi
  if ! have gdown; then
    err "gdown unavailable. Install with: pip install gdown"
    rm -rf "$tmp"; return 1
  fi

  gdown "$file_id" -O "$tmp/iu_xray.zip" || { err "IU-Xray: gdown failed."; rm -rf "$tmp"; return 1; }
  file "$tmp/iu_xray.zip" | grep -qi zip && {
    if have unzip; then unzip -q -o "$tmp/iu_xray.zip" -d "$dest"; else err "unzip not found."; rm -rf "$tmp"; return 1; fi
  } || cp -r "$tmp/iu_xray.zip" "$dest/"

  rm -rf "$tmp"
  log "IU-Xray: done -> $dest ($(find "$dest" -type f | wc -l) files)"
}

# ---------------------------------------------------------------------------
# Source: MIMIC-CXR-JPG  (PhysioNet, credentialed access only)
#   img_name pattern: p##/p######/s#######/<32-hex>.jpg
#   Used by every benchmark_data/**/MIMIC-CXR_*.json file.
#
# Credentialed bucket is s3://physionet-data/mimic-cxr-jpg/2.0.0/  (NOT the
# open bucket; verified 2026-07-02: physionet-data returns 301 = exists but
# requires signing, physionet-open returns 404 for the same key). Downloads
# must be SIGNED (--no-sign-request gets nothing here). You must be a
# credentialed PhysioNet user and run `aws configure` with the AWS keys
# PhysioNet issues under the dataset's "AWS S3" panel.
# ---------------------------------------------------------------------------
download_mimic_cxr() {
  local dest="$IMG_ROOT/MIMIC-CXR"
  mkdir -p "$dest"

  # 1) If user pointed us at an existing extract, just symlink it in.
  if [[ -n "${MIMIC_CXR_JPG_DIR:-}" && -d "$MIMIC_CXR_JPG_DIR" ]]; then
    log "MIMIC-CXR: using existing extract at $MIMIC_CXR_JPG_DIR"
    ln -sfn "$MIMIC_CXR_JPG_DIR" "$dest/files" 2>/dev/null || ln -sfn "$MIMIC_CXR_JPG_DIR" "$dest"
    return
  fi

  if [[ -d "$dest" && $(find "$dest" -type f 2>/dev/null | head -1 | wc -l) -gt 0 ]]; then
    log "MIMIC-CXR: already present at $dest, skipping."
    return
  fi

  warn "MIMIC-CXR-JPG requires PhysioNet credentialing (CITI training + signed DUA)."
  warn "Apply at: https://physionet.org/content/mimic-cxr-jpg/2.0.0/"
  warn "After credentialing, open that page -> 'AWS S3' tab to get your access/secret keys,"
  warn "then run: aws configure  (region: us-east-1)"
  warn ""

  if have aws; then
    log "Attempting aws s3 sync (signed request, requires `aws configure` with PhysioNet-issued keys) ..."
    # Use physionet-data (credentialed). The per-dataset keys PhysioNet issues
    # are scoped to this prefix. --no-sign-request is intentionally NOT used.
    local sub="$dest/files"
    mkdir -p "$sub"
    aws s3 sync "s3://physionet-data/mimic-cxr-jpg/2.0.0/files" "$sub" \
      || { err "MIMIC-CXR: aws sync failed. Verify keys are configured and you are credentialed."; return 1; }
  else
    err "MIMIC-CXR: aws-cli not installed and \$MIMIC_CXR_JPG_DIR not set."
    err "Options:"
    err "  (a) pip install awscli, then: aws configure   # use keys from PhysioNet's 'AWS S3' panel"
    err "      aws s3 sync s3://physionet-data/mimic-cxr-jpg/2.0.0/files $dest/files"
    err "  (b) Download the zip from the PhysioNet page and extract into:"
    err "        $dest/"
    err "  (c) Set MIMIC_CXR_JPG_DIR=/path/to/already/extracted and re-run."
    return 1
  fi
  log "MIMIC-CXR: done -> $dest ($(find "$dest" -type f | wc -l) files)"
}

# ---------------------------------------------------------------------------
# Verify downloaded images against the img_name fields in benchmark_data/.
# Reports how many referenced images are present / missing per source.
# ---------------------------------------------------------------------------
verify() {
  log "Verifying benchmark_data annotations against images/ ..."
  local missing=0
  local bf="benchmark_data"
  local checks=(
    "slake:Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json:images/Slake"
    "slake:Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json:images/Slake"
    "vqarad:Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json:images/VQA-RAD"
    "iu_xray:Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json:images/IU-Xray"
    "iu_xray:Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json:images/IU-Xray"
    "mimic:Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json:images/MIMIC-CXR"
    "mimic:Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json:images/MIMIC-CXR"
    "mimic:Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json:images/MIMIC-CXR"
    "mimic:Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json:images/MIMIC-CXR"
    "mimic:Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json:images/MIMIC-CXR"
  )
  local row
  for row in "${checks[@]}"; do
    local ann="${row#*:}"
    local ann_name="${row%%:*}"
    local ann_path="$bf/${ann%:*}"
    local root="${ann##*:}"
    [[ -f "$ann_path" ]] || { warn "missing annotation: $ann_path"; continue; }
    if [[ ! -d "$root" ]]; then
      printf '  %-22s %-60s  [root missing: %s]\n' "$ann_name" "$(basename "$ann_path")" "$root"
      continue
    fi
    local total present
    total=$(jq -r '.[].img_name' "$ann_path" 2>/dev/null | sort -u | wc -l)
    if [[ "$total" -eq 0 ]]; then continue; fi
    local i=0 found=0
    while IFS= read -r img; do
      found=0
      # try under root, root/images (Slake), root/files (MIMIC-CXR aws sync layout)
      for base in "$root" "$root/images" "$root/files"; do
        if [[ -e "$base/$img" ]]; then found=1; break; fi
      done
      [[ "$found" -eq 1 ]] && i=$((i+1))
    done < <(jq -r '.[].img_name' "$ann_path" 2>/dev/null | sort -u)
    local pct=$(( i * 100 / (total>0?total:1) ))
    printf '  %-22s %-60s  %4d/%4d  (%2d%%)\n' "$ann_name" "$(basename "$ann_path")" "$i" "$total" "$pct"
    missing=$(( missing + (total - i) ))
  done
  echo ""
  log "Verification complete. Missing referenced images (total, deduplicated overlaps): $missing"
}

# ---------------------------------------------------------------------------
# CLI dispatch.
# ---------------------------------------------------------------------------
declare -A FN=( [slake]=download_slake [vqarad]=download_vqarad [iu_xray]=download_iu_xray [mimic_cxr]=download_mimic_cxr )
ALL=( slake vqarad iu_xray mimic_cxr )

targets=()
for arg in "$@"; do
  case "$arg" in
    --verify) verify; exit 0 ;;
    --help|-h)
      sed -n '1,28p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) targets+=("$arg") ;;
  esac
done
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=( "${ALL[@]}" )
fi

for t in "${targets[@]}"; do
  if [[ -z "${FN[$t]:-}" ]]; then
    err "Unknown source '$t'. Valid: ${ALL[*]}"
    continue
  fi
  "${FN[$t]}"
done

echo
log "All requested downloads attempted. Run '$0 --verify' to check coverage."