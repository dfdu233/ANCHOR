#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-tiny}"
DATASET_NAME="${2:-cxr_vishal}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
METHODS="${METHODS:-greedy VCD}"
QUESTION_TYPE="${QUESTION_TYPE:-binary}"

ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
MED="/root/autodl-tmp/MedHEval"
RUNNER="${MED}/code/baselines/Mitigation/llava-med-1.5/llava/eval/model_vqa.py"
WORKDIR="${MED}/code/baselines/Mitigation/llava-med-1.5"
MODEL_PATH="/root/autodl-tmp/LLaVA-Med/microsoft/llava-med-v1.5-mistral-7b"
IMAGE_FOLDER="${MED}/images/IU-Xray"
PYTHON="/root/autodl-tmp/envs/medheval-mitigation/bin/python"
PYTHONPATH_VALUE="${WORKDIR}:${MED}/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2/src"

case "${PHASE}" in
  tiny) MAX_SAMPLES=4 ;;
  smoke) MAX_SAMPLES=32 ;;
  pilot) MAX_SAMPLES=128 ;;
  val256) MAX_SAMPLES=256 ;;
  val512) MAX_SAMPLES=512 ;;
  *) echo "unknown phase: ${PHASE}" >&2; exit 2 ;;
esac

case "${DATASET_NAME}" in
  cxr_vishal)
    DATASET="${MED}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
    IMAGE_FOLDER="${MED}/images/IU-Xray"
    ;;
  knowledge_ce)
    DATASET="${MED}/benchmark_data/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json"
    IMAGE_FOLDER="${MED}/images"
    ;;
  *) echo "unknown dataset: ${DATASET_NAME}" >&2; exit 2 ;;
esac

OUT="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/${DATASET_NAME}"
SUBSET="${OUT}/subset_seed${SEED}_n${MAX_SAMPLES}.json"
mkdir -p "${OUT}"

"${PYTHON}" -m corrected_sgta.make_medheval_subset \
  --input "${DATASET}" \
  --output "${SUBSET}" \
  --image-folder "${IMAGE_FOLDER}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}" \
  --question-type "${QUESTION_TYPE}" \
  --require-image

for METHOD in ${METHODS}; do
  ANSWERS="${OUT}/${METHOD}_seed${SEED}_tok${MAX_NEW_TOKENS}.jsonl"
  EVAL="${OUT}/${METHOD}_seed${SEED}_tok${MAX_NEW_TOKENS}.eval.json"
  set +e
  env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PYTHONPATH_VALUE}:${ROOT}" "${PYTHON}" "${RUNNER}" \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_FOLDER}" \
    --question-file "${SUBSET}" \
    --answers-file "${ANSWERS}" \
    --conv-mode mistral_instruct \
    --temperature 0 \
    --top_p 1 \
    --num_beams 1 \
    --baseline "${METHOD}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}"
  STATUS="$?"
  set -e
  if [[ "${STATUS}" -ne 0 ]]; then
    echo "method ${METHOD} failed with status ${STATUS}" | tee "${OUT}/${METHOD}_seed${SEED}_tok${MAX_NEW_TOKENS}.failed"
    if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
      continue
    fi
    exit "${STATUS}"
  fi
  "${PYTHON}" -m corrected_sgta.evaluate_medheval_answers --answers "${ANSWERS}" --output "${EVAL}"
done

echo "results: ${OUT}"
