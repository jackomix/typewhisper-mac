#!/bin/bash
set -euo pipefail

# The upstream Core ML export is ~754 MB. Keep it out of git; cache it on the
# build machine, then copy it into the distributable plugin bundle.
DESTINATION="${1:?usage: fetch_spokenly_cleanup_coreml.sh <bundle-resource-dir>}"
REPOSITORY="spokenly/cleanup-1-coreml"
REVISION="8820121fbf214445b6106332adfe13fafc55bbb6"
CACHE_ROOT="${HOME}/Library/Caches/TypeWhisper/SpokenlyCleanupCoreML/${REVISION}"
MODEL_REL="CleanupNeuralEngine.mlmodelc"
BASE_URL="https://huggingface.co/${REPOSITORY}/resolve/${REVISION}"

files=(
  "${MODEL_REL}/analytics/coremldata.bin"
  "${MODEL_REL}/coremldata.bin"
  "${MODEL_REL}/metadata.json"
  "${MODEL_REL}/model.mil"
  "${MODEL_REL}/weights/weight.bin"
  "tokenizer.json"
  "LICENSE"
  "NOTICE"
)

mkdir -p "${CACHE_ROOT}"
for path in "${files[@]}"; do
  output="${CACHE_ROOT}/${path}"
  if [[ ! -s "${output}" ]]; then
    mkdir -p "$(dirname "${output}")"
    echo "Downloading bundled Spokenly Cleanup asset: ${path}"
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 30 \
      "${BASE_URL}/${path}?download=true" \
      --output "${output}.partial"
    mv "${output}.partial" "${output}"
  fi
done

# Detect partial/corrupt large-file downloads before producing a broken bundle.
weight="${CACHE_ROOT}/${MODEL_REL}/weights/weight.bin"
if [[ ! -s "${weight}" || "$(stat -f '%z' "${weight}")" -lt 700000000 ]]; then
  echo "Spokenly Core ML weight file is incomplete." >&2
  exit 1
fi

mkdir -p "${DESTINATION}"
for path in "${files[@]}"; do
  mkdir -p "${DESTINATION}/$(dirname "${path}")"
  ditto "${CACHE_ROOT}/${path}" "${DESTINATION}/${path}"
done

# tokenizer_config.json is required by AutoTokenizer (from spokenly/cleanup-1)
tok_cfg="${CACHE_ROOT}/tokenizer_config.json"
if [[ ! -s "${tok_cfg}" ]]; then
  echo "Downloading bundled Spokenly Cleanup asset: tokenizer_config.json"
  curl --fail --location --retry 5 --retry-all-errors --connect-timeout 30 \
    "https://huggingface.co/spokenly/cleanup-1/resolve/a83ba762e3c4ab6a3a20814ec68050f381d92cf8/tokenizer_config.json?download=true" \
    --output "${tok_cfg}.partial"
  mv "${tok_cfg}.partial" "${tok_cfg}"
fi
ditto "${tok_cfg}" "${DESTINATION}/tokenizer_config.json"

touch "${DESTINATION}/model-bundle.stamp"
echo "Spokenly Cleanup Core ML model bundled in ${DESTINATION}"
