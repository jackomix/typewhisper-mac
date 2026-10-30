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

touch "${DESTINATION}/model-bundle.stamp"
echo "Spokenly Cleanup Core ML model bundled in ${DESTINATION}"
