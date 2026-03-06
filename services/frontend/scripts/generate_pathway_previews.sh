#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FRONTEND_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
OUTPUT_DIR="${FRONTEND_DIR}/public/pathway-previews"

mkdir -p "${OUTPUT_DIR}"

PATHWAY_SOURCES=(
  "asthma:asthma_emergency_department_algorithm_-_9.8.23.pdf"
  "anaphylaxis:anaphylaxis_-_1.16.25.pdf"
  "bronchiolitis:bronchiolitis-educational-module-10.13.23.pdf"
  "croup:croup-module-3.25.25.pdf"
  "appendicitis:appendicitis-6.30.23.pdf"
)

for pathway in "${PATHWAY_SOURCES[@]}"; do
  IFS=":" read -r pathway_id pdf_name <<< "${pathway}"
  input_pdf="${DATA_DIR}/${pdf_name}"
  output_png="${OUTPUT_DIR}/${pathway_id}.png"

  if [[ ! -f "${input_pdf}" ]]; then
    echo "Missing source PDF: ${input_pdf}" >&2
    exit 1
  fi

  sips -s format png "${input_pdf}" --out "${output_png}" >/dev/null
  echo "Generated ${output_png}"
done
