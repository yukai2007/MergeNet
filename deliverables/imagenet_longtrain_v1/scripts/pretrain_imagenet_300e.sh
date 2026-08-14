#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical paper-scale ImageNet-1K pretraining entrypoint. Keep the protocol
# identity fixed here and delegate all runtime validation, resume handling, and
# the narrow --lr/--prefetcher override surface to the audited launcher.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
LAUNCHER="${SCRIPT_DIR}/train_imagenet_300e.sh"
CONFIG="${ROOT_DIR}/configs/mergenet_lambda4.yaml"

die() { printf '[pretrain-launcher][FATAL] %s\n' "$*" >&2; exit 2; }

[[ -r "${LAUNCHER}" ]] || die "audited launcher is missing: ${LAUNCHER}"
[[ -f "${CONFIG}" ]] || die "canonical protocol is missing: ${CONFIG}"

exec bash "${LAUNCHER}" "${CONFIG}" "$@"
