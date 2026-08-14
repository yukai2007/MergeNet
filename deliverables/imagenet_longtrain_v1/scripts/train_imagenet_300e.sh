#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"

info() { printf '[launcher] %s\n' "$*"; }
warn() { printf '[launcher][WARN] %s\n' "$*" >&2; }
die() { printf '[launcher][FATAL] %s\n' "$*" >&2; exit 2; }
is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

GPU_VISIBILITY_HELPER="${SCRIPT_DIR}/gpu_visibility.sh"
[[ -r "${GPU_VISIBILITY_HELPER}" ]] || die "GPU visibility helper is missing: ${GPU_VISIBILITY_HELPER}"
# shellcheck source=gpu_visibility.sh
source "${GPU_VISIBILITY_HELPER}"

CONFIG="${1:-${ROOT_DIR}/configs/mergenet_lambda4.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")
if [[ ! -f "${CONFIG}" && -f "${ROOT_DIR}/${CONFIG}" ]]; then
  CONFIG="${ROOT_DIR}/${CONFIG}"
fi
[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"

# Extra CLI options are applied after YAML, so an open-ended pass-through can
# invalidate preflight without changing the checked config (for example
# --epochs, --initial_checkpoint, or --lambda_local). Keep the supported
# override surface intentionally tiny: LR scaling is documented for a changed
# GLOBAL_BATCH, and --prefetcher is an explicitly opt-in runtime path. All
# identity, path, resume, epoch, batch, dataset, output, and architecture state
# must come from the audited YAML or dedicated launcher variables.
extra_index=0
while (( extra_index < ${#EXTRA_ARGS[@]} )); do
  arg="${EXTRA_ARGS[extra_index]}"
  case "${arg}" in
    --lr)
      (( extra_index + 1 < ${#EXTRA_ARGS[@]} )) \
        || die '--lr requires a numeric value'
      lr_override="${EXTRA_ARGS[extra_index + 1]}"
      is_nonnegative_number "${lr_override}" \
        || die "--lr requires a non-negative numeric value; got ${lr_override}"
      extra_index=$((extra_index + 2))
      ;;
    --lr=*)
      lr_override="${arg#--lr=}"
      is_nonnegative_number "${lr_override}" \
        || die "--lr requires a non-negative numeric value; got ${lr_override}"
      extra_index=$((extra_index + 1))
      ;;
    --prefetcher)
      extra_index=$((extra_index + 1))
      ;;
    *)
      die "unsupported extra argument: ${arg}; only --lr and --prefetcher may override the audited config"
      ;;
  esac
done

[[ -n "${DATA_DIR:-}" ]] || die 'DATA_DIR is required'
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
variant=$(basename "${CONFIG}")
variant=${variant%.yaml}
variant=${variant%.yml}
RUN_NAME="${RUN_NAME:-in1k300_${variant}_seed42}"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die 'RUN_NAME must start with a letter or digit and contain only letters, digits, dot, underscore, and hyphen'

configure_gpu_visibility || exit $?

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python executable not found: ${PYTHON_BIN}"
command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 || die "torchrun executable not found: ${TORCHRUN_BIN}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE=$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
  )
fi
is_positive_int "${NPROC_PER_NODE}" || die "NPROC_PER_NODE must be a positive integer; got ${NPROC_PER_NODE}"

GLOBAL_BATCH="${GLOBAL_BATCH:-1024}"
MAX_MICRO_BATCH="${MAX_MICRO_BATCH:-64}"
is_positive_int "${GLOBAL_BATCH}" || die "GLOBAL_BATCH must be a positive integer; got ${GLOBAL_BATCH}"
is_positive_int "${MAX_MICRO_BATCH}" || die "MAX_MICRO_BATCH must be a positive integer; got ${MAX_MICRO_BATCH}"

if [[ -n "${BATCH_SIZE:-}" ]]; then
  is_positive_int "${BATCH_SIZE}" || die "BATCH_SIZE must be a positive integer; got ${BATCH_SIZE}"
fi
if [[ -n "${UPDATE_FREQ:-}" ]]; then
  is_positive_int "${UPDATE_FREQ}" || die "UPDATE_FREQ must be a positive integer; got ${UPDATE_FREQ}"
fi

if [[ -n "${BATCH_SIZE:-}" && -n "${UPDATE_FREQ:-}" ]]; then
  effective_batch=$((BATCH_SIZE * NPROC_PER_NODE * UPDATE_FREQ))
  [[ "${effective_batch}" -eq "${GLOBAL_BATCH}" ]] \
    || die "BATCH_SIZE*NPROC_PER_NODE*UPDATE_FREQ=${effective_batch}, expected GLOBAL_BATCH=${GLOBAL_BATCH}"
elif [[ -n "${BATCH_SIZE:-}" ]]; then
  denominator=$((BATCH_SIZE * NPROC_PER_NODE))
  (( GLOBAL_BATCH % denominator == 0 )) \
    || die "GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by BATCH_SIZE*NPROC_PER_NODE=${denominator}"
  UPDATE_FREQ=$((GLOBAL_BATCH / denominator))
elif [[ -n "${UPDATE_FREQ:-}" ]]; then
  denominator=$((NPROC_PER_NODE * UPDATE_FREQ))
  (( GLOBAL_BATCH % denominator == 0 )) \
    || die "GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by NPROC_PER_NODE*UPDATE_FREQ=${denominator}"
  BATCH_SIZE=$((GLOBAL_BATCH / denominator))
else
  (( GLOBAL_BATCH % NPROC_PER_NODE == 0 )) \
    || die "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}; set GLOBAL_BATCH explicitly"
  per_worker_effective=$((GLOBAL_BATCH / NPROC_PER_NODE))
  UPDATE_FREQ=$(((per_worker_effective + MAX_MICRO_BATCH - 1) / MAX_MICRO_BATCH))
  while (( UPDATE_FREQ <= per_worker_effective && per_worker_effective % UPDATE_FREQ != 0 )); do
    UPDATE_FREQ=$((UPDATE_FREQ + 1))
  done
  (( UPDATE_FREQ <= per_worker_effective )) || die 'could not derive an exact accumulation schedule'
  BATCH_SIZE=$((per_worker_effective / UPDATE_FREQ))
fi

effective_batch=$((BATCH_SIZE * NPROC_PER_NODE * UPDATE_FREQ))
[[ "${effective_batch}" -eq "${GLOBAL_BATCH}" ]] || die 'internal effective-batch validation failed'
if (( BATCH_SIZE > MAX_MICRO_BATCH )); then
  warn "micro-batch ${BATCH_SIZE} exceeds MAX_MICRO_BATCH=${MAX_MICRO_BATCH}; verify memory before training"
fi

if [[ "${GLOBAL_BATCH}" -ne 1024 ]]; then
  warn 'YAML lr=5e-4 is calibrated for global batch 1024; pass an intentional --lr override'
fi

RUN_DIR="${OUTPUT_DIR%/}/${RUN_NAME}"
RESUME="${RESUME:-auto}"
RESUME_ARGS=()

is_checkpoint_or_model_artifact() {
  local name="${1,,}"
  case "${name}" in
    *.pth*|*.pt|*.pt.*|*.ckpt*|*.safetensors*|checkpoint-*|model_best*|last*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

guard_fresh_run_dir() {
  local launch_mode="$1"
  local entry name
  local has_entries=0

  if [[ -e "${RUN_DIR}" || -L "${RUN_DIR}" ]]; then
    [[ -d "${RUN_DIR}" ]] || die "run path exists but is not a directory: ${RUN_DIR}"
  else
    return 0
  fi

  while IFS= read -r -d '' entry; do
    has_entries=1
    name=${entry##*/}
    if is_checkpoint_or_model_artifact "${name}"; then
      die "refusing ${launch_mode} run: checkpoint/model artifact exists in ${RUN_DIR}: ${name}"
    fi
    if [[ -L "${entry}" || ! -f "${entry}" || "${name}" != launcher_*.log ]]; then
      die "refusing ${launch_mode} run: only top-level regular launcher_*.log files may be reused; found ${name}"
    fi
  done < <(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print0)

  if (( has_entries )); then
    [[ "${ALLOW_EXISTING_RUN_DIR:-0}" == "1" ]] \
      || die "refusing ${launch_mode} run in non-empty directory: ${RUN_DIR}; only log-only directories may be reused with ALLOW_EXISTING_RUN_DIR=1"
    info "ALLOW_EXISTING_RUN_DIR=1 accepted existing launcher logs in ${RUN_DIR}"
  fi
}

case "${RESUME}" in
  auto)
    if [[ -f "${RUN_DIR}/last.pth.tar" ]]; then
      RESUME_ARGS=(--resume "${RUN_DIR}/last.pth.tar")
      info "auto-resume selected: ${RUN_DIR}/last.pth.tar"
    else
      guard_fresh_run_dir 'auto-without-last'
    fi
    ;;
  none|'')
    guard_fresh_run_dir 'scratch'
    ;;
  *)
    [[ -f "${RESUME}" ]] || die "explicit RESUME checkpoint not found: ${RESUME}"
    RESUME_ARGS=(--resume "${RESUME}")
    ;;
esac

mkdir -p "${RUN_DIR}"
[[ -w "${RUN_DIR}" ]] || die "run directory is not writable: ${RUN_DIR}"

export OPENTOME_SKIP_OPTIONAL_NLP=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED=1

if [[ "${RUN_PREFLIGHT:-1}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
    PYTHON_BIN="${PYTHON_BIN}" TORCHRUN_BIN="${TORCHRUN_BIN}" \
    bash "${SCRIPT_DIR}/preflight_imagenet.sh" "${CONFIG}"
else
  warn 'preflight disabled explicitly'
fi

CMD=(
  "${TORCHRUN_BIN}"
  --nnodes 1
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR:-127.0.0.1}"
  --master_port "${MASTER_PORT:-29500}"
  "${ROOT_DIR}/trainer/classification/in1k_trainer.py"
  --config "${CONFIG}"
  --data_dir "${DATA_DIR}"
  --train_split train
  --val_split val
  --batch_size "${BATCH_SIZE}"
  --update_freq "${UPDATE_FREQ}"
  --output "${OUTPUT_DIR}"
  --experiment "${RUN_NAME}"
  "${RESUME_ARGS[@]}"
)
if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
  is_positive_int "${VAL_BATCH_SIZE}" || die "VAL_BATCH_SIZE must be a positive integer; got ${VAL_BATCH_SIZE}"
  CMD+=(--validation-batch-size "${VAL_BATCH_SIZE}")
fi
CMD+=("${EXTRA_ARGS[@]}")

info "config=${CONFIG}"
info "run_dir=${RUN_DIR}"
info "workers=${NPROC_PER_NODE}, batch/GPU=${BATCH_SIZE}, update_freq=${UPDATE_FREQ}, effective_global_batch=${effective_batch}"
printf '[launcher] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" || "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  info 'not starting training (DRY_RUN or PREFLIGHT_ONLY requested)'
  exit 0
fi

LOG_FILE="${RUN_DIR}/launcher_$(date +%Y%m%d_%H%M%S).log"
info "streaming log to ${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
