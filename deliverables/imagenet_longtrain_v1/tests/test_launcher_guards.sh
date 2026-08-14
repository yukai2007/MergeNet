#!/usr/bin/env bash
# Launcher-only regression test. It never imports torch or starts torchrun.
set -Eeuo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${TEST_DIR}/.." && pwd)
LAUNCHER="${ROOT_DIR}/scripts/train_imagenet_300e.sh"
PRETRAIN_LAUNCHER="${ROOT_DIR}/scripts/pretrain_imagenet_300e.sh"
PREFLIGHT="${ROOT_DIR}/scripts/preflight_imagenet.sh"
GPU_VISIBILITY_HELPER="${ROOT_DIR}/scripts/gpu_visibility.sh"
CONFIG="${ROOT_DIR}/configs/mergenet_lambda4.yaml"
TMP_BASE="${TMPDIR:-/tmp}"
TEST_TMP=$(mktemp -d "${TMP_BASE%/}/mergenet-launcher-guards.XXXXXX")

cleanup() {
  if [[ -n "${TEST_TMP:-}" && -d "${TEST_TMP}" && "${TEST_TMP}" == "${TMP_BASE%/}"/mergenet-launcher-guards.* ]]; then
    rm -rf -- "${TEST_TMP}"
  fi
}
trap cleanup EXIT

fail() {
  printf '[launcher-guard-test][FAIL] %s\n' "$*" >&2
  exit 1
}

invoke_launcher_with_state() {
  local run_name="$1"
  local resume_mode="$2"
  local allow_existing="$3"
  shift 3
  DATA_DIR="${TEST_TMP}/fake-imagenet" \
  OUTPUT_DIR="${TEST_TMP}/outputs" \
  RUN_NAME="${run_name}" \
  GPUS= \
  NPROC_PER_NODE=2 \
  GLOBAL_BATCH=1024 \
  MAX_MICRO_BATCH=64 \
  BATCH_SIZE= \
  UPDATE_FREQ= \
  VAL_BATCH_SIZE= \
  RUN_PREFLIGHT=0 \
  DRY_RUN=1 \
  RESUME="${resume_mode}" \
  ALLOW_EXISTING_RUN_DIR="${allow_existing}" \
  PYTHON_BIN=true \
  TORCHRUN_BIN=true \
    bash "${LAUNCHER}" "${CONFIG}" "$@"
}

invoke_launcher() {
  invoke_launcher_with_state launcher_guard_test none 0 "$@"
}

expect_run_name_rejected() {
  local run_name="$1"
  local output
  if output=$(invoke_launcher_with_state "${run_name}" none 0 2>&1); then
    fail "unsafe RUN_NAME was accepted: ${run_name}"
  fi
  grep -q 'RUN_NAME must start with a letter or digit' <<< "${output}" \
    || fail "unsafe RUN_NAME failed for the wrong reason (${run_name}): ${output}"
}

expect_run_name_rejected '.'
expect_run_name_rejected '..'
expect_run_name_rejected '-outside'

invoke_pretrain_launcher() {
  DATA_DIR="${TEST_TMP}/fake-imagenet" \
  OUTPUT_DIR="${TEST_TMP}/outputs" \
  RUN_NAME=paper_pretrain_guard_test \
  GPUS= \
  NPROC_PER_NODE=2 \
  GLOBAL_BATCH=1024 \
  MAX_MICRO_BATCH=64 \
  BATCH_SIZE= \
  UPDATE_FREQ= \
  VAL_BATCH_SIZE= \
  RUN_PREFLIGHT=0 \
  DRY_RUN=1 \
  RESUME=none \
  ALLOW_EXISTING_RUN_DIR=0 \
  PYTHON_BIN=true \
  TORCHRUN_BIN=true \
    bash "${PRETRAIN_LAUNCHER}" "$@"
}

pretrain_output=$(invoke_pretrain_launcher --lr 2.5e-4 --prefetcher 2>&1) \
  || fail 'canonical paper-scale pretraining wrapper failed its dry run'
grep -Fq "[launcher] config=${CONFIG}" <<< "${pretrain_output}" \
  || fail 'pretraining wrapper did not select the canonical lambda4 protocol'
grep -Fq 'workers=2, batch/GPU=64, update_freq=8, effective_global_batch=1024' \
  <<< "${pretrain_output}" \
  || fail 'pretraining wrapper did not preserve the paper-scale global-batch protocol'
grep -q -- '--lr 2.5e-4 --prefetcher' <<< "${pretrain_output}" \
  || fail 'pretraining wrapper did not forward documented launcher overrides'

pretrain_override_output=''
if pretrain_override_output=$(invoke_pretrain_launcher --epochs 1 2>&1); then
  fail 'pretraining wrapper accepted an epoch override'
fi
grep -q 'only --lr and --prefetcher may override' <<< "${pretrain_override_output}" \
  || fail "pretraining wrapper epoch override failed for the wrong reason: ${pretrain_override_output}"

allowed_output=$(invoke_launcher --lr 2.5e-4 --prefetcher 2>&1) \
  || fail 'documented --lr/--prefetcher override was rejected'
grep -q -- '--lr 2.5e-4 --prefetcher' <<< "${allowed_output}" \
  || fail 'allowed overrides were not preserved in the dry-run command'

equals_output=$(invoke_launcher --lr=2.5e-4 2>&1) \
  || fail 'documented --lr=value override was rejected'
grep -q -- '--lr=2.5e-4' <<< "${equals_output}" \
  || fail '--lr=value was not preserved in the dry-run command'

mapped_gpus=$(GPUS=2,5 CUDA_VISIBLE_DEVICES=9 bash -c \
  'source "$1"; configure_gpu_visibility; printf "%s" "${CUDA_VISIBLE_DEVICES}"' \
  _ "${GPU_VISIBILITY_HELPER}") \
  || fail 'valid GPUS mapping was rejected'
[[ "${mapped_gpus}" == '2,5' ]] || fail "GPUS did not map to CUDA_VISIBLE_DEVICES: ${mapped_gpus}"

preserved_visibility=$(GPUS= CUDA_VISIBLE_DEVICES=7 bash -c \
  'source "$1"; configure_gpu_visibility; printf "%s" "${CUDA_VISIBLE_DEVICES}"' \
  _ "${GPU_VISIBILITY_HELPER}") \
  || fail 'empty GPUS unexpectedly failed'
[[ "${preserved_visibility}" == '7' ]] \
  || fail 'empty GPUS should preserve an existing CUDA_VISIBLE_DEVICES value'

expect_gpu_spec_rejected() {
  local spec="$1"
  local output
  if output=$(GPUS="${spec}" bash -c \
    'source "$1"; configure_gpu_visibility' _ "${GPU_VISIBILITY_HELPER}" 2>&1); then
    fail "invalid GPUS specification was accepted: ${spec}"
  fi
  grep -q '\[gpu-visibility\]\[FATAL\]' <<< "${output}" \
    || fail "invalid GPUS specification failed for the wrong reason (${spec}): ${output}"
}

expect_gpu_spec_rejected '0, 1'
expect_gpu_spec_rejected '0,a'
expect_gpu_spec_rejected '00,1'
expect_gpu_spec_rejected '0,1,0'

preflight_gpu_output=''
if preflight_gpu_output=$(DATA_DIR= GPUS=0,0 PYTHON_BIN=true TORCHRUN_BIN=true \
  bash "${PREFLIGHT}" "${CONFIG}" 2>&1); then
  fail 'standalone preflight accepted duplicate GPUS'
fi
grep -q 'duplicate GPU id in GPUS: 0' <<< "${preflight_gpu_output}" \
  || fail "standalone preflight did not apply the shared GPUS guard: ${preflight_gpu_output}"

expect_rejected() {
  local label="$1"
  shift
  local output
  if output=$(invoke_launcher "$@" 2>&1); then
    fail "protected override was accepted (${label}): $*"
  fi
  grep -q 'only --lr and --prefetcher may override' <<< "${output}" \
    || fail "protected override failed for the wrong reason (${label}): ${output}"
}

# Protocol identity and architecture.
expect_rejected model --model deit_tiny_patch16_224
expect_rejected dataset --dataset cifar100
expect_rejected classes --num_classes 100
expect_rejected image_size --img_size 32
expect_rejected patch_size --patch_size 16
expect_rejected compression --lambda_local 2

# Epoch, batch, and split state.
expect_rejected epochs --epochs 1
expect_rejected start_epoch --start_epoch 10
expect_rejected batch --batch_size 1
expect_rejected accumulation --update_freq 1
expect_rejected train_split --train_split alternate_train
expect_rejected val_split --val_split alternate_val

# Paths, run identity, initialization, and resume state.
expect_rejected config --config alternate.yaml
expect_rejected data_path --data_dir /alternate/data
expect_rejected output_path --output /alternate/output
expect_rejected experiment --experiment alternate_run
expect_rejected resume --resume=/alternate/last.pth.tar
expect_rejected initial_checkpoint --initial_checkpoint /alternate/init.pth.tar
expect_rejected distill_checkpoint --distill_teacher_checkpoint /alternate/teacher.pth.tar
expect_rejected branch_checkpoint --branch_a_checkpoint /alternate/branch_a.pth.tar

# Unknown/operational pass-through is also closed; change YAML instead.
expect_rejected unknown --workers 2

malformed_lr_output=''
if malformed_lr_output=$(invoke_launcher --lr not-a-number 2>&1); then
  fail 'malformed --lr value was accepted'
fi
grep -q -- '--lr requires a non-negative numeric value' <<< "${malformed_lr_output}" \
  || fail "malformed --lr failed for the wrong reason: ${malformed_lr_output}"

# ALLOW_EXISTING_RUN_DIR is a narrow exception for launcher logs, never for
# checkpoints, model artifacts, arbitrary files, symlinks, or directories.
mkdir -p "${TEST_TMP}/outputs/log_only_scratch"
touch "${TEST_TMP}/outputs/log_only_scratch/launcher_20260809_120000.log"
invoke_launcher_with_state log_only_scratch none 1 >/dev/null 2>&1 \
  || fail 'ALLOW_EXISTING_RUN_DIR=1 rejected a scratch directory containing only launcher logs'

mkdir -p "${TEST_TMP}/outputs/log_only_auto"
touch "${TEST_TMP}/outputs/log_only_auto/launcher_20260809_120001.log"
invoke_launcher_with_state log_only_auto auto 1 >/dev/null 2>&1 \
  || fail 'ALLOW_EXISTING_RUN_DIR=1 rejected an auto-without-last directory containing only launcher logs'

mkdir -p "${TEST_TMP}/outputs/log_without_allow"
touch "${TEST_TMP}/outputs/log_without_allow/launcher_20260809_120002.log"
log_without_allow_output=''
if log_without_allow_output=$(invoke_launcher_with_state log_without_allow none 0 2>&1); then
  fail 'a non-empty log-only directory was accepted without ALLOW_EXISTING_RUN_DIR=1'
fi
grep -q 'only log-only directories may be reused with ALLOW_EXISTING_RUN_DIR=1' \
  <<< "${log_without_allow_output}" \
  || fail "log-only directory failed for the wrong reason: ${log_without_allow_output}"

mkdir -p "${TEST_TMP}/outputs/arbitrary_file"
touch "${TEST_TMP}/outputs/arbitrary_file/notes.txt"
arbitrary_output=''
if arbitrary_output=$(invoke_launcher_with_state arbitrary_file none 1 2>&1); then
  fail 'ALLOW_EXISTING_RUN_DIR=1 accepted a non-launcher-log file'
fi
grep -q 'only top-level regular launcher_\*.log files may be reused' <<< "${arbitrary_output}" \
  || fail "arbitrary file failed for the wrong reason: ${arbitrary_output}"

artifact_names=(
  weights.pth
  last.pth.tar
  weights.pt
  epoch.ckpt
  model.safetensors
  checkpoint-12
  model_best.backup
  launcher_weights.pth.log
)
artifact_index=0
for artifact_name in "${artifact_names[@]}"; do
  artifact_run="scratch_artifact_${artifact_index}"
  mkdir -p "${TEST_TMP}/outputs/${artifact_run}"
  touch "${TEST_TMP}/outputs/${artifact_run}/${artifact_name}"
  artifact_output=''
  if artifact_output=$(invoke_launcher_with_state "${artifact_run}" none 1 2>&1); then
    fail "ALLOW_EXISTING_RUN_DIR=1 accepted checkpoint/model artifact: ${artifact_name}"
  fi
  grep -q 'checkpoint/model artifact exists' <<< "${artifact_output}" \
    || fail "artifact failed for the wrong reason (${artifact_name}): ${artifact_output}"
  artifact_index=$((artifact_index + 1))
done

mkdir -p "${TEST_TMP}/outputs/auto_artifact_without_last"
touch "${TEST_TMP}/outputs/auto_artifact_without_last/checkpoint-99"
auto_artifact_output=''
if auto_artifact_output=$(invoke_launcher_with_state auto_artifact_without_last auto 1 2>&1); then
  fail 'auto-without-last accepted a checkpoint artifact with ALLOW_EXISTING_RUN_DIR=1'
fi
grep -q 'checkpoint/model artifact exists' <<< "${auto_artifact_output}" \
  || fail "auto artifact failed for the wrong reason: ${auto_artifact_output}"

printf 'LAUNCHER_GUARD_TEST_PASS\n'
