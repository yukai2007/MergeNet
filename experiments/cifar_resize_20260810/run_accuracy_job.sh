#!/usr/bin/env bash
# Run one preregistered CIFAR-100 accuracy job on one physical GPU.
set -Eeuo pipefail

die() { printf '[accuracy][FATAL] %s\n' "$*" >&2; exit 2; }
info() { printf '[accuracy] %s\n' "$*"; }

[[ "$#" -eq 4 ]] || die "usage: $0 MODEL_ID RESIZE SEED PHYSICAL_GPU"
MODEL_ID_ARG="$1"
RESIZE_ARG="$2"
SEED_ARG="$3"
GPU_ARG="$4"

case "${MODEL_ID_ARG}" in
  deit_s8|mn_l2|mn_l4) ;;
  *) die "unknown MODEL_ID: ${MODEL_ID_ARG}" ;;
esac
case "${RESIZE_ARG}" in
  160|192|224|256|320) ;;
  *) die "RESIZE is outside the locked matrix: ${RESIZE_ARG}" ;;
esac
case "${SEED_ARG}" in
  42|43|44) ;;
  *) die "SEED is outside the locked matrix: ${SEED_ARG}" ;;
esac
[[ "${GPU_ARG}" =~ ^(0|[1-9][0-9]*)$ ]] || die "PHYSICAL_GPU must be a non-negative decimal index"

for required_name in CAMPAIGN_ROOT RUNTIME_ROOT DEPS_ROOT DATA_DIR; do
  [[ -n "${!required_name:-}" ]] || die "${required_name} is required"
done
CAMPAIGN_ROOT=$(realpath -m -- "${CAMPAIGN_ROOT}")
RUNTIME_ROOT=$(realpath -e -- "${RUNTIME_ROOT}") || die "RUNTIME_ROOT does not exist"
DEPS_ROOT=$(realpath -e -- "${DEPS_ROOT}") || die "DEPS_ROOT does not exist"
DATA_DIR=$(realpath -e -- "${DATA_DIR}") || die "DATA_DIR does not exist"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROTOCOL_PATH=$(realpath -e -- "${PROTOCOL_PATH:-${SCRIPT_DIR}/protocol.json}") \
  || die "protocol.json not found"
TRAINER="${RUNTIME_ROOT}/trainer/classification/in1k_trainer.py"
[[ -f "${TRAINER}" ]] || die "delivery trainer missing: ${TRAINER}"
[[ -f "${RUNTIME_ROOT}/opentome/models/mergenet/model.py" ]] \
  || die "delivery model source missing under RUNTIME_ROOT"
[[ -d "${DEPS_ROOT}/timm" ]] || die "target dependency root does not contain timm"
[[ -f "${DATA_DIR}/cifar-100-python/train" && -f "${DATA_DIR}/cifar-100-python/test" ]] \
  || die "DATA_DIR must contain an already-downloaded cifar-100-python/{train,test}"
[[ -x /usr/bin/python ]] || die "/usr/bin/python is unavailable"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
[[ ! -v PYTHONOPTIMIZE ]] || die "PYTHONOPTIMIZE must be unset; optimized Python disables protocol assertions"
for distributed_name in WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
    MASTER_ADDR MASTER_PORT; do
  [[ ! -v "${distributed_name}" ]] || die \
    "inherited distributed variable ${distributed_name} is forbidden for direct single-GPU training"
done
while IFS='=' read -r environment_name _; do
  [[ "${environment_name}" != TORCHELASTIC_* ]] || die \
    "inherited ${environment_name} is forbidden for direct single-GPU training"
done < <(env)
for perturbation_name in CUDA_LAUNCH_BLOCKING PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF \
    PYTORCH_NO_CUDA_MEMORY_CACHING CUBLAS_WORKSPACE_CONFIG CUDA_DEVICE_MAX_CONNECTIONS \
    NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; do
  [[ ! -v "${perturbation_name}" ]] || die \
    "inherited CUDA/PyTorch perturbation ${perturbation_name} is forbidden"
done
for locked_pair in OPENTOME_MERGENET_IMPL=new TIMM_FUSED_ATTN=1; do
  locked_name=${locked_pair%%=*}
  locked_value=${locked_pair#*=}
  [[ ! -v "${locked_name}" || "${!locked_name}" == "${locked_value}" ]] || die \
    "${locked_name} must equal ${locked_value}"
  export "${locked_name}=${locked_value}"
done

DRY_RUN_VALUE="${DRY_RUN:-0}"
[[ "${DRY_RUN_VALUE}" == "0" || "${DRY_RUN_VALUE}" == "1" ]] \
  || die "DRY_RUN must be 0 or 1"
if [[ "${DRY_RUN_VALUE}" == "0" ]]; then
  # These names have historically been used to override training scripts.  The
  # preregistered queue must never inherit them silently.
  for forbidden_name in DEBUG_SUBSET EPOCHS LR MIN_LR MIN_LR_RATIO WARMUP_EPOCHS \
      GLOBAL_BATCH BATCH_SIZE UPDATE_FREQ INITIAL_CHECKPOINT DISTILL_WEIGHT \
      DTEM_TRAIN_GROUPING DTEM_EVAL_GROUPING; do
    [[ ! -v "${forbidden_name}" ]] || die \
      "formal run refuses protocol override environment variable ${forbidden_name}"
  done
fi

case "${RESIZE_ARG}" in
  160|192|224) MICRO_BATCH=200; UPDATE_FREQ_LOCKED=1 ;;
  256) MICRO_BATCH=100; UPDATE_FREQ_LOCKED=2 ;;
  320) MICRO_BATCH=50; UPDATE_FREQ_LOCKED=4 ;;
esac
[[ $((MICRO_BATCH * UPDATE_FREQ_LOCKED)) -eq 200 ]] \
  || die "internal effective-global-batch invariant failed"

# Cross-check every positional/model-specific value against the preregistration.
/usr/bin/python -S - "${PROTOCOL_PATH}" "${MODEL_ID_ARG}" "${RESIZE_ARG}" "${SEED_ARG}" \
  "${MICRO_BATCH}" "${UPDATE_FREQ_LOCKED}" <<'PY'
import json
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise SystemExit("optimized Python is forbidden for protocol validation")

path, model_id, resize, seed, micro, update = sys.argv[1:]
protocol = json.loads(Path(path).read_text(encoding="utf-8"))
models = {entry["id"]: entry for entry in protocol["models"]}
resizes = {int(entry["size"]): entry for entry in protocol["resizes"]}
assert protocol.get("expected_environment") == {
    "python": "3.10",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "timm": "0.9.11",
    "flash_attn": "2.7.4.post1",
}
assert protocol.get("expected_runtime_env") == {
    "OPENTOME_MERGENET_IMPL": "new",
    "TIMM_FUSED_ATTN": "1",
}
assert [entry["id"] for entry in protocol["models"]] == ["deit_s8", "mn_l2", "mn_l4"]
assert protocol["seeds"] == [42, 43, 44]
assert protocol["resizes"] == [
    {"size": 160, "micro_batch": 200, "update_freq": 1, "effective_global_batch": 200},
    {"size": 192, "micro_batch": 200, "update_freq": 1, "effective_global_batch": 200},
    {"size": 224, "micro_batch": 200, "update_freq": 1, "effective_global_batch": 200},
    {"size": 256, "micro_batch": 100, "update_freq": 2, "effective_global_batch": 200},
    {"size": 320, "micro_batch": 50, "update_freq": 4, "effective_global_batch": 200},
]
assert model_id in models, model_id
assert int(resize) in resizes, resize
assert int(seed) in protocol["seeds"], seed
entry = resizes[int(resize)]
assert int(entry["micro_batch"]) == int(micro), (entry, micro)
assert int(entry["update_freq"]) == int(update), (entry, update)
assert int(entry["effective_global_batch"]) == 200
training = protocol["training"]
expected_training = {
    "dataset": "CIFAR100",
    "dataset_download": False,
    "num_classes": 100,
    "initialization": "scratch",
    "knowledge_distillation": False,
    "epochs": 200,
    "target_epoch": 199,
    "global_batch": 200,
    "optimizer": "adamw",
    "optimizer_eps": 1e-8,
    "optimizer_betas": [0.9, 0.999],
    "learning_rate": 0.001,
    "weight_decay": 0.05,
    "scheduler": "cosine",
    "minimum_learning_rate": 0.0001,
    "warmup_epochs": 20,
    "warmup_learning_rate": 0.000001,
    "cooldown_epochs": 0,
    "clip_grad_norm": 1.0,
    "clip_mode": "norm",
    "drop_rate": 0.0,
    "attention_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "mixup": 0.8,
    "cutmix": 1.0,
    "mixup_mode": "batch",
    "mixup_probability": 1.0,
    "mixup_switch_probability": 0.5,
    "label_smoothing": 0.1,
    "auto_augment": "rand-m9-mstd0.5-inc1",
    "color_jitter": 0.4,
    "random_resized_crop_scale": [0.08, 1.0],
    "random_resized_crop_ratio": [0.75, 1.3333333333333333],
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.0,
    "random_erasing_probability": 0.25,
    "random_erasing_mode": "pixel",
    "random_erasing_count": 1,
    "train_interpolation": "random",
    "validation_crop_pct": 0.9,
    "amp": "fp16",
    "model_ema": True,
    "model_ema_decay": 0.9998,
    "pin_memory": True,
    "prefetcher": False,
    "workers": 8,
    "worker_seeding": "all",
    "log_interval": 50,
    "recovery_interval": 0,
    "checkpoint_history": 3,
    "eval_metric": "top1",
    "find_unused_parameters": "false",
    "primary_accuracy": "epoch_199_ema_top1",
    "appendix_accuracy": "best_ema_top1_over_epochs",
    "distillation": {
        "logit_weight": 0.0,
        "logit_temperature": 2.0,
        "routing_weight": 0.0,
        "routing_temperature": 1.0,
        "feature_weight": 0.0,
        "feature_token_weight": 0.0,
        "teacher_checkpoint": "",
    },
}
for key, expected in expected_training.items():
    actual = training.get(key)
    assert actual == expected, f"training.{key}: {actual!r} != {expected!r}"
mn = training["mergenet"]
expected_mn = {
    "train_grouping": "random_per_sample",
    "train_grouping_seed": 0,
    "eval_grouping": "alternating_per_layer_fast",
    "eval_grouping_seed": 0,
    "dtem_feat_dim": 64,
    "dtem_r": 2,
    "dtem_t": 1,
    "metric_grad_scale": 0.1,
    "source_trace_mode": "center",
    "total_merge_latent": 0,
    "use_softkmax": True,
    "swa_size": 256,
    "local_cls_global": True,
    "lambda_ramp_start_epoch": 0,
    "lambda_ramp_epochs": 50,
    "soft_topk": True,
    "soft_topk_aux_weight": 0.05,
    "soft_topk_aux_start_epoch": 20,
    "soft_topk_aux_ramp_epochs": 20,
}
assert mn == expected_mn, f"training.mergenet drift: {mn!r} != {expected_mn!r}"
expected_geometry = {
    "deit_s8": {
        "model": "deit_small_patch16_224", "patch_size": 8, "depth": 12,
        "embed_dim": 384, "num_heads": 6,
    },
    "mn_l2": {
        "model": "mergenet_small_cls", "patch_size": 8, "local_depth": 6,
        "latent_depth": 6, "lambda_start": 2.0, "lambda_local": 2.0,
        "local_block_window": 16, "dtem_window_size": 8,
    },
    "mn_l4": {
        "model": "mergenet_small_cls", "patch_size": 8, "local_depth": 4,
        "latent_depth": 8, "lambda_start": 2.0, "lambda_local": 4.0,
        "local_block_window": 32, "dtem_window_size": 8,
    },
}
for locked_model_id, geometry in expected_geometry.items():
    assert models[locked_model_id]["geometry"] == geometry, (
        locked_model_id, models[locked_model_id]["geometry"], geometry
    )
PY

GPU_UUID=$(nvidia-smi -i "${GPU_ARG}" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')
[[ "${GPU_UUID}" == GPU-* ]] || die "physical GPU ${GPU_ARG} was not found"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "${GPU_ARG}" \
      && "${CUDA_VISIBLE_DEVICES}" != "${GPU_UUID}" ]]; then
  die "refusing ambiguous inherited CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} for physical GPU ${GPU_ARG}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
export OPENTOME_SKIP_OPTIONAL_NLP=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${RUNTIME_ROOT}:${DEPS_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Verify exact import provenance and the one-card mapping before touching the run.
/usr/bin/python -S - "${RUNTIME_ROOT}" "${DEPS_ROOT}" "${GPU_UUID}" "${PROTOCOL_PATH}" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys

runtime = Path(sys.argv[1]).resolve()
deps = Path(sys.argv[2]).resolve()
expected_uuid = sys.argv[3]
protocol_path = Path(sys.argv[4])
import torch
import torchvision
import timm
import flash_attn
import opentome

def under(module, root):
    return Path(module.__file__).resolve().is_relative_to(root)

if not under(opentome, runtime):
    raise SystemExit(f"opentome import escaped RUNTIME_ROOT: {opentome.__file__}")
if not under(timm, deps):
    raise SystemExit(f"timm import escaped DEPS_ROOT: {timm.__file__}")
import json
expected = json.loads(protocol_path.read_text(encoding="utf-8"))["expected_environment"]
actual = {
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "flash_attn": md.version("flash-attn"),
}
if actual != expected:
    raise SystemExit(f"locked dependency mismatch: actual={actual}, expected={expected}")
for module in (torch, torchvision, timm, flash_attn):
    if not under(module, deps):
        raise SystemExit(f"dependency import escaped DEPS_ROOT: {module.__name__}={module.__file__}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}")
torch.cuda.set_device(0)
probe = torch.ones(16, device="cuda", dtype=torch.float16)
if float(probe.sum().item()) != 16.0:
    raise SystemExit("CUDA arithmetic smoke failed")
print(
    "[accuracy] environment OK: "
    f"python={sys.executable} torch={torch.__version__} "
    f"torchvision={torchvision.__version__} timm={timm.__version__} "
    f"flash-attn={md.version('flash-attn')} gpu={torch.cuda.get_device_name(0)} "
    f"visible_uuid={expected_uuid}"
)
PY

JOB_DIR="${CAMPAIGN_ROOT}/runs/${MODEL_ID_ARG}/r${RESIZE_ARG}/seed${SEED_ARG}"
OUTPUT_PARENT="${CAMPAIGN_ROOT}/runs/${MODEL_ID_ARG}/r${RESIZE_ARG}"
LAST_CKPT="${JOB_DIR}/last.pth.tar"
SUMMARY_CSV="${JOB_DIR}/summary.csv"
COMPLETION_JSON="${JOB_DIR}/completion.json"
MANIFEST_JSON="${JOB_DIR}/manifest.json"
LOCK_PATH="${CAMPAIGN_ROOT}/locks/${MODEL_ID_ARG}_r${RESIZE_ARG}_seed${SEED_ARG}.lock"

# Inspect without mutation. Output is one of empty, partial, complete, invalid.
inspect_job() {
  /usr/bin/python -S - "${JOB_DIR}" <<'PY'
import csv
import json
from pathlib import Path
import sys

job = Path(sys.argv[1])
if job.is_symlink():
    print("invalid:job directory must not be a symlink")
    raise SystemExit(0)
if not job.exists() or not any(job.iterdir()):
    print("empty")
    raise SystemExit(0)
symlinks = [path.relative_to(job) for path in job.rglob("*") if path.is_symlink()]
if symlinks:
    print(f"invalid:job contains symlink(s): {','.join(map(str, symlinks[:5]))}")
    raise SystemExit(0)
summary = job / "summary.csv"
last = job / "last.pth.tar"
if not last.is_file():
    paths = list(job.rglob("*"))
    allowed = []
    unexpected = []
    for path in paths:
        relative = path.relative_to(job)
        if path.is_symlink():
            unexpected.append(relative)
            continue
        if path.is_dir():
            if relative.parts == ("attempts",):
                allowed.append(relative)
            else:
                unexpected.append(relative)
            continue
        if not path.is_file():
            unexpected.append(relative)
            continue
        is_attempt_manifest = (
            len(relative.parts) == 2
            and relative.parts[0] == "attempts"
            and relative.suffix == ".json"
        )
        if str(relative) in {"manifest.json", "args.yaml"} or is_attempt_manifest:
            allowed.append(relative)
        else:
            unexpected.append(relative)
    # A wrapper manifest proves this is an interrupted attempt owned by this
    # protocol. With no metric/checkpoint artifact it is safe to restart from
    # scratch; compatibility is checked before the next attempt is recorded.
    main_manifest = job / "manifest.json"
    if main_manifest.is_file() and not main_manifest.is_symlink() and not unexpected:
        print("precheckpoint")
    else:
        detail = ",".join(str(path) for path in unexpected[:5]) or "manifest.json missing"
        print(f"invalid:non-empty job has no last.pth.tar and is not a clean precheckpoint attempt ({detail})")
    raise SystemExit(0)

import torch
try:
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(last, map_location="cpu")
if not isinstance(checkpoint, dict):
    print("invalid:last checkpoint is not a mapping")
    raise SystemExit(0)
try:
    ckpt_epoch = int(checkpoint.get("epoch"))
except (TypeError, ValueError):
    print("invalid:last checkpoint has no integer epoch")
    raise SystemExit(0)
if not any(checkpoint.get(key) is not None for key in ("state_dict_ema", "model_ema")):
    print("invalid:last checkpoint lacks EMA state")
    raise SystemExit(0)
if not summary.is_file():
    print("invalid:last checkpoint exists but summary.csv is missing")
    raise SystemExit(0)
with summary.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
try:
    epochs = [int(float(row["epoch"])) for row in rows]
except (KeyError, TypeError, ValueError):
    print("invalid:summary.csv has malformed epoch data")
    raise SystemExit(0)
if not epochs or max(epochs) != ckpt_epoch:
    print(f"invalid:summary max epoch and last checkpoint disagree ({max(epochs) if epochs else None} vs {ckpt_epoch})")
    raise SystemExit(0)
if ckpt_epoch > 199:
    print(f"invalid:last checkpoint epoch {ckpt_epoch} exceeds locked target 199")
elif ckpt_epoch == 199:
    print("complete")
else:
    print(f"partial:{ckpt_epoch}")
PY
}

if [[ "${DRY_RUN_VALUE}" == "1" ]]; then
  JOB_STATE=$(inspect_job)
else
  # The state that controls resume/skip is observed only while holding this
  # exact job lock. Command construction happens afterward, closing the race.
  mkdir -p "$(dirname "${LOCK_PATH}")"
  exec 9>"${LOCK_PATH}"
  flock -n 9 || die "job is already locked: ${LOCK_PATH}"
  JOB_STATE=$(inspect_job)
fi
case "${JOB_STATE}" in
  empty|precheckpoint) RESUME_CKPT="" ;;
  partial:*) RESUME_CKPT="${LAST_CKPT}" ;;
  complete) RESUME_CKPT="" ;;
  invalid:*) die "${JOB_STATE#invalid:}" ;;
  *) die "unexpected job inspection result: ${JOB_STATE}" ;;
esac

if [[ "${DRY_RUN_VALUE}" == "0" && "${JOB_STATE}" == "complete" ]]; then
  info "verified complete epoch-199 EMA job; leaving manifest/completion untouched"
  exit 0
fi

COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --dataset CIFAR100
  --train_split train
  --val_split val
  --num_classes 100
  --img_size "${RESIZE_ARG}"
  --patch_size 8
  --batch_size "${MICRO_BATCH}"
  --validation-batch-size "${MICRO_BATCH}"
  --update_freq "${UPDATE_FREQ_LOCKED}"
  --epochs 200
  --opt adamw
  --opt_eps 1e-8
  --opt_betas 0.9 0.999
  --lr 0.001
  --weight_decay 0.05
  --sched cosine
  --min_lr 0.0001
  --warmup_epochs 20
  --warmup_lr 0.000001
  --cooldown_epochs 0
  --clip_grad 1.0
  --clip_mode norm
  --drop_rate 0.0
  --attn_drop_rate 0.0
  --drop_path_rate 0.1
  --mixup 0.8
  --cutmix 1.0
  --mixup_prob 1.0
  --mixup_switch_prob 0.5
  --mixup_mode batch
  --smoothing 0.1
  --aa rand-m9-mstd0.5-inc1
  --color_jitter 0.4
  --scale 0.08 1.0
  --ratio 0.75 1.3333333333333333
  --hflip 0.5
  --vflip 0.0
  --reprob 0.25
  --remode pixel
  --recount 1
  --train_interpolation random
  --crop_pct 0.9
  --distill_weight 0.0
  --distill_temperature 2.0
  --distill_start_epoch 0
  --distill_ramp_epochs 0
  --distill_decay_epochs 0
  --distill_teacher_checkpoint ""
  --routing_distill_weight 0.0
  --routing_distill_temperature 1.0
  --feat_distill_weight 0.0
  --feat_distill_token_weight 0.0
  --amp
  --no_prefetcher
  --model_ema
  --model_ema_decay 0.9998
  --pin_mem
  --workers 8
  --worker_seeding all
  --log_interval 50
  --recovery_interval 0
  --checkpoint_hist 3
  --eval_metric top1
  --find_unused_parameters false
  --output "${OUTPUT_PARENT}"
  --experiment "seed${SEED_ARG}"
  --seed "${SEED_ARG}"
)

case "${MODEL_ID_ARG}" in
  deit_s8)
    MODEL_ARGS=(
      --model deit_small_patch16_224
    )
    ;;
  mn_l2)
    MODEL_ARGS=(
      --model mergenet_small_cls
      --local_depth 6
      --latent_depth 6
      --lambda_start 2.0
      --lambda_local 2.0
      --lambda_ramp_start_epoch 0
      --lambda_ramp_epochs 50
      --total_merge_latent 0
      --dtem_window_size 8
      --dtem_feat_dim 64
      --dtem_r 2
      --dtem_t 1
      --dtem_train_grouping random_per_sample
      --dtem_train_grouping_seed 0
      --dtem_eval_grouping alternating_per_layer_fast
      --dtem_eval_grouping_seed 0
      --use_softkmax
      --metric_grad_scale 0.1
      --source_trace_mode center
      --swa_size 256
      --local_block_window 16
      --local_cls_global
      --soft_topk
      --soft_topk_aux_weight 0.05
      --soft_topk_aux_start_epoch 20
      --soft_topk_aux_ramp_epochs 20
    )
    ;;
  mn_l4)
    MODEL_ARGS=(
      --model mergenet_small_cls
      --local_depth 4
      --latent_depth 8
      --lambda_start 2.0
      --lambda_local 4.0
      --lambda_ramp_start_epoch 0
      --lambda_ramp_epochs 50
      --total_merge_latent 0
      --dtem_window_size 8
      --dtem_feat_dim 64
      --dtem_r 2
      --dtem_t 1
      --dtem_train_grouping random_per_sample
      --dtem_train_grouping_seed 0
      --dtem_eval_grouping alternating_per_layer_fast
      --dtem_eval_grouping_seed 0
      --use_softkmax
      --metric_grad_scale 0.1
      --source_trace_mode center
      --swa_size 256
      --local_block_window 32
      --local_cls_global
      --soft_topk
      --soft_topk_aux_weight 0.05
      --soft_topk_aux_start_epoch 20
      --soft_topk_aux_ramp_epochs 20
    )
    ;;
esac

CMD=(/usr/bin/python -S "${TRAINER}" "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}")
if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume "${RESUME_CKPT}")
fi

emit_manifest() {
  local output_path="$1"
  /usr/bin/python -S - "${output_path}" "${PROTOCOL_PATH}" "${RUNTIME_ROOT}" "${DEPS_ROOT}" \
    "${DATA_DIR}" "${MODEL_ID_ARG}" "${RESIZE_ARG}" "${SEED_ARG}" "${GPU_ARG}" \
    "${GPU_UUID}" "${MICRO_BATCH}" "${UPDATE_FREQ_LOCKED}" "${RESUME_CKPT}" -- "${CMD[@]}" <<'PY'
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path
import socket
import sys
import tempfile

(output, protocol_path, runtime_root, deps_root, data_dir, model_id, resize,
 seed, gpu, gpu_uuid, micro, update, resume, separator, *command) = sys.argv[1:]
assert separator == "--"

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tree_hashes(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        result[str(path.relative_to(root))] = file_hash(path)
    return result

protocol_sha = file_hash(protocol_path)
runtime_files = tree_hashes(runtime_root)
runtime_sha = hashlib.sha256(
    json.dumps(runtime_files, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
command_sha = hashlib.sha256(
    json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()

import torch
import torchvision
import timm
payload = {
    "schema_version": "mergenet.cifar_resize_accuracy_manifest.v1",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "platform": platform.platform(),
    "job": {
        "model_id": model_id,
        "resize": int(resize),
        "seed": int(seed),
        "micro_batch": int(micro),
        "update_freq": int(update),
        "effective_global_batch": int(micro) * int(update),
        "target_epoch": 199,
        "primary_metric": "epoch_199_ema_top1",
        "resume": resume or None,
    },
    "physical_gpu": {"index": int(gpu), "uuid": gpu_uuid},
    "paths": {
        "protocol": str(Path(protocol_path).resolve()),
        "runtime_root": str(Path(runtime_root).resolve()),
        "deps_root": str(Path(deps_root).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
    },
    "hashes": {
        "protocol_sha256": protocol_sha,
        "runtime_tree_sha256": runtime_sha,
        "runtime_files": runtime_files,
        "command_sha256": command_sha,
    },
    "environment": {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "flash_attn": md.version("flash-attn"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "opentome_mergenet_impl": os.environ.get("OPENTOME_MERGENET_IMPL"),
        "timm_fused_attn": os.environ.get("TIMM_FUSED_ATTN"),
    },
    "command": command,
}
if output == "-":
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
else:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
PY
}

validate_existing_manifest() {
  local existing="$1"
  /usr/bin/python -S - "${existing}" "${PROTOCOL_PATH}" "${RUNTIME_ROOT}" \
    "${MODEL_ID_ARG}" "${RESIZE_ARG}" "${SEED_ARG}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path, protocol_path, runtime_root = map(Path, sys.argv[1:4])
model_id, resize, seed = sys.argv[4:]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

files = {}
for path in sorted(runtime_root.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
        continue
    files[str(path.relative_to(runtime_root))] = digest(path)
runtime_sha = hashlib.sha256(
    json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
job = manifest.get("job", {})
problems = []
if job.get("model_id") != model_id or int(job.get("resize", -1)) != int(resize) or int(job.get("seed", -1)) != int(seed):
    problems.append("job identity")
hashes = manifest.get("hashes", {})
if hashes.get("protocol_sha256") != digest(protocol_path):
    problems.append("protocol hash")
if hashes.get("runtime_tree_sha256") != runtime_sha:
    problems.append("runtime tree hash")
if problems:
    raise SystemExit("existing manifest is incompatible with resume: " + ", ".join(problems))
print(f"[accuracy] existing manifest resume guard OK: {manifest_path}")
PY
}

info "job=${MODEL_ID_ARG}/r${RESIZE_ARG}/seed${SEED_ARG} physical_gpu=${GPU_ARG} (${GPU_UUID})"
info "micro_batch=${MICRO_BATCH} update_freq=${UPDATE_FREQ_LOCKED} effective_global_batch=200 state=${JOB_STATE}"
printf '[accuracy] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN_VALUE}" == "1" ]]; then
  info "DRY_RUN=1: manifest preview follows; no lock, directory, manifest, or trainer artifact will be created"
  emit_manifest -
  exit 0
fi

mkdir -p "${JOB_DIR}"
if [[ -f "${MANIFEST_JSON}" ]]; then
  validate_existing_manifest "${MANIFEST_JSON}"
  mkdir -p "${JOB_DIR}/attempts"
  ATTEMPT_TAG=$(date -u +%Y%m%dT%H%M%SZ)_pid$$
  ACTIVE_MANIFEST="${JOB_DIR}/attempts/${ATTEMPT_TAG}.json"
else
  ACTIVE_MANIFEST="${MANIFEST_JSON}"
fi
emit_manifest "${ACTIVE_MANIFEST}"

info "starting delivery trainer"
"${CMD[@]}"

# Validate the final EMA state and atomically publish the completion marker.
/usr/bin/python -S - "${SUMMARY_CSV}" "${LAST_CKPT}" "${COMPLETION_JSON}" \
  "${ACTIVE_MANIFEST}" "${MODEL_ID_ARG}" "${RESIZE_ARG}" "${SEED_ARG}" <<'PY'
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import torch

summary = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
output = Path(sys.argv[3])
manifest = Path(sys.argv[4])
model_id, resize, seed = sys.argv[5:]
if not summary.is_file() or not checkpoint.is_file():
    raise SystemExit("final summary.csv/last.pth.tar missing")
with summary.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
target = [row for row in rows if int(float(row.get("epoch", -1))) == 199]
if not target:
    raise SystemExit("summary.csv lacks epoch 199")
row = target[-1]
if row.get("eval_top1") in (None, ""):
    raise SystemExit("epoch 199 lacks eval_top1")
ema_top1 = float(row["eval_top1"])
if not math.isfinite(ema_top1):
    raise SystemExit("epoch 199 eval_top1 is non-finite")
try:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(checkpoint, map_location="cpu")
if int(ckpt.get("epoch", -1)) != 199:
    raise SystemExit(f"last checkpoint epoch={ckpt.get('epoch')}, expected 199")
if not any(ckpt.get(key) is not None for key in ("state_dict_ema", "model_ema")):
    raise SystemExit("last checkpoint lacks EMA state")

valid = []
for candidate in rows:
    try:
        value = float(candidate["eval_top1"])
        epoch = int(float(candidate["epoch"]))
    except (KeyError, TypeError, ValueError):
        continue
    if math.isfinite(value):
        valid.append((value, epoch))
best_value, best_epoch = max(valid)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

payload = {
    "schema_version": "mergenet.cifar_resize_completion.v1",
    "status": "complete",
    "epoch": 199,
    "ema": True,
    "metric_key": "eval_top1",
    "ema_top1": ema_top1,
    "best_ema_top1_appendix": best_value,
    "best_ema_epoch_appendix": best_epoch,
    "model_id": model_id,
    "resize": int(resize),
    "seed": int(seed),
    "summary_path": str(summary.resolve()),
    "checkpoint_path": str(checkpoint.resolve()),
    "manifest_path": str(manifest.resolve()),
    "summary_sha256": digest(summary),
    "checkpoint_sha256": digest(checkpoint),
    "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
}
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY

info "complete: ${COMPLETION_JSON}"
