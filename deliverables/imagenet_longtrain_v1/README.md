# MergeNet ImageNet-1K 300-epoch handoff

This directory is a self-contained training handoff for single-node, multi-GPU ImageNet-1K runs. It contains source code, pinned CV dependencies, three auditable YAML protocols, a guarded `torchrun` launcher, and preflight tests. It intentionally contains no dataset, checkpoint, training log, or prior experiment output.

## Readiness and scope

- `configs/mergenet_lambda4.yaml` is the recommended long-run candidate: p8, 4 local + 8 latent blocks, lambda=4, local window 32, and the deterministic fast evaluation grouping.
- `configs/mergenet_lambda2.yaml` is the conservative fallback: p8, 6 local + 6 latent blocks, lambda=2, and local window 16. It retains more tokens and therefore costs more memory/compute.
- `configs/deit_small_p8_baseline.yaml` is the matched DeiT-S/8 baseline.

The architecture and efficiency choices above come from the completed CIFAR-100 campaign. **ImageNet-1K accuracy has not yet been measured.** Treat lambda=4 as the recommended scale-up candidate, not as a claimed ImageNet result. Run the baseline and lambda=4 under the same protocol; use lambda=2 if the lambda=4 accuracy curve is clearly under the baseline early in training.

## 1. Environment

The lock targets Linux x86_64, Python 3.10, Ninja 1.11.1, PyTorch 2.6.0 + CUDA 12.4, and FlashAttention 2.7.4.post1.

```bash
conda env create -f environment.yml
conda activate mergenet-in1k
python -m pip install -r requirements-lock.txt
```

The FlashAttention entry is the upstream prebuilt `cp310`, CUDA 12, PyTorch 2.6, CXX11-ABI-false wheel. If the target machine does not match that ABI/platform, install the same FlashAttention release from source, then run the full preflight. Do not start a long run after skipping the attention tests.

## 2. Dataset layout

`DATA_DIR` must point to an ImageFolder-style ImageNet-1K root with 1,000 matching class directories in both splits:

```text
${DATA_DIR}/
├── train/
│   ├── n01440764/*.JPEG
│   └── ...
└── val/
    ├── n01440764/*.JPEG
    └── ...
```

The original ILSVRC validation archive is flat; reorganize it into class directories before launching.

## 3. Preflight

Run from this directory. The preflight checks the 1,000 matching class folders and official image counts (1,281,167 train / 50,000 val), dependency versions, CUDA/FlashAttention availability, YAML keys, launcher state guards, a real temporary ImageFolder loader, accumulation/soft-target scheduling, attention forward/backward parity, and a CUDA model smoke test.

```bash
DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=/path/to/output \
GPUS=0,1,2,3,4,5,6,7 \
bash scripts/preflight_imagenet.sh configs/mergenet_lambda4.yaml
```

All checks must pass. `SKIP_GPU_TESTS=1` exists only for CPU packaging inspection and is not approval for a long run.
The standalone preflight validates `GPUS` and maps it to
`CUDA_VISIBLE_DEVICES`, exactly like the launcher; duplicate, malformed, or
zero-padded GPU IDs are rejected before any CUDA import.

## 4. Launch

The safe default targets an effective global batch of 1,024 and keeps the micro-batch at or below 64 by selecting gradient accumulation automatically. For 8 GPUs this resolves to batch 64/GPU and `update_freq=2`; for 4 GPUs it resolves to batch 64/GPU and `update_freq=4`.

```bash
DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=/path/to/output \
GPUS=0,1,2,3,4,5,6,7 \
RUN_NAME=in1k300_mergenet_lambda4_seed42 \
bash scripts/train_imagenet_300e.sh configs/mergenet_lambda4.yaml
```

Matched baseline:

```bash
DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=/path/to/output \
GPUS=0,1,2,3,4,5,6,7 \
RUN_NAME=in1k300_deit_small_p8_seed42 \
bash scripts/train_imagenet_300e.sh configs/deit_small_p8_baseline.yaml
```

Useful launcher variables:

| Variable | Default | Meaning |
|---|---:|---|
| `GPUS` | all visible GPUs | Comma-separated physical GPU IDs; sets `CUDA_VISIBLE_DEVICES`. |
| `NPROC_PER_NODE` | inferred | Number of local `torchrun` workers. |
| `GLOBAL_BATCH` | `1024` | Effective batch across all workers and accumulation steps. |
| `MAX_MICRO_BATCH` | `64` | Largest automatically selected per-GPU batch. |
| `UPDATE_FREQ` | auto | Gradient accumulation steps. If set, the launcher derives the micro-batch and validates exact divisibility. |
| `BATCH_SIZE` | auto | Per-GPU micro-batch. Can be set together with, or instead of, `UPDATE_FREQ`. |
| `VAL_BATCH_SIZE` | YAML value | Optional per-GPU validation batch override. |
| `RESUME` | `auto` | `auto`, `none`, or an explicit checkpoint file. |
| `ALLOW_EXISTING_RUN_DIR` | `0` | `1` permits scratch/auto-without-last reuse only when every existing entry is a top-level regular `launcher_*.log` file. |
| `MASTER_PORT` | `29500` | Local rendezvous port. |
| `RUN_PREFLIGHT` | `1` | Run all safety gates immediately before `torchrun`. |
| `DRY_RUN` | `0` | Print the resolved command without starting training. |

The invariant is:

```text
effective global batch = BATCH_SIZE * NPROC_PER_NODE * UPDATE_FREQ
```

The YAML learning rate (`5e-4`) is the protocol value for global batch 1,024. If `GLOBAL_BATCH` is changed, pass an intentionally scaled `--lr` after the YAML path, for example:

```bash
GLOBAL_BATCH=512 bash scripts/train_imagenet_300e.sh \
  configs/mergenet_lambda4.yaml --lr 2.5e-4
```

Only two arguments may follow the YAML path: a numeric `--lr` override and the
opt-in `--prefetcher` flag. All protocol identity, path, resume, epoch, batch,
dataset, output, checkpoint, and architecture settings are rejected there so
the executed command cannot diverge from the preflighted YAML and launcher
state. To change any of those settings, create and preflight a new YAML or use
the documented launcher variable (`DATA_DIR`, `OUTPUT_DIR`, `RUN_NAME`,
`RESUME`, `BATCH_SIZE`, `UPDATE_FREQ`, or `VAL_BATCH_SIZE`).

The three formal YAML files train from scratch (`pretrained: false`). This
trimmed handoff deliberately excludes MergeNet/ToMe's historical pretrained
loader and differential-local-LR optimizer: a MergeNet/ToMe invocation with
`--pretrained`, or with `--lr_local` different from `--lr`, fails before model
construction. Use `--initial_checkpoint` for a local checkpoint and keep a
single learning rate. The pinned upstream timm DeiT baseline retains its own
`--pretrained` support, although the formal baseline protocol also starts from
scratch.

All delivered YAML files explicitly keep `no_prefetcher: true`, which uses the verified host-side Mixup path. The optional trainer flag `--prefetcher` enables timm's CUDA prefetcher/FastCollateMixup path; use it only after an additional short smoke run confirms finite loss and correct soft-target handling on the target environment.

## 5. Resume and outputs

Each run writes to `${OUTPUT_DIR}/${RUN_NAME}`. The trainer maintains `last.pth.tar`, `model_best.pth.tar`, a short checkpoint history, `args.yaml`, and `summary.csv`. Launcher stdout/stderr is appended to a timestamped file in the same run directory.

- `RESUME=auto` resumes `${OUTPUT_DIR}/${RUN_NAME}/last.pth.tar` when it exists; otherwise it starts from scratch only if the run directory is empty.
- `RESUME=/path/to/checkpoint.pth.tar` requires that exact file to exist.
- `RESUME=none` starts from scratch and refuses to reuse a non-empty run directory, except for the log-only case below.

Use a new `RUN_NAME` for a fresh replicate. To retry an aborted launch in a
directory containing only top-level regular `launcher_*.log` files, explicitly
set `ALLOW_EXISTING_RUN_DIR=1`. This exception never admits arbitrary files,
symlinks, directories, or checkpoint/model artifacts (`*.pth*`, `*.pt`,
`*.ckpt`, `*.safetensors`, `checkpoint-*`, `model_best*`, or `last*`); those
remain fatal for scratch and auto-without-last runs regardless of the flag.

## 6. First-run monitoring

Before leaving a 300-epoch job unattended, verify the first optimizer updates and the first validation pass:

- every rank reports the expected world size and no unused-parameter/DDP error;
- loss and gradients remain finite under AMP;
- the default host-side Mixup path is active unless `--prefetcher` was deliberately smoke-tested;
- the printed micro-batch, accumulation, and effective global batch match the intended protocol;
- `summary.csv`, `last.pth.tar`, and recovery checkpoints appear in the selected run directory;
- restarting once with the same `RUN_NAME` and `RESUME=auto` continues at the next epoch.

## Attribution and license

This handoff is derived from OpenToMe by Westlake University CAIRI AI Lab and retains the upstream Apache-2.0 license and source-file notices. See `LICENSE`.
