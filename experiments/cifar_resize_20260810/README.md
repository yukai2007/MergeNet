# MergeNet CIFAR-100 resize 交付态验证

这套实验回答一个严格问题：在相同 CIFAR-100、DeiT-S/8 宽度、200 epoch、no-KD、global batch 200 协议下，交付态 MergeNet 随输入分辨率增长后，是否相对 DeiT 同时保住准确率并改善训练效率。

本目录只提供可审计、可恢复的执行层；历史 seed42 resize 结果仅用于筛选，不并入本轮统计。正式结论必须来自本协议新跑出的完整结果。

## 锁定矩阵

- resize：160、192、224、256、320，patch size 均为 8。
- 模型：`deit_s8`、`mn_l2`（6+6、lambda=2、window=16）、`mn_l4`（4+8、lambda=4、window=32）。
- seed：42、43、44，共 45 个单卡准确率任务。
- 准确率：scratch、no KD、200 epochs、AdamW、cosine、global batch 200、FP16、EMA。
- batch/累积：160/192/224 为 `200×1`，256 为 `100×2`，320 为 `50×4`。
- MergeNet：训练 `random_per_sample`，交付态评估 `alternating_per_layer_fast`；lambda curriculum 0–50 epoch，soft-topk 20–40 epoch ramp。
- 主准确率：最后一轮 epoch 199 的 EMA top-1。训练期间 best EMA 只进附录，不能替代主指标。
- 效率：每张物理 GPU 独立跑完整矩阵，synthetic FP32 input tensor + FP16 autocast、batch 32、20 次 warmup、100 次计时；训练测 random grouping，推理同时报告 generic/fast 及 logits parity。保留逐 seed/逐卡值并报告 mean ± sample SD，不声称 95% 置信区间。

完整单真源在 [protocol.json](protocol.json)。`run_accuracy_job.sh` 会在 GPU 初始化和写 artifact 前逐项核对所有影响结果的字段，协议漂移会 fail closed。

## 判定规则

`mn_l4` 只有在 resize 256 和 320 **各自**都满足以下三项时，才判定“有效”：

1. 三 seed 配对的 epoch-199 EMA top-1 差值 `mn_l4 - deit_s8 > 0 pp`；
2. 八张卡训练吞吐比 `mn_l4 / deit_s8 >= 1.0`；
3. 八张卡训练 peak allocated 显存比 `mn_l4 / deit_s8 < 1.0`。

必须具有 3 个完整准确率 seed 和 8 张卡的正式效率复现。推理吞吐、延迟、显存和 generic/fast parity 完整报告，但不作为总 PASS 的强制条件；160/192/224 与 `mn_l2` 用于解释趋势和 Pareto，不改变上述门槛。token reduction 不能替代实测效率。

mandatory efficiency 是 target-lambda 下 synthetic model-only 的 steady-state forward/backward/optimizer-step microbenchmark。MN 准确率前 50 epoch 有 lambda curriculum，因此这不是 200 epoch 端到端 wall-clock 平均；`summary.csv` 的真实 train/eval throughput 与 memory 只作补充，必须和 microbenchmark 分开命名。

45 个准确率任务完成后还有一个不阻塞启动、但阻塞最终 release 的后验：对 30 个 MN epoch-199 EMA checkpoint（2 模型 × 5 resize × 3 seed），用同一个 CIFAR-100 test 10k deterministic loader 做 generic/fast 全量复评，记录 top-1 delta、argmax mismatch、最大/平均 logit diff。每个 run 必须 `|Δtop1| <= 0.05 pp`（最多 5/10000）；任一失败则 release NO-GO，但性能 gate 数值仍原样报告。

## 环境与数据

默认正式依赖根为：

```text
/liziqing/yukai/.deps_mergenet_resize20260810
```

执行统一使用 `/usr/bin/python -S` 和隔离的 `PYTHONPATH=RUNTIME_ROOT:DEPS_ROOT`。协议锁定 Python 3.10、torch 2.6.0+cu124、torchvision 0.21.0+cu124、timm 0.9.11、FlashAttention 2.7.4.post1；模块若从系统 site-packages 泄漏会直接失败。

运行时还固定 `OPENTOME_MERGENET_IMPL=new`、`TIMM_FUSED_ATTN=1`；会拒绝 torchrun/DDP 遗留变量、`PYTHONOPTIMIZE` 以及会改变 CUDA/allocator/TF32 行为的环境变量。

`DATA_DIR` 应指向已下载的 torchvision CIFAR 根目录，并包含：

```text
${DATA_DIR}/cifar-100-python/train
${DATA_DIR}/cifar-100-python/test
```

默认是 `/liziqing/yukai/data`，正式任务禁止下载数据。

## 启动前审计

下面只检查依赖来源并打印完整 45-job 计划，不创建 runtime snapshot、不写 campaign 状态、不启动训练，也不会初始化 CUDA：

```bash
cd /liziqing/yukai/MergeNet
bash experiments/cifar_resize_20260810/launch_all.sh --dry-run
```

单个 accuracy wrapper 还支持更强的 `DRY_RUN=1`：执行全部协议、数据、依赖、GPU 映射、既有 run 状态与 manifest 预览 guard，但不创建 lock、目录或训练 artifact。它会在指定卡上做一个极小 CUDA 算术检查，因此只能对已确认空闲的卡使用：

```bash
CAMPAIGN_ROOT=/path/to/campaign \
RUNTIME_ROOT=/liziqing/yukai/MergeNet/deliverables/imagenet_longtrain_v1 \
DEPS_ROOT=/liziqing/yukai/.deps_mergenet_resize20260810 \
DATA_DIR=/liziqing/yukai/data \
DRY_RUN=1 \
bash experiments/cifar_resize_20260810/run_accuracy_job.sh mn_l4 320 42 7
```

固定接口为：

```text
run_accuracy_job.sh MODEL_ID RESIZE SEED PHYSICAL_GPU
```

其中 `MODEL_ID` 只能是 `deit_s8`、`mn_l2`、`mn_l4`。

## 正式挂起 8 卡队列

确认 campaign root 是本轮专用的新目录后执行：

```bash
cd /liziqing/yukai/MergeNet
CAMPAIGN_ROOT=/liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
DEPS_ROOT=/liziqing/yukai/.deps_mergenet_resize20260810 \
DATA_DIR=/liziqing/yukai/data \
GPUS=0,1,2,3,4,5,6,7 \
bash experiments/cifar_resize_20260810/launch_all.sh
```

正式启动器会：

1. 一次性复制交付代码和实验 harness 到 `${CAMPAIGN_ROOT}/runtime/`；
2. 为两个树生成逐文件 SHA-256，整体设为只读；已有 snapshot 只验证，绝不覆盖；
3. 隐藏 CUDA 后验证 exact dependencies，不碰正在使用的卡；
4. 通过 `setsid` + `nohup` 启动 snapshot 内的 `campaign.py`；
5. 等 master heartbeat 出现并确认进程存活后才返回成功。

不要给 `launch_all.sh` 设置 `DRY_RUN`，它会主动拒绝，以免变量泄漏让正式队列变成空跑。协议、交付代码或 harness 有任何改变时，必须换一个新的 `CAMPAIGN_ROOT`，不能修改已有 snapshot。

## 资源门禁

“8 卡空着”由队列在每次任务启动前重新验证，不能只看一次 `nvidia-smi`：

- 目标物理卡无 compute process；
- free memory 至少 70,000 MiB；
- GPU utilization 不超过 5%；
- 连续两次探测通过，间隔 10 秒；
- 主机 `load1 / process_affinity_cpu_count <= 1.5`。

效率计时项的开始和结束都必须通过 host load 门禁；超阈值样本保留为 `timing_valid=false` 供审计，但不计入正式效率并会重测。logits parity 不受 timing 门禁影响。若某张卡或主机当前繁忙，master 会等待，不会抢卡。

## 监控

一次性查看 master、heartbeat、CPU load gate、8 卡状态、任务计数和每个在跑任务的最新 epoch/EMA top-1：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/monitor.py \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810
```

持续刷新或输出 JSON：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/monitor.py \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --watch --interval 10

/usr/bin/python -S experiments/cifar_resize_20260810/monitor.py \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --json
```

### Benchmark GPU co-tenancy 独立审计

启动门禁只能证明任务启动前卡为空闲；若外部进程在 benchmark 已经开始后抢入，同卡计时仍可能被污染。`watch_gpu_cotenancy.py` 是 source tree 中的独立、只读 watcher，不修改 immutable runtime、campaign state 或调度状态，也不会向任何进程发信号。它只读取 heartbeat 中 `kind=benchmark` 的物理卡映射，查询 `nvidia-smi` compute apps，并只写入：

```text
${CAMPAIGN_ROOT}/audit/gpu_cotenancy/
```

先运行不访问 GPU、也不需要 campaign 的标准库自测：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/watch_gpu_cotenancy.py --self-test
```

通过复核后可单实例 detached 启动；子进程写出匹配 PID 的 `status.json` 后命令才返回成功，campaign master 退出或身份改变时 watcher 会自动结束：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/watch_gpu_cotenancy.py \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --interval-sec 5 \
  --detach
```

查看总体状态、异常事件和逐 benchmark session 的最大 compute-app 数：

```bash
jq . /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/audit/gpu_cotenancy/status.json

tail -f /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/audit/gpu_cotenancy/events.jsonl

jq '{session_id, physical_gpu, samples, max_compute_app_count, anomalous_samples, possible_cotenancy_observed}' \
  /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/audit/gpu_cotenancy/sessions/*.json
```

单个 heartbeat benchmark 正常最多对应一个 compute app；`compute_app_count > 1` 会记录为 possible co-tenancy，`0` 则允许出现在 CUDA 初始化前或退出阶段。所有轮询样本原子追加到 `samples.jsonl`，异常开始/恢复、query failure、session start/end 写入 `events.jsonl`，每个 session 的累计 `samples`、`max_compute_app_count` 和异常计数原子更新到 `sessions/*.json`。watcher 仅提供审计证据，不会自行宣布 benchmark 无效或触发重跑。

## 聚合证据

任务进行中可随时生成 partial evidence（缺证据只标 `INCOMPLETE`）：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/aggregate_results.py \
  --protocol /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/runtime/cifar_resize_20260810/protocol.json \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --out-dir /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/aggregate
```

`--strict-complete` 只锁主实验语义：若 45 个准确率任务、8 份效率文件或性能 gate 证据不完整，以状态码 2 拒绝完成；完整且科学结论为 FAIL 时仍返回 0，使失败报告能够正常发布。它不替代 30-checkpoint 后验：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/aggregate_results.py \
  --protocol /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/runtime/cifar_resize_20260810/protocol.json \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --out-dir /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/aggregate \
  --strict-complete
```

证据包实际文件名为 `aggregate_results.json`（机器可读全量证据）、`aggregate_results.csv`（表格）和 `aggregate_results.md`（人读报告）。JSON/Markdown 分开显示主实验 λ4 `PASS|FAIL|INCOMPLETE` 与最终发布 `READY|NO_GO|INCOMPLETE`，性能数值不会因后验失败而删除或改写。

## 训练后 checkpoint parity（最终发布门禁）

只在 30 个 MergeNet accuracy job 全部生成合法 epoch-199 EMA completion marker 后运行。runner 位于 source harness，但只读取当前 campaign 的 immutable protocol 和 delivery runtime；结果写到 `${CAMPAIGN_ROOT}/post_training_parity/`，不会修改 `${CAMPAIGN_ROOT}/runtime/`。每个 checkpoint 一个独立、带 flock 的原子 JSON，合法 PASS 和合法 FAIL 都是可恢复复用的终态。

先做只读审计。该命令会验证 snapshot/protocol/data/checkpoint readiness 并打印 30-task 计划，不导入 torch、不初始化 CUDA、不创建 lock 或结果目录：

```bash
cd /liziqing/yukai/MergeNet
PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python -S experiments/cifar_resize_20260810/post_training_parity.py \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --dry-run
```

正式单卡顺序跑完整 30 项时，先确认目标卡空闲，并确保 torchrun/DDP、`PYTHONOPTIMIZE`、CUDA allocator/TF32 扰动变量均未继承：

```bash
cd /liziqing/yukai/MergeNet
CAMPAIGN_ROOT=/liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810
RUNTIME_ROOT=${CAMPAIGN_ROOT}/runtime/imagenet_longtrain_v1
DEPS_ROOT=/liziqing/yukai/.deps_mergenet_resize20260810

env -u CUDA_VISIBLE_DEVICES -u PYTHONOPTIMIZE \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u ROLE_RANK -u MASTER_ADDR -u MASTER_PORT \
  PYTHONDONTWRITEBYTECODE=1 \
  OPENTOME_MERGENET_IMPL=new TIMM_FUSED_ATTN=1 \
  PYTHONPATH="${RUNTIME_ROOT}:${DEPS_ROOT}" \
  /usr/bin/python -S experiments/cifar_resize_20260810/post_training_parity.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --protocol "${CAMPAIGN_ROOT}/runtime/cifar_resize_20260810/protocol.json" \
    --runtime-root "${RUNTIME_ROOT}" \
    --deps-root "${DEPS_ROOT}" \
    --data-dir /liziqing/yukai/data \
    --gpu 7
```

要在 8 张空闲卡上并行，分别启动 8 个相同命令，并给第 `k` 个进程增加 `--shard-count 8 --shard-index k --gpu <physical_gpu>`（`k=0..7`）。shard 依据固定 30-task 顺序划分，输出文件互不冲突；进程中断后重跑同一命令只补非终态项。每份证据记录 protocol 原始/规范 digest、immutable runtime fingerprint、runner SHA、checkpoint/completion SHA、CIFAR test SHA、exact dependency 版本和物理 GPU UUID。

后验只以每个 run 的 generic/fast top-1 正确数差判定：`abs(delta) <= 5/10000`，即 `|Δtop1| <= 0.05 pp`。`argmax_mismatch_count` 可以大于 5；它和 max/mean logit diff 仅用于诊断，不是额外否决条件。runner 返回 0 表示所选项全 PASS、2 表示执行/证据不完整、3 表示至少一个经过验证的 run FAIL。

最终发布检查使用 source 中新增了后验逻辑的 aggregator（当前 immutable campaign snapshot 中的旧 aggregator 不具备这项门禁）：

```bash
/usr/bin/python -S experiments/cifar_resize_20260810/aggregate_results.py \
  --protocol /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/runtime/cifar_resize_20260810/protocol.json \
  --campaign-root /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810 \
  --out-dir /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_resize_delivery_validation_20260810/aggregate \
  --strict-final-release
```

`--strict-final-release`（别名 `--require-release-go`）仅在主性能 gate PASS 且 30/30 checkpoint parity 全部有效并 PASS 时返回 0/`READY`；缺失或非法证据返回 2/`INCOMPLETE`；任何已验证的 mandatory failure 返回 3/`NO_GO`。聚合器会重新计算 top-1、计数和 gate，并重新散列当前 checkpoint/data/runtime，不能靠修改结果 JSON 绕过。

## 恢复与完成判据

每个 job 固定落在：

```text
${CAMPAIGN_ROOT}/runs/<model_id>/r<resize>/seed<seed>/
```

- `summary.csv`：每 epoch 指标；其 `eval_top1` 是 EMA 验证结果。
- `last.pth.tar`：自动 resume 的唯一 checkpoint。
- `manifest.json`：首个 attempt 的协议、runtime、环境和命令证据。
- `attempts/*.json`：后续 resume attempt，保留历史，不覆盖首个 manifest。
- `completion.json`：仅当 summary epoch=199 且 last checkpoint epoch=199、含 EMA state 后原子写入。

状态处理是 fail closed：

- epoch 199 + EMA 完整：验证后 skip，绝不重写 manifest/completion；
- partial 且存在合法 `last.pth.tar`：完整恢复 model、optimizer、AMP、EMA；
- 首个 checkpoint 前中断：只有目录中严格限于 regular `manifest.json`、`args.yaml`、`attempts/*.json` 时才允许 scratch retry；
- partial 无 last、checkpoint/summary 冲突、未知 artifact、任意 symlink：拒绝继续，等待人工审计。

master 被中断后，用同一条正式启动命令即可恢复；snapshot 和已完成任务会被复用。不要手工删改 run 目录来“修复”状态。

## 文件职责

- [campaign.py](campaign.py)：8 卡可恢复调度、双重空闲门禁、状态与 heartbeat。
- [benchmark_resize.py](benchmark_resize.py)：每卡完整效率矩阵、parity 与有效性标记。
- [run_accuracy_job.sh](run_accuracy_job.sh)：单 GPU 准确率、协议锁、resume、manifest/completion。
- [launch_all.sh](launch_all.sh)：不可变 snapshot、exact dependency 验证、后台启动。
- [monitor.py](monitor.py)：只读状态/GPU/summary 监控。
- [watch_gpu_cotenancy.py](watch_gpu_cotenancy.py)：benchmark 运行期 compute-app co-tenancy 的独立原子审计。
- [aggregate_results.py](aggregate_results.py)：partial/final 准确率与效率证据聚合、预注册 gate 判定。
- [post_training_parity.py](post_training_parity.py)：30 个 epoch-199 EMA checkpoint 的全量 CIFAR-100 generic/fast 最终发布门禁。
- [test_post_training_parity.py](test_post_training_parity.py)：CPU-only matrix、边界、resume identity 与聚合 fail-closed fixtures。

在 45 个准确率任务和 8 份效率矩阵全部完成前，不应把这轮实验描述为“已经证明 resize scale-up 有效”；即使主实验 PASS，在 30/30 checkpoint parity 全部通过前也不能交付为 final-release-ready。
