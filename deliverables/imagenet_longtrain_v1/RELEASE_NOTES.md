# imagenet_longtrain_v1 发布说明

发布日期：2026-08-09

状态：可交付的 ImageNet-1K 长训候选；ImageNet 最终精度待本次大规模训练验证。

## 本版解决的问题

- 修正有偏局部 FlashAttention 的布局契约：BHND/BNHD 在 kernel 边界显式转换，输出恢复到输入布局。
- 消除 attention 双重缩放：query 与 `softmax_scale` 只应用一次标准缩放。
- 补齐 `alternating_per_layer_fast` 的训练参数解析与确定性评估路径。
- 修正 ImageNet 数据入口的函数名遮蔽与 `imagenet`/`imagefolder` 别名问题，YAML 统一使用 timm 0.9.11 的空 reader 名。
- 梯度累积现在覆盖 DDP `no_sync`、尾部不足组的正确 loss 缩放、optimizer/EMA/scheduler 更新边界。
- 显式 resume 路径不存在时立即失败；恢复时重建 checkpoint history/best metric，避免续训覆盖更优模型或重复 summary header。
- 修正 Mixup soft target 下的 accuracy 条件；安全默认固定 `no_prefetcher: true`，CUDA prefetcher 仅作为额外 smoke 后的可选项。
- W&B 只在 rank 0 初始化，避免 `torchrun` 多 rank 重复 run。
- 关闭未审计的 trainer 参数透传：YAML 后仅允许 `--lr` 与可选
  `--prefetcher`，其余协议、路径、resume、epoch、batch、数据和输出状态
  必须经过 YAML/preflight 或专用 launcher 变量。
- launcher 与独立 preflight 现在共用 `GPUS` 校验、去重及
  `CUDA_VISIBLE_DEVICES` 映射，README 的单独预检命令只检查指定设备。
- 修正 conda-forge 可解析的 Ninja 锁定版本为 `1.11.1`。
- `ALLOW_EXISTING_RUN_DIR=1` 现在只允许顶层普通 `launcher_*.log`；scratch
  或 auto-without-last 遇到任意 checkpoint/model artifact 都会无条件失败。
- 删除未交付的 p16 pooling 消融、自定义 DeiT wrapper 及模型文件内 dummy
  forward；对缺少随包实现的 MergeNet/ToMe `--pretrained` 与差分
  `--lr_local` 显式 fail-closed。正式三份 YAML 均不触发这些门禁；原生 timm
  DeiT 的 `--pretrained` 行为不受影响。

其中 biased FlashAttention 修复属于兼容路径加固：canonical
`mergenet_small_cls` 的模块树是 `LocalBlock`（unbiased attention）+
`DTEMMergeOnly`，不包含 `DTEMAttention`。回归测试会把兼容路径的
`biased_local_attention` 替换为 fatal stub 后再跑 canonical smoke，因此该修复不改变
历史 canonical CIFAR 结果的解释；ImageNet 仍需 fresh scale-up 验证。

## 交付训练协议

- 共同设置：ImageNet-1K、224×224、patch size 8、300 epochs、AdamW、global batch 1024、lr `5e-4`、5-epoch warmup、cosine、AMP、EMA、DeiT augmentation。
- 主训练：`configs/mergenet_lambda4.yaml`，4 local + 8 latent、lambda=4、window 32；这是 CIFAR-100 工程结果支持的效率优先候选。
- 对照训练：`configs/deit_small_p8_baseline.yaml`，DeiT-S/8，同一优化与增强协议。
- 回退训练：`configs/mergenet_lambda2.yaml`，6+6、lambda=2、window 16；保留更多 token，准确率风险更保守但成本更高。

启动器默认让单卡 micro-batch 不超过 64，并按 GPU 数自动解析 `update_freq`，保证：

```text
BATCH_SIZE × NPROC_PER_NODE × UPDATE_FREQ = GLOBAL_BATCH
```

默认 `RESUME=auto` 只从当前 `RUN_NAME` 下的 `last.pth.tar` 恢复；不存在
该文件时，非空目录默认拒绝。即使显式启用 `ALLOW_EXISTING_RUN_DIR=1`，
也只放过顶层普通 launcher 日志，checkpoint/model artifact 始终拒绝。

## 已通过验证

| 验证 | 结果 |
|---|---|
| `bash -n`：launcher + preflight + GPU helper + shell regression | PASS |
| `test_launcher_guards.sh` | `LAUNCHER_GUARD_TEST_PASS`；CLI 覆盖/GPU 映射门禁生效；log-only 目录可显式复用，scratch/auto 的 checkpoint artifact 均拒绝 |
| 三份 YAML 解析、trainer 完整 argparse、p8/300e 协议断言 | PASS |
| 三个实际模型工厂构造 | PASS：DeiT 22,055,272；lambda=2 23,097,064；lambda=4 23,047,784 参数 |
| `test_imagefolder_loader.py` | `IMAGEFOLDER_TRAIN_VAL_TEST_PASS` |
| `test_accumulation_schedule.py` | `ACCUMULATION_AND_MIXUP_STATE_TEST_PASS` |
| `test_biased_local_attention.py` | `BIASED_LOCAL_ATTENTION_TEST_PASS`；BHND/BNHD、N>H/N<H、局部性、DTEM 均通过 |
| `test_model_smoke.py` | `MERGENET_MODEL_SMOKE_PASS`；CUDA forward/backward 与有限梯度通过 |
| Apache-2.0 LICENSE | 与上游文件 SHA-256 一致 |
| 本机绝对路径/debug 环境变量扫描 | PASS；交付 docs/config/scripts 中未发现 |

## 尚未宣称的内容

- 尚未完成 ImageNet-1K 300e，因此本版不宣称 ImageNet Top-1、训练吞吐或最终 Pareto 优势。
- 当前 CIFAR 结果用于选择 scale-up 候选，不能直接外推为 ImageNet 精度。
- 预置 FlashAttention wheel 只面向 Linux x86_64、CPython 3.10、CUDA 12、PyTorch 2.6、CXX11 ABI false；其他平台需按同版本源码安装并重跑门禁。
- 启动器定位为单机多卡。多机训练需要另行确认共享存储、网络 rendezvous 与跨节点 resume，不在 v1 自动化范围内。
- 交付包不附带预训练权重、数据或历史日志。

## 学长侧验收顺序

```bash
conda env create -f environment.yml
conda activate mergenet-in1k
python -m pip install -r requirements-lock.txt

DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=/path/to/output \
GPUS=0,1,2,3,4,5,6,7 \
bash scripts/preflight_imagenet.sh configs/mergenet_lambda4.yaml

DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=/path/to/output \
GPUS=0,1,2,3,4,5,6,7 \
RUN_NAME=in1k300_mergenet_lambda4_seed42 \
bash scripts/train_imagenet_300e.sh configs/mergenet_lambda4.yaml
```

开始无人值守长训前，至少观察首个 optimizer update、首轮 validation、checkpoint 写入，并做一次 `RESUME=auto` 重启演练。完整操作与回退说明见 `README.md`。
