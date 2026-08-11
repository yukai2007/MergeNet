# MergeNet ImageNet 长训包源码清单

## 来源与版本锚点

- 上游项目：OpenToMe，Westlake University CAIRI AI Lab。
- 上游许可：Apache License 2.0；原始源码版权头与完整 `LICENSE` 均保留。
- 开发分支锚点：`mergenet_on_cifar_work`。
- 最近的 Git 基线提交：`b6cd1b4ea76148127e30be4a650e0251fe757015`。
- 交付标识：`imagenet_longtrain_v1`，整理日期 2026-08-09。

该提交只用于追溯开发起点。交付包还包含在该基线之后完成并通过回归测试的 ImageNet 长训修复，因此它是一个经过裁剪的发布快照，不是上述提交的逐字节导出。最终交付仓库的提交 SHA 才是本包的完整内容标识。

## 随包源码

| 路径 | 用途 |
|---|---|
| `opentome/__init__.py`, `opentome/version.py` | 最小包入口与上游版本信息。 |
| `opentome/models/__init__.py` | MergeNet 模型导出；DeiT-S/8 对照由锁定的上游 timm 提供。 |
| `opentome/models/mergenet/` | `mergenet_small_cls` 分类模型、4+8/6+6 几何及路由逻辑。 |
| `opentome/timm/attention.py` | 通用 attention 辅助实现。 |
| `opentome/timm/bias_local_attn.py` | 有偏/无偏局部 FlashAttention，显式支持 BHND/BNHD 布局并保证只缩放一次。 |
| `opentome/timm/dtem.py` | DTEM 分组、合并与 `alternating_per_layer_fast` 路径。 |
| `opentome/timm/tome.py` | timm 侧 token merge 接口。 |
| `opentome/tome/tome.py` | merge/unmerge 与 source map 基础实现。 |
| `opentome/utils/dataset_loader.py` | CIFAR/ImageFolder 数据入口；ImageNet 别名统一映射到 timm ImageFolder reader。 |
| `opentome/utils/thetopk.py` | soft top-k 选择。 |
| `trainer/classification/in1k_trainer.py` | ImageNet DDP/AMP、梯度累积、EMA、恢复、日志与 checkpoint 主入口。 |

仅为维持 Python 包导入而存在的同级 `__init__.py` 也属于交付源码。

## 配置与运维入口

| 路径 | 用途 |
|---|---|
| `configs/deit_small_p8_baseline.yaml` | 300e DeiT-S/8 对照。 |
| `configs/mergenet_lambda4.yaml` | 推荐主候选：p8、4+8、lambda=4、window 32、fast eval grouping。 |
| `configs/mergenet_lambda2.yaml` | 保守回退：p8、6+6、lambda=2、window 16。 |
| `scripts/train_imagenet_300e.sh` | 无本机绝对路径的单机多卡 `torchrun` 启动器；负责批量/累积、resume，以及 checkpoint-safe 的输出目录保护。 |
| `scripts/gpu_visibility.sh` | launcher 与独立 preflight 共用的 `GPUS` 合法性、去重及 `CUDA_VISIBLE_DEVICES` 映射。 |
| `scripts/preflight_imagenet.sh` | 数据、版本、CUDA、FlashAttention、配置与五项回归测试门禁。 |
| `environment.yml`, `requirements-lock.txt` | Python 3.10、Ninja 1.11.1、PyTorch 2.6/CUDA 12.4 与 CV 依赖锁。 |
| `README.md`, `RELEASE_NOTES.md` | 学长交接说明与发布边界。 |

## 回归测试

| 路径 | 覆盖范围 |
|---|---|
| `tests/test_launcher_guards.sh` | launcher dry-run；拒绝未审计覆盖；验证独立 preflight GPU 映射，以及 log-only 例外和 scratch/auto checkpoint artifact 拒绝。 |
| `tests/test_imagefolder_loader.py` | 构造真实临时 `train/val` ImageFolder，并走交付数据加载路径。 |
| `tests/test_accumulation_schedule.py` | 完整/尾部梯度累积组、更新编号，以及 host/prefetcher Mixup active 判定。 |
| `tests/test_biased_local_attention.py` | CUDA + FlashAttention 的 BHND/BNHD 前后向 parity、局部性与 DTEM 调用约定。 |
| `tests/test_model_smoke.py` | MergeNet CUDA 小图前向、反向、所有已产生梯度的有限性、保留 token 数，并断言 canonical 模块树不含 `DTEMAttention`/不调用 biased 兼容路径。 |

## 有意排除

交付包不包含以下开发期内容：

- ImageNet/CIFAR 数据、checkpoint、optimizer state、训练日志、W&B 缓存和 `work_dirs`；
- 历史消融模型、旧启动脚本、benchmark 原始输出和可视化中间产物；
- NLP 模型、tokenizer、LM trainer 及其依赖；
- 开发工作树的嵌套 `.git`、远端配置、个人路径和凭据；
- `__pycache__`、`.pytest_cache` 等可再生缓存（已由 `.gitignore` 排除）。

这些排除项不影响三份配置对应的 ImageNet 分类训练。发布前可用以下命令确认 Git 将要纳入的内容：

```bash
git status --short -- deliverables/imagenet_longtrain_v1
git ls-files deliverables/imagenet_longtrain_v1
```
