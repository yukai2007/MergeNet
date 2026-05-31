# OpenToMe MergeNet Source Trace Update

本文记录 2026-05-30 至 2026-05-31 对 `/liziqing/yukai/OpenToMe` 中视觉版 MergeNet 的机制修正、消融脚本与代码理解更新。它不是论文方法的最终表述，而是当前实现的工程事实记录。

## 1. 已完成的代码侧更新

### 1.1 `source_trace_mode`

`LocalEncoder` 现在支持四种 source trace 模式：

| 模式 | 行为 | 用途 |
| --- | --- | --- |
| `matrix` | 维护完整 source matrix，cross attention 使用 `log(source)` 作为 additive bias | 语义最完整的参考路径 |
| `detached` | 维护完整 source matrix，但更新在 `torch.no_grad()` 下执行 | 检查反向图与显存的影响 |
| `center` | LocalEncoder 内维护 token center-of-mass，不构造 dense source matrix | 当前轻量默认路径 |
| `none` | 不维护 source trace / source bias | 下界对照 |

关键区别：`center` 不是 `matrix` 的语义等价替代。它不提供 Perceiver-style cross attention 的 source-aware bias；在当前 CLS 分类训练路径中，`token_center` 也尚未被 `CLSHybridToMeModel.forward()` 消费，所以 `center` 消融更准确地解释为“关闭 dense source matrix / source bias 的轻量下界”，而不是完整的 center-guided source trace。

### 1.2 global source matrix 正确维护

之前 global DTEM 分支存在两个问题：

1. `dtem_window_size=None/0` 时仍按 local window 的窄带宽度初始化 source matrix。
2. global assign 后没有把 Group A 的 source 分布平移累加到 Group B，而是直接保留原 source。

现在 global 模式按完整相对位置范围维护：

```text
width = 2N - 1
center = N - 1
```

其中 `N` 是当前包含 CLS 的 token 序列长度。若 A token 以权重 `T_ab[i,j]` 转移到 B token，则 A 的 source 分布会从 A 的相对坐标系平移到 B 的相对坐标系：

```text
delta = pos(a_i) - pos(b_j)
target_k = source_k + delta
```

然后 scatter-add 到 B 的 source 分布中。这个修正只会在 global DTEM 路径触发；正常超参脚本若显式设置 `branch_b_dtem_window_size=8`，走的是 local-window 窄带路径。

### 1.3 每次 forward 重置 `token_center`

`LocalEncoder.forward()` 每次开始时会重置：

```python
self._tome_info["source_matrix"] = None
self._tome_info["token_center"] = None
```

否则 `center` 模式在连续 batch 间可能复用上一次 forward 的 center trace 状态。

## 2. 当前实现的真实架构行为

### 2.1 Local Encoder 并不物理删除 token

当前训练路径是 soft merge：Group A 的质量衰减，Group B 吸收质量和特征，但 A/B token 都会被保留并按原始空间位置重新排序。也就是说 `x_local.shape[1]` 在 local merge 后通常仍保持原长度。

真正的压缩发生在 LocalEncoder 结束后的 top-k selection：

```text
k = L_full - total_merge_local - 1
```

对 patch size 8、image size 224、`lambda_local=4`：

```text
num_patches = 28 * 28 = 784
total_merge_local = floor(784 * 3 / 4) = 588
k = 784 - 588 = 196
topk_x length = 196 + CLS = 197
```

因此当前视觉版 MergeNet 的“压缩后 token 数”不是通过物理删 token 得到，而是通过质量转移后的 top-k 选择得到。

### 2.2 `matrix` 对 Perceiver-style cross attention 的作用

`CLSHybridToMeModel.forward()` 中：

1. 用 token size 选择 top-k token。
2. 在 `matrix` 路径用 source center-of-mass 对 top-k token 排序；当前 CLS 路径在 `center` 模式下没有消费 `token_center`，会退回原索引排序。
3. 构造 `topk_x` 作为 query。
4. 用 full local tokens `x_embed` 作为 key/value。
5. 执行：

```python
x_trace = encode_cross_attention(topk_x, x_embed, mask=bias) + topk_x
```

当 `source_trace_mode=matrix` 时，`bias` 由 top-k token 对原始 token 的 source distribution 构造，形式近似：

```text
bias[i, j] = log(source_i[j] + eps)
```

当 `source_trace_mode=center` 或 `none` 时，当前实现不会提供这个 source-aware bias，而是使用零 bias。因此如果论文叙事强调“word 从 byte level 获得语义时受 source matrix 指导”，严格对应的是 `matrix/detached` 路径，而不是当前 CLS 训练使用的 `center` 路径。

### 2.3 Dual AB 共享范围

当前 `CLSDualBranchHybridToMeModel` 不只是共享 embedding。代码中还共享：

- `patch_embed`
- `cls_token`
- `pos_embed`
- `norm_pre`
- local 4 层 `LocalBlock`
- local metric heads
- latent encoder
- encode/decode cross attention 模块

不共享的是两个分支各自的分类 `head`，以及 fusion head。Branch A 的 `lambda_local=1`，`total_merge_local=0`，因此不做 DTEM 合并；Branch B 按 `branch_b_lambda_local` 做压缩。

## 3. 新增 2k source matrix 消融脚本

在 OpenToMe 中新增：

```text
trainer/classification/2000s2_mergenet_source_matrix_on_0123.sh
trainer/classification/2000s2_mergenet_source_matrix_off_4567.sh
```

二者都沿用 `1000s2.sh` 的正常 MergeNet 超参，差异仅为：

```text
ON:  --source_trace_mode matrix, GPUs 0,1,2,3
OFF: --source_trace_mode center, GPUs 4,5,6,7
```

注意：这两个脚本都设置了：

```text
--branch_b_dtem_window_size 8
```

所以它们验证的是“正常 local-window 超参下 dense source matrix / source bias on/off”，不是 global source matrix 消融，也不是完整 center-guided 消融。

## 4. 实际训练核验

从当前 `summary.csv` 和 `args.yaml` 检查到：

| 配置 | source trace | branch B DTEM window | train throughput | train allocated memory | eval allocated memory |
| --- | --- | --- | --- | --- | --- |
| ON | `matrix` | 8 | epoch 0/1: 484.8 / 499.5 samples/s | about 31.2 GB | about 8.5 GB |
| OFF | `center` | 8 | epoch 0/1/2: 672.4 / 698.3 / 696.7 samples/s | about 13.1 GB | about 2.9 GB |

结论：

- 开关确实生效，显存差距很明显。
- throughput 差距小于单分支 synthetic efficiency bench，是因为 2k 训练是 dual AB joint：Branch A、LocalBlock、cross attention、fusion、loss、数据增强和 dataloader 都进入总耗时。
- 不能用这两个脚本判断 global source matrix 的成本；global 需要去掉或置零 `branch_b_dtem_window_size`。

## 5. 文档表述需要保持的边界

后续写论文/技术报告时建议保持以下说法：

1. local-window source matrix 是带状存储，宽度为 `2 * window_size * local_depth + 1`。
2. global source matrix 若要语义正确，需要完整相对位置范围，宽度为 `2N - 1`；不能再说“不需要 source matrix”。
3. 当前 CLS 训练里的 `center` 模式只是 dense source/bias off 的轻量下界；若要真正使用 center trace，还需让 `CLSHybridToMeModel.forward()` 消费 `token_center` 来排序或构造 center-band bias。
4. 当前训练实现是 soft merge + top-k selection，不是每层物理删除 token。
5. 视觉版 OpenToMe 的 CIFAR100 正常脚本使用 patch size 8，所以 patch 数是 784，不是 patch size 16 下的 196。
