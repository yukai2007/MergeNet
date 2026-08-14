# MergeNet CV Release

This repository is the handoff surface for the MergeNet CV results and the
reproducible training materials.

- [ImageNet long-training handoff](deliverables/imagenet_longtrain_v1/): a
  ready-to-run, 300-epoch ImageNet paper-scale pretraining package with a
  canonical MergeNet launcher and a matched DeiT baseline protocol.
<!-- CIFAR_RESIZE_FINAL_ROOT_CAMPAIGN_STATUS:START -->
- [CIFAR resize validation](experiments/cifar_resize_20260810/): the completed 45-run accuracy / 8-card efficiency / 30-checkpoint parity campaign and its reproducible harness.
<!-- CIFAR_RESIZE_FINAL_ROOT_CAMPAIGN_STATUS:END -->
- [CV results report](reports/mergenet_cv_results_20260809.html): the HTML
  report summarizing the currently verified outcomes and delivery boundary.

Start with the ImageNet handoff README when launching large-scale training.

<!-- CIFAR_RESIZE_FINAL:START -->
## CIFAR resize 最终结果

- [最终 HTML 汇报](reports/mergenet_cifar_resize_final_20260814.html)：完整 5 resize × 3 model 精度、paired delta、8 卡效率和 30-checkpoint parity。
- [图形化 HTML 看板](reports/mergenet_cifar_resize_visual_report_20260814.html)：基于同一锁定 aggregate 的 Top-1 趋势、paired delta 和训练吞吐–显存权衡图。
- [最终证据包](reports/evidence/cifar_resize_20260810/)：aggregate JSON / CSV / Markdown 及 SHA-256 manifest。
- 状态：accuracy `45/45`，efficiency `8/8`，checkpoint parity `30/30`（**PASS**）；λ4 CIFAR 预注册逐尺度子检查 **5/6**，顶层条件 **2/3**（严格 overall **FAIL**，归档 release **NO_GO**）。
- 研究结论：**ImageNet 规模预训练实验 GO**。建议用 λ4 与 matched DeiT-S/8 启动 300e 对照长训，并跟踪端到端 wall-clock；这不等同于 ImageNet 已验证。

锁定的 CIFAR 门禁字段与完整数值保持不变；ImageNet `GO` 是独立的下一阶段实验建议。
<!-- CIFAR_RESIZE_FINAL:END -->
