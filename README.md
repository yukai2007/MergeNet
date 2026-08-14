# MergeNet CV Release

This repository is the handoff surface for the MergeNet CV results and the
reproducible training materials.

- [ImageNet long-training handoff](deliverables/imagenet_longtrain_v1/): a
  ready-to-run, 300-epoch ImageNet training package for the DeiT baseline and
  MergeNet variants.
<!-- CIFAR_RESIZE_FINAL_ROOT_CAMPAIGN_STATUS:START -->
- [CIFAR resize validation](experiments/cifar_resize_20260810/): the completed 45-run accuracy / 8-card efficiency / 30-checkpoint parity campaign and its reproducible harness.
<!-- CIFAR_RESIZE_FINAL_ROOT_CAMPAIGN_STATUS:END -->
- [CV results report](reports/mergenet_cv_results_20260809.html): the HTML
  report summarizing the currently verified outcomes and delivery boundary.

Start with the ImageNet handoff README when launching large-scale training.

<!-- CIFAR_RESIZE_FINAL:START -->
## CIFAR resize 最终结果

- [最终 HTML 汇报](reports/mergenet_cifar_resize_final_20260814.html)：完整 5 resize × 3 model 精度、paired delta、8 卡效率和 30-checkpoint parity。
- [最终证据包](reports/evidence/cifar_resize_20260810/)：aggregate JSON / CSV / Markdown 及 SHA-256 manifest。
- 状态：accuracy `45/45`，efficiency `8/8`，checkpoint parity `30/30`（**PASS**）；λ4 预注册判定 **FAIL**；最终发布 **NO_GO**。

最终状态是对 CIFAR resize 预注册问题的完整回答，不代表 ImageNet 已训练或已验证。
<!-- CIFAR_RESIZE_FINAL:END -->
