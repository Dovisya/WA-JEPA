# WA-JEPA

本仓库包含 **PushT、TwoRoom、OGBench-Cube 短期目标到达实验**，以及使用论文中 **VQ-BeT 动作头的 Push-T、LIBERO-Goal 完整任务代码**。完整任务的数据预处理、码本预训练、多卡训练、微调与评测命令见[完整任务指南](docs/fulltask.md)。长期 AP 规划代码、数据、权重与日志未包含。

## 网络框架

![WA-JEPA 网络框架：动作提案、未来预测与两条目标感知反馈路径](assets/figures/overview.png)

**图 1：WA-JEPA 总体框架。** 模型通过动作提案、未来预测与目标感知反馈，在执行前迭代修正同一动作计划。目标差异残差反馈指导逐 chunk 的动作调整，未来场景上下文反馈为动作修正提供更细致的预测场景信息。

## 短期任务实验

![TwoRoom、PushT 和 Cube 上的短期目标到达成功率](assets/figures/short_horizon_results.png)

**图 2：目标到达表现。** 在论文的短期评测设置下，WA-JEPA 在 TwoRoom、PushT 和 Cube 上分别取得 100%、96% 和 100% 的成功率。

![短期任务计算开销：完整计划延迟、计算量与参数量](assets/figures/computational_cost.png)

**图 3：NVIDIA A40 上的计算开销。** 完整计划生成延迟与神经网络计算量采用对数轴；参数量采用线性轴，并区分可训练参数与冻结参数。

## 完整任务实验

**表 2：完整任务性能与计算开销。** Push-T 汇报最终覆盖率，LIBERO-Goal 汇报任务成功率，均以百分比表示（均值 ± 已报告的标准差）。粗体标出各任务的最优值。

| 方法 | 动作头 | Push-T (%) ↑ | LIBERO-Goal (%) ↑ | Params. (M) ↓ | Mem. (MiB) ↓ | FLOPs (G) ↓ | Latency (ms) ↓ |
|:---|:---|---:|---:|---:|---:|---:|---:|
| WebSSL (CLS) | VQ-BeT | 59 ± 1 | 95 ± 1 | 332.7 / 308.0 | 665.0 / 647.3 | 162.5 / 324.6 | 19.20 / 17.49 |
| Patch Policy | VQ-BeT | 68 ± 3 | 94 ± 1 | 334.0 / 308.7 | 894.5 / 2552.6 | 255.2 / 413.5 | 26.49 / 81.35 |
| ResNet-18 Patch | ACT | 64 ± 3 | 93 ± 2 | 66.5 / 66.7 | 144.0 / 152.6 | **6.1** / **11.9** | **11.87** / **12.01** |
| OpenVLA-OFT | MLP | 59 ± 2 | 95 | 7608.4 / 7709.2 | 14837.7 / 15331.6 | 4202.6 / 8766.6 | 91.13 / 147.16 |
| Fast-WAM | Flow matching | — | 96 | 12406.3 | 23959.7 | 2545.1 | 438.53 |
| RynnVLA-002 | Regression | — | 96 | 7150.3 | 15429.3 | 16297.3 | 436.55 |
| DreamVLA | Diffusion | — | 86 | 734.0 | 1515.7 | 1001.8 | 85.77 |
| **WA-JEPA** | VQ-BeT | **71 ± 2** | **97 ± 1** | **37.2** / **37.8** | **92.2** / **115.6** | 13.5 / 26.5 | 27.21 / 26.95 |

计算开销的成对数值依次对应 **Push-T / LIBERO-Goal**；单个数值仅对应 **LIBERO-Goal**。“—”表示未报告结果。计算开销在 NVIDIA A40 上以 BF16、batch size 1 测量；显存为峰值已分配 GPU 显存，延迟为完成一次在线规划、生成一个动作 chunk 的耗时。完整任务训练与评测入口现已发布，见[完整任务指南](docs/fulltask.md)。

## 短期任务使用说明

详细安装、数据格式、训练及评测命令见 [README.md](README.md)。在仓库根目录运行命令。训练/评测需要 Linux CUDA 环境，本地 Mac 可用于编辑和上传代码。

- 网络：`wa_jepa.py`；实际执行策略：`wa_policy.py`。
- PushT 训练：`train_pusht.py`，默认 batch 512、20 epochs。
- TwoRoom/Cube 训练：`train_tasks.py --task tworoom/cube`，默认 batch 128、10 epochs。
- 评测：`eval_pusht.py`、`eval_tasks.py`，默认 50 个案例、目标间隔 25 步、执行预算 50 步。
- 数据路径传给 `--dataset-root`，该路径下面应有 `datasets/` 子目录。
- DINO 源码和预训练权重通过 `DINO_REPO`、`DINO_CHECKPOINT` 指定，旧权重内的服务器路径可以被这两个变量覆盖。

原训练/评测脚本的计算逻辑已迁入独立入口，不再依赖旧版本网络、服务器历史日志或私有参考配置。保留了原实验的数据划分差异和随机数消耗；这些细节在英文 README 中有说明。全部输出请使用新目录，避免覆盖已有结果。

本目录可独立维护；`.gitignore` 已排除数据、权重、日志和缓存。
