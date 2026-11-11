# WA-JEPA 短期任务代码

本目录只包含 **V4-1 短期目标到达实验**：PushT、TwoRoom、OGBench-Cube。网络结构、原训练损失及两条反馈路径均保留；V4-1 不使用 SIGReg。完整任务动作头、LIBERO、长期 AP 规划、其他架构变体、数据、权重和日志均未包含。

详细安装、数据格式、训练及评测命令见 [README.md](README.md)。在仓库根目录运行命令。训练/评测需要 Linux CUDA 环境，本地 Mac 可用于编辑和上传代码。

- 网络：`wa_jepa_v4.py`；实际执行策略：`wa_policy_v4.py`。
- PushT 训练：`train_pusht.py`，默认 batch 512、20 epochs。
- TwoRoom/Cube 训练：`train_tasks.py --task tworoom/cube`，默认 batch 128、10 epochs。
- 评测：`eval_pusht.py`、`eval_tasks.py`，默认 50 个案例、目标间隔 25 步、执行预算 50 步。
- 数据路径传给 `--dataset-root`，该路径下面应有 `datasets/` 子目录。
- DINO 源码和预训练权重通过 `DINO_REPO`、`DINO_CHECKPOINT` 指定，旧权重内的服务器路径可以被这两个变量覆盖。

原训练/评测脚本的计算逻辑已迁入独立入口，不再依赖旧版本网络、服务器历史日志或私有参考配置。保留了原实验的数据划分差异和随机数消耗；这些细节在英文 README 中有说明。全部输出请使用新目录，避免覆盖已有结果。

本目录可独立维护；`.gitignore` 已排除数据、权重、日志和缓存。
