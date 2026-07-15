---
name: experiment-report
description: 生成代理模型或 RL 训练的实验报告（单实验 / 多实验对比）
license: MIT
compatibility: opencode
---

## 功能

- 读取 TensorBoard 事件文件（`EventAccumulator`），提取训练/验证 loss、学习率等标量数据
- 单实验报告：训练曲线 + 测试集评估 + 预测 vs 真实对比图
- 多实验对比：loss 曲线叠加、收敛速度、最优超参比较
- 报告生成到 `experiment/` 目录，保存为中文 markdown + PNG 图表

## 使用时机

训练完成后或需要对比多组实验时使用。当用户要求"写实验报告"、"分析结果"或"对比实验"时，调用此 skill。

## 使用方法

```python
# 1. 读取 TensorBoard 标量数据
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("surrogate_output/s2p_s11_02")
ea.Reload()
tags = ea.Tags()["scalars"]      # ['eval/loss', 'train/lr', ...]
events = ea.Scalars("eval/loss")  # [ScalarEvent(step, value, wall_time), ...]
```

```bash
# 2. 运行评估脚本（单实验报告 + 对比图）
cd AI4EMDesign
uv run python3 scripts/evaluate_s11.py \
  --checkpoint surrogate_output/<run_name>/best_checkpoint.pt \
  --out experiment/<run_name>

# 输出：
# experiment/<run_name>/
#   ├── experiment_report.md      # 中文实验报告
#   ├── loss_curve.png             # 训练曲线
#   ├── sample_0000.png            # 样本对比图
#   └── ...
```

## 报告结构

单实验报告包含以下章节：
1. **实验目的** — 训练目标说明
2. **实验配置** — 模型、数据、超参
3. **训练过程** — loss 曲线 + 最优 epoch
4. **测试集评估** — MAE/RMSE/Median/Max
5. **样本可视化** — 10 个随机样本的预测 vs 真实对比图
6. **结论** — 模型效果总结

多实验对比报告额外包含：
- 多个实验的 loss 曲线叠加图
- 超参对比柱状图
- 收敛速度分析
- 最优配置推荐

## 依赖

- `tensorboard` (EventAccumulator)
- `matplotlib`
- `pandas`, `pyarrow`, `numpy`
- `torch`
