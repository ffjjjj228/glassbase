# 数据集说明

## 数据来源

原始数据来自 HFSS 全波仿真的 .s4p 文件（4 端口 Touchstone 格式），
每个样本包含一个 22×22 的拓扑结构及其对应的 4×4 S 参数矩阵（60 个频点）。

## 当前数据集文件

| 文件 | 说明 | 比例 |
|------|------|------|
| `topology_s4p_toponly.parquet` | 原始完整数据集 | 100% |
| `train_s4p_proc.parquet` | 处理后训练集 | 80% |
| `val_s4p_proc.parquet` | 处理后验证集 | 10% |
| `test_s4p_proc.parquet` | 处理后测试集 | 10% |

## 数据格式

每行包含两个字段：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `topology_top_flat` | (484,) → reshape to (22, 22) | int8 | 二值拓扑矩阵（0=空，1=金属） |
| `s_curve` | (1200,) | float32 | S 参数曲线 |

### S 参数排列

1200 维向量结构：

```
60 频点（0.1 ~ 29.6 GHz，步长 0.5 GHz）
× 10 个上三角 4×4 S 矩阵元素（利用互易性，S_ij = S_ji）
× 2（实部 + 虚部）
= 60 × 10 × 2 = 1200
```

10 个上三角索引（行主序）：
```
[S11, S12, S13, S14, S22, S23, S24, S33, S34, S44]
```

排列方式：前 600 维为实部，后 600 维为虚部。
每个频点内按上三角索引顺序排列。

## 预处理流程

```bash
python data/preprocess_s4p.py
```

处理步骤：
1. 读取 `topology_s4p_toponly.parquet`
2. 随机打乱（固定种子 `random_state=42`，保证可复现）
3. 按 8:1:1 切分为训练集、验证集、测试集
4. 从原始 `s_real` / `s_imag` 中提取上三角 S 参数，拼接为 1200 维 `s_curve`
5. 验证边框约束（每边恰好 1 个金属单元）
6. 保存为 zstd 压缩的 Parquet 文件

## 测试集评估

训练完成后，可用测试集评估代理模型精度：

```bash
python scripts/evaluate_test_set.py \
    --checkpoint output/surrogate_lr0.01/best_checkpoint.pt \
    --test-set data/test_s4p_proc.parquet \
    --out report/test_evaluation
```

输出：
- `summary.md` — 各 S 参数的误差统计（MAE / RMSE / Median / Max）
- `sample_XXXX.png` — 前 10 个样本的预测 vs 仿真对比图
