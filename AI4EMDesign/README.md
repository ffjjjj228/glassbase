# AI4EMDesign

智能电磁设计工具，结合**仿真代理模型（Surrogate Model）** + **强化学习（RL）** 进行天线拓扑结构的自动优化。

## 目录结构

```
AI4EMDesign/
├── conf/                          # 配置文件（HOCON 格式）
│   ├── surrogate.conf             # 代理模型训练配置
│   └── rl.conf                    # RL 强化学习训练配置
├── data/                          # 数据目录
│   ├── topology_s4p_toponly.parquet  # 原始数据集（22×22 拓扑 + S 参数）
│   ├── train_s4p_proc.parquet     # 预处理后的训练集
│   ├── val_s4p_proc.parquet       # 预处理后的验证集
│   ├── preprocess_s4p.py          # 数据预处理脚本（切分 + 提取 S 曲线）
│   ├── split_topology.py          # （旧）简单切分脚本
│   ├── analysze.py                # S 参数分布分析工具
│   └── read_example.py            # Parquet 数据读取示例
├── src/                           # 源代码
│   ├── surrogate_model_training/  # 代理模型训练
│   │   ├── main.py                # 训练入口
│   │   └── engine.py              # train/eval 循环
│   ├── policy_model_training/     # 强化学习训练
│   │   ├── main.py                # 训练入口
│   │   ├── engine.py              # GRPO 训练循环
│   │   └── infer.py               # 推理 & Top-K 导出
│   ├── models/                    # 模型定义
│   │   ├── resnet.py              # EMResnet（代理模型）
│   │   └── policy.py              # Policy 模型（拓扑生成策略）
│   ├── data/                      # 数据加载
│   │   └── parquet_dataset.py     # Parquet 数据集 & DataLoader
│   ├── reward/                    # 奖励函数
│   │   └── band_rew.py            # 频段阈值奖励（BandThresholdReward）
│   ├── config/                    # 参数解析
│   │   └── __init__.py            # Args, 自动输出目录命名
│   ├── loss/                      # 损失函数注册
│   ├── optimizer/                 # 优化器注册
│   ├── scheduler/                 # 学习率调度器注册
│   ├── registry.py                # 全局注册表
│   └── torchutils/                # PyTorch 工具库
│       ├── registry.py            # Registry 基类
│       ├── distributed.py         # 分布式训练工具
│       ├── logging.py             # 日志
│       ├── metrics.py             # 指标统计
│       └── serialization.py       # 序列化工具
├── output/                        # 训练输出目录（自动命名）
├── tests/                         # 测试
│   └── test_parquet_dataset.py    # 数据集 & DataLoader 测试
├── PROBLEM.md                     # 数据分布问题分析 & 解决思路
└── pyproject.toml                 # 项目配置
```

## 环境配置

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install pyhocon tensorboard pandas pyarrow numpy
```

## 工作流程

```
 原始 S4P 数据 ──→ 预处理 ──→ 代理模型训练 ──→ RL 策略优化 ──→ Top-K 设计导出
(topology_s4p_toponly.parquet)   (surrogate_XX)   (rl_XX)        (topk_samples.md)
```

### 1. 数据预处理

原始数据为 `topology_s4p_toponly.parquet`，包含 5000+ 条天线拓扑及其全波仿真 S 参数。

```bash
python data/preprocess_s4p.py
```

输出：
- `data/train_s4p_proc.parquet` — 训练集（80%）
- `data/val_s4p_proc.parquet` — 验证集（20%）

每行包含：
- `topology_top_flat` — 22×22 拓扑矩阵（展平为 484 int, int8）
- `s_curve` — 60 频点 × 10 S 参数分量 = 600 实部 + 600 虚部 = 1200 维向量（float32）

### 2. 代理模型训练

训练一个 ResNet 模型作为电磁仿真器的代理：

```bash
python -m src.surrogate_model_training.main --conf conf/surrogate.conf
```

- **模型**: EMResnet（CifarResNet + BasicBlock, ~0.25M 参数）
- **输入**: (B, 1, 22, 22) 拓扑，缩放到 [-1, 1]
- **输出**: (B, 1200) S 参数曲线
- **损失**: L1Loss
- **优化器**: AdamW（lr=1e-3, betas=[0.9, 0.95]）
- **调度器**: CosineAnnealingLR（T_max=90, eta_min=1e-5）
- **Epochs**: 90, **batch_size**: 32
- **输出**: `output/surrogate_01/`, `output/surrogate_02/`, ...（自动递增命名）

也可手动指定输出目录：

```bash
python -m src.surrogate_model_training.main --conf conf/surrogate.conf -o output/my_model
```

### 3. 强化学习训练（GRPO）

基于代理模型，用 GRPO 算法优化天线拓扑，使目标频段 S 参数最小：

```bash
python -m src.policy_model_training.main --conf conf/rl.conf
```

- **Policy 模型**: `AntennaTopologyPolicyModel`
  - 边框策略（4 步，每步选 22 个位置之一）
  - 内部策略（400 步，每步决定 0/1）
- **代理模型**: 加载 `output/surrogate_01/best_checkpoint.pt`（冻结）
- **算法**: GRPO（Group Relative Policy Optimization）
  - 每个 batch 作为一个 group，用 group 内相对 advantage 更新
  - entropy 正则化（coef=0.001）
- **Reward**: `BandThresholdReward`
  - 目标频段 5–6 GHz
  - 超过阈值（0.1）的部分给惩罚
  - 完全满足则给 bonus
- **Episodes**: 10000, **batch_size**: 32
- **输出**: `output/rl_01/`, `output/rl_02/`, ...

### 4. 结果查看

导出 Top-K 设计样本：

```bash
# 训练完成后自动导出到 output/rl_XX/topk_samples.md
```

使用 TensorBoard 可视化训练过程：

```bash
tensorboard --logdir=output/
```

打开 `http://localhost:6006` 查看。

## 测试

```bash
python -m pytest tests/
```

## 自定义配置

配置文件使用 [HOCON](https://github.com/chimpler/pyhocon) 格式。可通过 `-M` 参数覆盖配置：

```bash
# 修改代理模型加载路径
python -m src.policy_model_training.main --conf conf/rl.conf \
    -M surrogate_model.load_from=output/surrogate_02/best_checkpoint.pt

# 修改训练轮数
python -m src.surrogate_model_training.main --conf conf/surrogate.conf \
    -M max_epochs=120
```
