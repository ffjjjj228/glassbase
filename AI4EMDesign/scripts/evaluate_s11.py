import os, sys, argparse, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N_FREQ = 60
FREQ_START = 0.1
FREQ_END = 29.6
FREQS = np.linspace(FREQ_START, FREQ_END, N_FREQ)


def load_model(checkpoint_path: str, device: str):
    from src.models.resnet import EMResnet
    model = EMResnet(layers=[2, 2, 2], in_channels=1, out_dim=60)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def preprocess_topo(topo_flat) -> torch.Tensor:
    arr = np.asarray(topo_flat, dtype=np.float32).reshape(22, 22)
    t = torch.from_numpy(arr).unsqueeze(0)
    t = t * 2.0 - 1.0
    t = t.unsqueeze(0)
    return t


def read_tensorboard_scalars(event_dir: str, tag: str):
    ev = EventAccumulator(event_dir)
    ev.Reload()
    if tag in ev.Tags().get("scalars", []):
        return [(e.step, e.value) for e in ev.Scalars(tag)]
    return []


def plot_sample(freqs, pred, true, save_path, title=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, pred, "b-", lw=1.5, label="Predicted")
    ax.plot(freqs, true, "r--", lw=1.5, label="True (simulation)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S11|")
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_loss_curve(train_steps_values, val_steps_values, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if train_steps_values:
        steps, vals = zip(*train_steps_values)
        ax.plot(steps, vals, "b-", lw=1.2, label="Train loss")
    if val_steps_values:
        steps, vals = zip(*val_steps_values)
        ax.plot(steps, vals, "r-", lw=1.2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Evaluate S11 surrogate model and generate experiment report")
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt")
    parser.add_argument("--test-set", default="data/test_s2p.parquet")
    parser.add_argument("--train-set", default="data/train_s2p.parquet")
    parser.add_argument("--out", default=None, help="Output directory (default: experiment/<run_name>)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-plot-samples", type=int, default=10)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    run_name = os.path.basename(checkpoint_dir)
    out_dir = args.out or os.path.join(os.path.dirname(checkpoint_dir), "..", "experiment", run_name)
    out_dir = os.path.abspath(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    print(f"Loading test set from {args.test_set} ...")
    df = pd.read_parquet(args.test_set)
    if args.max_samples:
        df = df.iloc[:args.max_samples]
    print(f"Test samples: {len(df)}")

    all_mae = []
    indices = list(range(len(df)))
    plot_indices = set(random.sample(indices, min(args.n_plot_samples, len(df))))

    for idx, row in df.iterrows():
        topo_flat = row["topology_top_flat"]
        true_mag = np.asarray(row["s11_mag"], dtype=np.float32)

        topo_tensor = preprocess_topo(topo_flat)
        with torch.no_grad():
            pred = model(topo_tensor.to(device))
        pred_mag = pred.cpu().numpy().flatten()

        mae = float(np.abs(pred_mag - true_mag).mean())
        all_mae.append(mae)

        if idx in plot_indices:
            save_path = os.path.join(plot_dir, f"sample_{idx:04d}.png")
            plot_sample(
                FREQS, pred_mag, true_mag, save_path,
                title=f"Sample #{idx} | MAE={mae:.6f}",
            )
            print(f"  saved plot: sample_{idx:04d}.png (MAE={mae:.6f})")

        if (idx + 1) % 200 == 0 or idx == len(df) - 1:
            print(f"  processed {idx + 1}/{len(df)}")

    overall_mae = float(np.mean(all_mae))
    overall_rmse = float(np.sqrt(np.mean(np.array(all_mae) ** 2)))
    overall_median = float(np.median(all_mae))
    overall_max = float(np.max(all_mae))

    print(f"\nTest set results ({len(df)} samples):")
    print(f"  MAE:    {overall_mae:.6f}")
    print(f"  RMSE:   {overall_rmse:.6f}")
    print(f"  Median: {overall_median:.6f}")
    print(f"  Max:    {overall_max:.6f}")

    train_loss = read_tensorboard_scalars(checkpoint_dir, "train/loss")
    val_loss = read_tensorboard_scalars(checkpoint_dir, "eval/loss")
    lrs = read_tensorboard_scalars(checkpoint_dir, "train/lr")

    loss_curve_path = os.path.join(plot_dir, "loss_curve.png")
    plot_loss_curve(train_loss, val_loss, loss_curve_path)
    print(f"  saved loss curve: loss_curve.png")

    n_train = len(pd.read_parquet(args.train_set)) if os.path.exists(args.train_set) else "?"

    best_val = min(v for _, v in val_loss) if val_loss else "?"
    final_val = val_loss[-1][1] if val_loss else "?"
    best_epoch = val_loss[val_loss.index(min(val_loss, key=lambda x: x[1]))][0] if val_loss else "?"

    report_path = os.path.join(out_dir, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write("# 实验报告\n\n")

        f.write("## 1. 实验目的\n\n")
        f.write("训练代理模型（Surrogate Model），从 22×22 的二值拓扑结构预测 S11 幅度（`|S11|`）。\n\n")

        f.write("## 2. 实验配置\n\n")
        f.write("| 项目 | 值 |\n")
        f.write("|------|-----|\n")
        f.write("| 模型 | EMResnet (CifarResNet, layers=[2,2,2]) |\n")
        f.write("| 输入 | 22×22 拓扑（1 通道，缩放到 [-1, 1]） |\n")
        f.write("| 输出 | S11 幅度，60 个频点（0.1 ~ 29.6 GHz） |\n")
        f.write("| 损失函数 | L1 |\n")
        f.write("| 优化器 | AdamW (lr=0.001, betas=[0.9,0.95], wd=1e-5) |\n")
        f.write("| 学习率调度 | CosineAnnealingLR (T_max=200, eta_min=1e-5) |\n")
        f.write("| 训练轮数 | 200 |\n")
        f.write("| 训练集 | {} 条 |\n".format(n_train))
        f.write("| 测试集 | {} 条 |\n".format(len(df)))
        f.write("| 配置文件 | `conf/surrogate_s11.conf` |\n")
        f.write("| 检查点 | `{}` |\n\n".format(os.path.basename(args.checkpoint)))

        f.write("## 3. 训练过程\n\n")
        f.write("![Loss 曲线](plots/loss_curve.png)\n\n")
        f.write("- 最优验证损失：**{:.6f}**（epoch {}) 。\n".format(best_val, best_epoch))
        f.write("- 最终验证损失：{:.6f}\n".format(final_val))
        if lrs:
            f.write("- 最终学习率：{:.6f}\n\n".format(lrs[-1][1]))

        f.write("## 4. 测试集评估\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write("| MAE | {:.6f} |\n".format(overall_mae))
        f.write("| RMSE | {:.6f} |\n".format(overall_rmse))
        f.write("| 中位数 | {:.6f} |\n".format(overall_median))
        f.write("| 最大值 | {:.6f} |\n\n".format(overall_max))

        f.write("## 5. 样本预测对比\n\n")
        f.write("以下为 {} 个随机测试样本的预测值（蓝色实线）与仿真值（红色虚线）对比：\n\n".format(len(plot_indices)))
        sorted_plots = sorted(plot_indices)
        for i, idx in enumerate(sorted_plots):
            f.write("### 样本 #{}\n\n".format(idx))
            f.write("![]({})\n\n".format("plots/sample_{:04d}.png".format(idx)))
            if i < len(sorted_plots) - 1:
                f.write("---\n\n")

        f.write("## 6. 结论\n\n")
        f.write("代理模型在测试集上达到 **MAE={:.6f}**、**RMSE={:.6f}**".format(overall_mae, overall_rmse))
        f.write("（S11 幅度范围为 [0, 1]）。")
        if overall_mae < 0.01:
            f.write("模型能够高精度地近似电磁仿真结果。")
        elif overall_mae < 0.03:
            f.write("模型能够较好地近似电磁仿真结果。")
        else:
            f.write("模型近似精度一般，仍有改进空间。")
        f.write("\n")

    print(f"\nExperiment report saved to {report_path}")


if __name__ == "__main__":
    main()
