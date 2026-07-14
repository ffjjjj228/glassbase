import os
import sys
import argparse
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

UNIQUE_IDX = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]
N_FREQ = 60
FREQ_START = 0.1
FREQ_END = 29.6


def load_model(checkpoint_path: str, device: str):
    from src.models.resnet import CifarResNet, BasicBlock
    model = CifarResNet(BasicBlock, [2, 2, 2], in_channels=1, out_dim=1200)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def preprocess_topo(topo_flat: list | np.ndarray) -> torch.Tensor:
    arr = np.asarray(topo_flat, dtype=np.float32).reshape(22, 22)
    t = torch.from_numpy(arr)
    t = t.unsqueeze(0)
    t = t * 2.0 - 1.0
    t = t.unsqueeze(0)
    return t


def decode_to_smatrix(pred_1200: np.ndarray, n_freq: int = N_FREQ) -> np.ndarray:
    real_part = pred_1200[:600].reshape(n_freq, 10)
    imag_part = pred_1200[600:].reshape(n_freq, 10)
    cmplx = real_part + 1j * imag_part
    smat = np.zeros((n_freq, 4, 4), dtype=complex)
    for f in range(n_freq):
        for i, idx in enumerate(UNIQUE_IDX):
            r = idx // 4
            c = idx % 4
            smat[f, r, c] = cmplx[f, i]
            smat[f, c, r] = cmplx[f, i]
    return smat


def plot_sample(axs, freqs: np.ndarray, pred: np.ndarray, sim: np.ndarray,
                sample_idx: int, title: str):
    s_labels = [f"S{r+1}{c+1}" for r in range(4) for c in range(4)]
    for idx, ax in enumerate(axs.flat):
        r = idx // 4
        c = idx % 4
        p_mag = np.clip(np.abs(pred[:, r, c]), None, 1.0)
        s_mag = np.clip(np.abs(sim[:, r, c]), None, 1.0)
        ax.plot(freqs, p_mag, "b-", lw=1.2, label="Pred")
        ax.plot(freqs, s_mag, "r--", lw=1.2, label="Sim")
        ax.set_title(s_labels[idx], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if r == 3:
            ax.set_xlabel("Frequency (GHz)")
        if c == 0:
            ax.set_ylabel("|S|")
    axs[0, 0].legend(fontsize=8)
    fig = axs[0, 0].figure
    fig.suptitle(title, fontsize=13)


def compute_errors(pred_smat: np.ndarray, sim_smat: np.ndarray) -> dict:
    errors = {}
    for r in range(4):
        for c in range(4):
            key = f"S{r+1}{c+1}"
            p = np.clip(np.abs(pred_smat[:, r, c]), None, 1.0)
            s = np.clip(np.abs(sim_smat[:, r, c]), None, 1.0)
            abs_err = np.abs(p - s)
            errors[key] = {
                "mae": float(abs_err.mean()),
                "rmse": float(np.sqrt((abs_err ** 2).mean())),
                "max": float(abs_err.max()),
                "median": float(np.median(abs_err)),
            }
    return errors


def main():
    parser = argparse.ArgumentParser("Evaluate surrogate model on test set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-set", default="data/test_s4p_proc.parquet")
    parser.add_argument("--out", default="report/test_evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit test samples (for quick test)")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    print(f"Loading test set from {args.test_set} ...")
    df = pd.read_parquet(args.test_set)
    if args.max_samples:
        df = df.iloc[:args.max_samples]
    print(f"Test samples: {len(df)}")

    freqs = np.linspace(FREQ_START, FREQ_END, N_FREQ)

    all_errors = []

    for idx, row in df.iterrows():
        topo_flat = row["topology_top_flat"]
        s_curve = np.asarray(row["s_curve"], dtype=np.float32)

        topo_tensor = preprocess_topo(topo_flat)
        with torch.no_grad():
            pred = model(topo_tensor.to(device))
        pred_1200 = pred.cpu().numpy().flatten()

        pred_smat = decode_to_smatrix(pred_1200)
        sim_smat = decode_to_smatrix(s_curve)

        errors = compute_errors(pred_smat, sim_smat)
        all_errors.append(errors)

        if idx < 10:
            fig, axs = plt.subplots(4, 4, figsize=(16, 12))
            plot_sample(axs, freqs, pred_smat, sim_smat, idx,
                        f"Sample #{idx} | Total MAE: {sum(e['mae'] for e in errors.values()) / 16:.6f}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"sample_{idx:04d}.png"), dpi=130)
            plt.close(fig)

        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            print(f"  processed {idx + 1}/{len(df)}")

    agg = {}
    for key in all_errors[0]:
        agg[key] = {}
        for metric in ["mae", "rmse", "max", "median"]:
            vals = [e[key][metric] for e in all_errors]
            agg[key][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    overall = {
        m: float(np.mean([agg[k][m]["mean"] for k in agg]))
        for m in ["mae", "rmse", "max", "median"]
    }

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# Test Set Evaluation Report\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Test samples: {len(df)}\n")
        f.write(f"- Freq range: {FREQ_START} ~ {FREQ_END} GHz ({N_FREQ} points)\n\n")

        f.write("## Overall Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        for m in ["mae", "rmse", "median", "max"]:
            f.write(f"| {m.upper()} | {overall[m]:.6f} |\n")
        f.write("\n")

        f.write("## Per S-Parameter Metrics (mean ± std)\n\n")
        f.write("| S-param | MAE | RMSE | Median | Max |\n")
        f.write("|---------|-----|------|--------|-----|\n")
        for key in ["S11", "S12", "S13", "S14",
                     "S21", "S22", "S23", "S24",
                     "S31", "S32", "S33", "S34",
                     "S41", "S42", "S43", "S44"]:
            a = agg[key]
            mae = f"{a['mae']['mean']:.6f} ± {a['mae']['std']:.6f}"
            rmse = f"{a['rmse']['mean']:.6f} ± {a['rmse']['std']:.6f}"
            med = f"{a['median']['mean']:.6f} ± {a['median']['std']:.6f}"
            mx = f"{a['max']['mean']:.6f} ± {a['max']['std']:.6f}"
            f.write(f"| {key} | {mae} | {rmse} | {med} | {mx} |\n")

    print(f"\nReport saved to {out_dir}/")
    print(f"Overall MAE: {overall['mae']:.6f}")
    print(f"Overall RMSE: {overall['rmse']:.6f}")


if __name__ == "__main__":
    main()
