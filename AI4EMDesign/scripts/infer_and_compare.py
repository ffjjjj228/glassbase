import os
import sys
import argparse
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

UNIQUE_IDX = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]
N_FREQ = 60
FREQ_START = 0.1
FREQ_END = 29.6


def parse_topology(text: str) -> np.ndarray:
    lines = text.strip().splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if len(line) != 22:
            raise ValueError(f"每行需要 22 个字符，得到 {len(line)}: {line}")
        rows.append([int(c) for c in line])
    arr = np.array(rows, dtype=np.int8)
    if arr.shape != (22, 22):
        raise ValueError(f"拓扑矩阵应为 (22,22)，得到 {arr.shape}")
    return arr


def preprocess_topo(topo: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(topo.astype(np.float32))
    t = t.unsqueeze(0)
    t = t * 2.0 - 1.0
    t = t.unsqueeze(0)
    return t


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


def infer(model, topo_tensor: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        pred = model(topo_tensor.to(device))
    return pred.cpu().numpy().flatten()


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


def parse_s4p(path: str):
    with open(path) as f:
        raw = f.read()

    fmt = "MA"
    for line in raw.splitlines():
        if line.startswith("#"):
            parts = line.upper().split()
            if "MA" in parts:
                fmt = "MA"
            elif "RI" in parts:
                fmt = "RI"
            break

    lines = raw.splitlines()
    freq_list = []
    smat_list = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("!"):
            i += 1
            continue
        tokens = line.split()
        if not tokens:
            i += 1
            continue

        try:
            freq = float(tokens[0])
        except ValueError:
            i += 1
            continue

        row_data = [float(t) for t in tokens[1:]]
        block = [row_data]
        for j in range(1, 4):
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("!")):
                i += 1
            rline = lines[i].strip() if i < len(lines) else ""
            rtokens = rline.split() if rline else []
            row_data = []
            for t in rtokens:
                try:
                    row_data.append(float(t))
                except ValueError:
                    pass
            block.append(row_data)

        vals = [v for row in block for v in row]
        if fmt == "MA":
            smat = np.zeros((4, 4), dtype=complex)
            for r in range(4):
                for c in range(4):
                    idx = (r * 4 + c) * 2
                    if idx + 1 < len(vals):
                        mag = vals[idx]
                        ang_deg = vals[idx + 1]
                        ang_rad = math.radians(ang_deg)
                        smat[c, r] = mag * (math.cos(ang_rad) + 1j * math.sin(ang_rad))
            smat_list.append(smat)
            freq_list.append(freq)
        elif fmt == "RI":
            smat = np.zeros((4, 4), dtype=complex)
            for r in range(4):
                for c in range(4):
                    idx = (r * 4 + c) * 2
                    if idx + 1 < len(vals):
                        smat[c, r] = vals[idx] + 1j * vals[idx + 1]
            smat_list.append(smat)
            freq_list.append(freq)
        i += 1

    return np.array(freq_list), np.array(smat_list)


def interpolate_to_60(sim_freqs: np.ndarray, sim_smat: np.ndarray) -> np.ndarray:
    target_freqs = np.linspace(FREQ_START, FREQ_END, N_FREQ)
    result = np.zeros((N_FREQ, 4, 4), dtype=complex)
    for r in range(4):
        for c in range(4):
            real_interp = np.interp(target_freqs, sim_freqs, sim_smat[:, r, c].real)
            imag_interp = np.interp(target_freqs, sim_freqs, sim_smat[:, r, c].imag)
            result[:, r, c] = real_interp + 1j * imag_interp
    return result


def to_db(val: np.ndarray) -> np.ndarray:
    mag = np.clip(np.abs(val), None, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20 * np.log10(mag)
        db[~np.isfinite(db)] = -60
    return db


def plot_comparison(freqs: np.ndarray, pred_smat: np.ndarray, sim_smat: np.ndarray | None,
                    out_path: str, mode: str = "mag"):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("S-Parameter Comparison: Prediction vs Simulation", fontsize=16)

    labels = [f"S{r+1}{c+1}" for r in range(4) for c in range(4)]

    for idx, ax in enumerate(axes.flat):
        r = idx // 4
        c = idx % 4
        pred_vals = np.clip(np.abs(pred_smat[:, r, c]), None, 1.0)
        if mode == "db":
            pred_vals = to_db(pred_smat[:, r, c])

        ax.plot(freqs, pred_vals, "b-", linewidth=1.5, label="Prediction")

        if sim_smat is not None:
            sim_vals = np.clip(np.abs(sim_smat[:, r, c]), None, 1.0)
            if mode == "db":
                sim_vals = to_db(sim_smat[:, r, c])
            ax.plot(freqs, sim_vals, "r--", linewidth=1.5, label="Simulation (.s4p)")

        ax.set_title(labels[idx], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if mode == "db":
            ax.set_ylabel("dB")
        else:
            ax.set_ylabel("|S| (linear)")
        if r == 3:
            ax.set_xlabel("Frequency (GHz)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Infer S-parameters from topology and compare with .s4p")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pt path")
    parser.add_argument("--topology", required=True, help="22x22 topology file path, or '-' for stdin")
    parser.add_argument("--s4p", default=None, help="Ground truth .s4p file path")
    parser.add_argument("--out", default="comparison.png", help="Output plot path")
    parser.add_argument("--mode", choices=["mag", "db"], default="mag", help="Plot mode: linear magnitude or dB")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    if args.topology == "-":
        topo_text = sys.stdin.read()
    else:
        with open(args.topology) as f:
            topo_text = f.read()

    topo = parse_topology(topo_text)
    topo_tensor = preprocess_topo(topo)
    print(f"Topology shape: {topo.shape}")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    pred_1200 = infer(model, topo_tensor, device)
    pred_smat = decode_to_smatrix(pred_1200)
    print(f"Predicted S-matrix: {pred_smat.shape}")

    sim_smat = None
    if args.s4p:
        sim_freqs, raw_smat = parse_s4p(args.s4p)
        sim_smat = interpolate_to_60(sim_freqs, raw_smat)
        print(f"Simulation S-matrix: {sim_smat.shape} (interpolated from {len(sim_freqs)} freq points)")

    freqs = np.linspace(FREQ_START, FREQ_END, N_FREQ)
    plot_comparison(freqs, pred_smat, sim_smat, args.out, args.mode)


if __name__ == "__main__":
    main()
