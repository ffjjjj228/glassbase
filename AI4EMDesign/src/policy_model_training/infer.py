import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn


@dataclass
class TopKSample:
    reward: float
    topo_bits: torch.Tensor      # (T, row_bits) 0/1 long
    band_s: torch.Tensor         # (N_band,) float
    full_s: torch.Tensor         # (1200,) float


def _format_matrix_as_markdown(topo_2d: torch.Tensor) -> str:
    """
    把 2D tensor 格式化成 markdown 代码块，便于阅读。
    """
    topo_cpu = topo_2d.detach().cpu()
    lines = []
    for r in topo_cpu:
        lines.append(" ".join(str(int(x)) for x in r.tolist()))
    return "```text\n" + "\n".join(lines) + "\n```"


def _format_vector_as_markdown(vec_1d: torch.Tensor, max_items: Optional[int] = None) -> str:
    """
    把 1D tensor 格式化成 markdown 代码块（vector 很长时可截断显示）。
    """
    v = vec_1d.detach().cpu().flatten()
    if max_items is not None and v.numel() > max_items:
        head = v[:max_items].tolist()
        text = ", ".join(f"{x:.6f}" for x in head) + f", ... (total={v.numel()})"
    else:
        text = ", ".join(f"{x:.6f}" for x in v.tolist())
    return "```text\n[" + text + "]\n```"


def export_topk_policy_samples_to_markdown(
    policy_model: nn.Module,
    surrogate_model: nn.Module,
    reward_fn,
    device: str,
    out_md_path: str,
    # sampling config
    n_samples: int = 100,
    batch_size: int = 16,
    top_k: int = 10,
    # topology -> surrogate config（与你训练一致）
    topology_scale_m11: bool = True,       # True: topo*2-1, False: 0/1
    topology_unsqueeze_channel: bool = True,  # True: (B,1,H,W)
    # band extraction（优先从 reward_fn 读取 band 信息）
    band_ghz: Optional[Tuple[float, float]] = (5.0, 6.0),
    # vector display
    max_band_points_to_print: Optional[int] = None,  # 例如 200，避免 markdown 过长；None 表示全打印
):
    """
    纯推理：采样 n_samples 个拓扑，评估 reward，选 top_k 写入 markdown。

    Markdown 内容（每个样本）：
    - Topology 矩阵
    - 5-6 GHz（或指定 band）频段上的预测 S 参数向量
    - reward 数值

    注意：
    - policy_model：保持 train()/eval() 均可，但这里建议 eval() 以减少随机层干扰（若有 dropout）
    - surrogate_model：eval() + no_grad()
    """
    os.makedirs(os.path.dirname(out_md_path) or ".", exist_ok=True)

    policy_model.eval()
    surrogate_model.eval()
    for p in surrogate_model.parameters():
        p.requires_grad_(False)

    # ---------- 解析 band index：优先从 reward_fn 里取 ----------
    # 兼容你目前单频段版 BandThresholdReward：有 self.band / self.band_idx
    band_idx = None
    if hasattr(reward_fn, "band_idx"):
        band_idx = reward_fn.band_idx
    elif hasattr(reward_fn, "band_indices") and isinstance(reward_fn.band_indices, (list, tuple)) and len(reward_fn.band_indices) > 0:
        band_idx = reward_fn.band_indices[0]

    # 如果 reward_fn 没有索引，我们用 band_ghz + 频率轴重建（需要知道 2~20GHz & 1000点）
    freqs = None
    if band_idx is None:
        if band_ghz is None:
            raise ValueError("reward_fn 不含 band 索引时，必须显式传入 band_ghz=(f0,f1)")
        # 兼容 reward_fn 存了 freqs/num_points
        if hasattr(reward_fn, "freqs"):
            freqs = reward_fn.freqs
        else:
            # default: 60 points from metadata (overridden by reward_fn if available)
            freqs = torch.linspace(0.1, 29.6, 60)

        f0, f1 = float(band_ghz[0]), float(band_ghz[1])
        freqs = freqs.to(device)
        mask = (freqs >= f0) & (freqs <= f1)
        band_idx = torch.where(mask)[0]
        if band_idx.numel() == 0:
            raise ValueError(f"band_ghz={band_ghz} 不在频率轴范围内")

    # 确保 band_idx 在 device 上
    band_idx = band_idx.to(device)

    # ---------- 开始采样累计 ----------
    collected: List[TopKSample] = []
    remaining = n_samples

    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)

            # policy 采样
            _, topo_bits, logp_sum, entropy_sum = policy_model(batch_size=bs)
            topo_bits = topo_bits.to(device)

            # topo_bits: (B, 22, 22) -> (B, 1, 22, 22), scale to [-1, 1]
            topo = topo_bits.float().unsqueeze(1)
            if topology_scale_m11:
                topo = topo * 2.0 - 1.0
            topo_in = topo if topology_unsqueeze_channel else topo.squeeze(1)

            # surrogate 预测
            pred_s = surrogate_model(topo_in)   # 期望 (B,1200)

            if pred_s.ndim != 2:
                raise ValueError(f"surrogate_model 输出应为 (B,1200)，当前 pred_s.shape={tuple(pred_s.shape)}")

            # reward
            rewards = reward_fn(pred_s)         # (B,)
            if rewards.ndim != 1 or rewards.shape[0] != bs:
                raise ValueError(f"reward_fn 输出应为 (B,)，当前 rewards.shape={tuple(rewards.shape)}")

            # 取频段曲线
            band_s = pred_s[:, band_idx]        # (B, N_band)

            # 存样本（逐个存，方便 TopK 排序）
            for i in range(bs):
                collected.append(
                    TopKSample(
                        reward=float(rewards[i].item()),
                        topo_bits=topo_bits[i].detach().cpu(),  # 存 CPU，避免占 GPU
                        band_s=band_s[i].detach().cpu(),
                        full_s=pred_s[i].detach().cpu(),
                    )
                )

            remaining -= bs

    # ---------- TopK ----------
    collected.sort(key=lambda x: x.reward, reverse=True)
    top_samples = collected[:top_k]

    # ---------- 写 Markdown ----------
    lines: List[str] = []
    lines.append("# Top-K Antenna Designs (Policy Inference)")
    lines.append("")
    lines.append(f"- Total samples evaluated: **{n_samples}**")
    lines.append(f"- Top-K selected: **{top_k}**")
    lines.append("")

    # band 文案
    if hasattr(reward_fn, "band"):
        f0, f1 = reward_fn.band
        lines.append(f"- Band used for export: **{f0:.3f}–{f1:.3f} GHz**")
    elif band_ghz is not None:
        lines.append(f"- Band used for export: **{band_ghz[0]:.3f}–{band_ghz[1]:.3f} GHz**")
    lines.append("")

    for rank, s in enumerate(top_samples, start=1):
        topo_2d = s.topo_bits  # (T,row_bits) long 0/1
        lines.append(f"## #{rank} (reward = {s.reward:.8f})")
        lines.append("")
        lines.append("### Topology")
        lines.append(_format_matrix_as_markdown(topo_2d))
        lines.append("")
        lines.append("### Predicted S-curve (band)")
        lines.append(_format_vector_as_markdown(s.band_s, max_items=max_band_points_to_print))
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return top_samples
