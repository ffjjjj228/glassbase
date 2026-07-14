import torch
from typing import Sequence, Optional
from src.registry import REWARD


@REWARD.register
class BandThresholdReward:
    """
    单频段阈值约束的 S 参数 reward 计算器（面向强化学习/GRPO 训练）。

    约定：
    - 频率范围：2~20 GHz（默认）
    - 采样点数：1000（默认）
    - 输入 s_curve：线性幅度，范围 [0,1]，shape=(B, num_points)

    Reward：
    - 频段内 S11 超过阈值的部分给连续惩罚（默认平方惩罚）
    - 可选：若频段内所有点均满足 S11<=threshold，给一个 bonus（减少 reward 稀疏）
    """

    def __init__(
        self,
        bands: Sequence[float],          # [f_start, f_end] 或 (f_start, f_end)
        threshold: float,                # 单个阈值
        weight: float,             # 单频段权重（保留可扩展性）
        # frequency axis
        freq_start_ghz: float,
        freq_end_ghz: float,
        num_points: int,
        # penalty
        penalty_power: float,
        # RL-friendly
        satisfied_bonus: float,
        # numerical
        clamp_s: bool = False,
    ):
        if len(bands) != 2:
            raise ValueError("bands 必须是长度为 2 的序列，例如 [5.0, 6.0]")

        f_start, f_end = float(bands[0]), float(bands[1])
        if f_end < f_start:
            raise ValueError(f"频段范围错误：[{f_start}, {f_end}]")
        if penalty_power < 1.0:
            raise ValueError("penalty_power 建议 >= 1.0")

        self.band = (f_start, f_end)
        self.threshold = float(threshold)
        self.weight = float(weight)

        self.freq_start = float(freq_start_ghz)
        self.freq_end = float(freq_end_ghz)
        self.num_points = int(num_points)

        self.penalty_power = float(penalty_power)
        self.satisfied_bonus = float(satisfied_bonus)
        self.clamp_s = bool(clamp_s)

        # 频率轴（默认 CPU，可用 .to(device) 迁移）
        self.freqs = torch.linspace(self.freq_start, self.freq_end, self.num_points)

        # 预计算 band 的索引（性能更好）
        self.band_idx = self._band_indices(*self.band)

    def to(self, device):
        """
        把内部频率轴与索引迁移到指定 device（GPU / CPU），避免跨设备索引错误。
        """
        self.freqs = self.freqs.to(device)
        self.band_idx = self.band_idx.to(device)
        return self

    def _band_indices(self, f_start: float, f_end: float) -> torch.Tensor:
        mask = (self.freqs >= f_start) & (self.freqs <= f_end)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            raise ValueError(
                f"频段 [{f_start}, {f_end}] GHz 不在频率轴范围 "
                f"[{self.freq_start}, {self.freq_end}] GHz 内"
            )
        return idx

    def __call__(self, s_curve: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - s_curve: (B, num_points)

        输出：
        - reward: (B,)
        """
        if s_curve.ndim != 2 or s_curve.shape[1] != self.num_points:
            raise ValueError(
                f"s_curve 形状应为 (B, {self.num_points})，当前为 {tuple(s_curve.shape)}"
            )

        if self.clamp_s:
            s_curve = torch.clamp(s_curve, 0.0, 1.0)

        # 取频段内曲线：(B, N_band)
        band_s = s_curve[:, self.band_idx]

        # 违反阈值的部分：>0 表示超标
        violation = band_s - self.threshold
        # import ipdb; ipdb.set_trace()

        # 奖励与惩罚
        band_penalty = violation.mean(dim=1)  # (B,)

        # reward：越小越好 => 负惩罚
        reward = -band_penalty

        # 完全满足则奖励 bonus（可选）
        if self.satisfied_bonus > 0.0:
            satisfied = (band_s <= self.threshold).all(dim=1)  # (B,)
            reward = reward + satisfied.float() * self.satisfied_bonus

        return reward * self.weight
