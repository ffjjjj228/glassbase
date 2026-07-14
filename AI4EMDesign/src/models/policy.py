import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torchutils.common import get_device
from src.registry import MODEL

@MODEL.register
class AntennaRowPolicyModel(nn.Module):
    """
    天线拓扑生成的 RL Policy Model（支持 batch）

    设计：
    - 生成 topology bits：shape = (B, n_steps, row_bits)
    - 每一步动作空间大小为 2^row_bits
      action ∈ [0, 2^row_bits - 1]
      将 action 转为 row_bits 位二进制，得到该行 0/1

    forward 返回：
    - actions: torch.LongTensor, shape=(B, n_steps)
    - topo_bits: torch.LongTensor, shape=(B, n_steps, row_bits)
    - logp_sum: torch.Tensor, shape=(B,)      # 每个样本序列 log-prob 之和
    - entropy_sum: torch.Tensor, shape=(B,)   # 每个样本序列 entropy 之和
    """

    def __init__(
        self,
        n_steps: int = 4,
        row_bits: int = 4,
        hidden_size: int = 64,
        temperature=None,
        tanh_constant: float = 1.5,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.row_bits = row_bits
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.device = get_device()

        self.vocab_size = 2 ** row_bits

        self.action_embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.start_token = nn.Parameter(torch.zeros(hidden_size))

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.logits_head = nn.Linear(hidden_size, self.vocab_size)

        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for p in self.parameters():
            if p.requires_grad:
                p.data.uniform_(-init_range, init_range)

    # 注意：cache 对 batch_size 生效；device 如果可能变化，建议不要跨 device 复用
    @functools.lru_cache(maxsize=64)
    def _zeros(self, batch_size: int):
        return torch.zeros((batch_size, self.hidden_size), device=self.device)

    def _scale_logits(self, logits: torch.Tensor):
        if self.temperature is not None:
            logits = logits / self.temperature
        if self.tanh_constant is not None:
            logits = self.tanh_constant * torch.tanh(logits)
        return logits

    def _sample_action(self, probs: torch.Tensor):
        """
        probs: (B, vocab_size)
        return:
          action: (B,)
          logp: (B,)
          entropy: (B,)
        """
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample()
        logp = m.log_prob(action)
        entropy = m.entropy()
        return action, logp, entropy

    @staticmethod
    def _int_to_bits(action_int: torch.Tensor, num_bits: int) -> torch.Tensor:
        """
        action_int: (B, n_steps) 或 (B,) 的 int64
        return:     (..., num_bits) 的 0/1 bits
        """
        shifts = torch.arange(num_bits - 1, -1, -1, device=action_int.device, dtype=torch.long)
        # 通过 broadcast，把最后一维变成 bits
        bits = (action_int.unsqueeze(-1) >> shifts) & 1
        return bits

    def forward(self, batch_size: int = 1, force_uniform: bool = False):
        """
        batch_size: 每次采样多少个 architecture / topology
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，当前为 {batch_size}")

        # buffers
        actions_buf = []      # 每步 action: (B,)
        logp_buf = []         # 每步 logp: (B,)
        entropy_buf = []      # 每步 entropy: (B,)

        # 初始隐状态
        hx = self._zeros(batch_size)
        cx = self._zeros(batch_size)

        # 第一步输入：start token，扩展到 batch
        embed = self.start_token.unsqueeze(0).expand(batch_size, -1)  # (B, H)

        for _ in range(self.n_steps):
            hx, cx = self.lstm(embed, (hx, cx))

            if force_uniform:
                logits = torch.zeros((batch_size, self.vocab_size), device=self.device)
            else:
                logits = self.logits_head(hx)
                logits = self._scale_logits(logits)

            probs = F.softmax(logits, dim=-1)
            action, logp, entropy = self._sample_action(probs)

            actions_buf.append(action)      # (B,)
            logp_buf.append(logp)           # (B,)
            entropy_buf.append(entropy)     # (B,)

            # 下一步输入：action embedding
            embed = self.action_embedding(action)  # (B, H)

        # (B, n_steps)
        actions = torch.stack(actions_buf, dim=1).to(torch.long)

        # (B, n_steps, row_bits)
        topo_bits = self._int_to_bits(actions, self.row_bits).to(torch.long)

        # 每个样本的序列 logp/entropy 之和：(B,)
        logp_sum = torch.stack(logp_buf, dim=1).sum(dim=1)
        entropy_sum = torch.stack(entropy_buf, dim=1).sum(dim=1)

        return actions, topo_bits, logp_sum, entropy_sum


@MODEL.register
class AntennaTopologyPolicyModel(nn.Module):
    """
    22x22 拓扑生成策略（边框约束 + 内部随机）。

    边框：每边恰好 1 个 1，位置由 4 步决策（vocab_size=22）
    内部：20x20 逐 cell 生成（400 步，每步 0/1）

    forward 返回：
    - actions: None（与旧接口对齐）
    - topo_bits: torch.LongTensor, shape=(B, 22, 22)
    - logp_sum: torch.Tensor, shape=(B,)
    - entropy_sum: torch.Tensor, shape=(B,)
    """

    def __init__(
        self,
        hidden_size: int = 64,
        temperature=None,
        tanh_constant: float = 1.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = get_device()

        self.border = AntennaRowPolicyModel(
            n_steps=4, row_bits=5, hidden_size=hidden_size,
            temperature=temperature, tanh_constant=tanh_constant,
        )
        self.inner = AntennaRowPolicyModel(
            n_steps=400, row_bits=1, hidden_size=hidden_size,
            temperature=temperature, tanh_constant=tanh_constant,
        )

    def forward(self, batch_size: int = 1, force_uniform: bool = False):
        b_actions, _, b_logp, b_entropy = self.border(
            batch_size=batch_size, force_uniform=force_uniform,
        )
        _, i_bits, i_logp, i_entropy = self.inner(
            batch_size=batch_size, force_uniform=force_uniform,
        )

        topo = self._assemble(b_actions.clamp(0, 21), i_bits)

        logp_sum = b_logp + i_logp
        entropy_sum = b_entropy + i_entropy

        return None, topo, logp_sum, entropy_sum

    @staticmethod
    def _assemble(b_pos: torch.Tensor, i_bits: torch.Tensor) -> torch.Tensor:
        B = b_pos.size(0)
        dev = b_pos.device
        topo = torch.zeros(B, 22, 22, device=dev, dtype=torch.long)

        topo[:, 1:21, 1:21] = i_bits.squeeze(-1).reshape(B, 20, 20)

        # Corners assigned to top/bottom edges exclusively.
        # Left/right vertical edges map to rows 1..20 (non-corner).
        topo[range(B), 0, b_pos[:, 0]] = 1
        topo[range(B), 21, b_pos[:, 2]] = 1
        topo[range(B), b_pos[:, 1].clamp(1, 20), 21] = 1
        topo[range(B), b_pos[:, 3].clamp(1, 20), 0] = 1

        return topo


if __name__ == "__main__":
    policy_model = AntennaRowPolicyModel(n_steps=4, row_bits=4, hidden_size=64, device="cpu")

    actions, topo_bits, logp, entropy = policy_model(batch_size=3)

    print("actions shape:", actions.shape)         # (3, 4)
    print(actions)
    print("topo_bits shape:", topo_bits.shape)     # (3, 4, 4)
    print(topo_bits)
    print("logp shape:", logp.shape)               # (3,)
    print("entropy shape:", entropy.shape)         # (3,)
