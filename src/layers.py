"""
Neural Network Layers: RMSNorm and SwiGLU Feed-Forward Network.
Implements modern LLM sublayers from first principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Normalizes inputs strictly by RMS without subtracting the mean:
    y = (x / RMS(x)) * gamma, where RMS(x) = sqrt(mean(x^2) + eps).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square over the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)
        return x * rms * self.gamma


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (Shazeer, 2020).
    Computes: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    Enhances representation capacity with smooth gating.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gating: SiLU(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
