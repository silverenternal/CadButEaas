"""Low-rank compression layers for controlled CadStruct ablations."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class CompressionEstimate:
    dense_params: int
    compressed_params: int
    parameter_ratio: float
    method: str


def low_rank_parameter_count(input_dim: int, output_dim: int, rank: int, bias: bool = True) -> CompressionEstimate:
    dense = input_dim * output_dim + (output_dim if bias else 0)
    compressed = input_dim * rank + rank * output_dim + (output_dim if bias else 0)
    return CompressionEstimate(
        dense_params=dense,
        compressed_params=compressed,
        parameter_ratio=round(compressed / max(dense, 1), 6),
        method="low_rank_linear",
    )


if torch is not None:

    class LowRankLinear(torch.nn.Module):
        """Drop-in linear layer using two smaller matrices for ablation."""

        def __init__(self, input_dim: int, output_dim: int, rank: int, bias: bool = True) -> None:
            super().__init__()
            self.down = torch.nn.Linear(input_dim, rank, bias=False)
            self.up = torch.nn.Linear(rank, output_dim, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.up(self.down(x))
