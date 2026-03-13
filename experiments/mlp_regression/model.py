"""
FunctionMLP: A small MLP for approximating 1D parameterized functions.

Architecture: 1 -> hidden -> hidden -> 1 (3-layer MLP with ReLU)
Default hidden=64 gives ~4,353 parameters.

Target function family:
    y = a * sin(f * x + phi) + b * x

Condition vector: c = (a, b, f, phi)
"""
import torch
import torch.nn as nn


class FunctionMLP(nn.Module):
    """Small MLP for 1D function approximation."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),       # 1 -> hidden
            nn.ReLU(),
            nn.Linear(hidden, hidden),  # hidden -> hidden
            nn.ReLU(),
            nn.Linear(hidden, 1),       # hidden -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def target_function(x: torch.Tensor, a: float, b: float, f: float, phi: float) -> torch.Tensor:
    """
    Compute the target function y = a * sin(f * x + phi) + b * x.

    Args:
        x: Input tensor of shape (N, 1) or (N,)
        a: Sine amplitude [0.2, 2.0]
        b: Linear slope [-1.0, 1.0]
        f: Frequency [0.5, 3.0]
        phi: Phase [0, 2*pi]

    Returns:
        y: Output tensor matching x shape
    """
    return a * torch.sin(f * x + phi) + b * x


if __name__ == "__main__":
    model = FunctionMLP(hidden=64)
    print(f"FunctionMLP parameters: {count_parameters(model)}")
    print(f"State dict keys: {list(model.state_dict().keys())}")
    for k, v in model.state_dict().items():
        print(f"  {k}: {v.shape}")
