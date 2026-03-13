"""
Train a single FunctionMLP for a specific condition vector (a, b, f, phi).

Usage:
    python train_single.py --a 1.0 --b 0.5 --f 2.0 --phi 1.57 --output_dir ./dataset
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import FunctionMLP, target_function


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single FunctionMLP")
    parser.add_argument("--a", type=float, required=True, help="Sine amplitude [0.2, 2.0]")
    parser.add_argument("--b", type=float, required=True, help="Linear slope [-1.0, 1.0]")
    parser.add_argument("--f", type=float, required=True, help="Frequency [0.5, 3.0]")
    parser.add_argument("--phi", type=float, required=True, help="Phase [0, 2*pi]")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_train", type=int, default=200, help="Number of training points")
    parser.add_argument("--n_eval", type=int, default=1000, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_mse", type=float, default=0.01, help="Quality gate: max eval MSE")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    return parser.parse_args()


def make_filename(a: float, b: float, f: float, phi: float) -> str:
    """Create standardized filename from condition parameters."""
    return f"func_a{a:.2f}_b{b:.2f}_f{f:.2f}_p{phi:.2f}.pth"


def train_mlp(a: float, b: float, f: float, phi: float,
              hidden: int = 64, epochs: int = 500, lr: float = 1e-3,
              n_train: int = 200, n_eval: int = 1000, seed: int = 42,
              quiet: bool = False):
    """
    Train a FunctionMLP for the given condition parameters.

    Returns:
        model: Trained FunctionMLP
        eval_mse: Final MSE on evaluation set
    """
    torch.manual_seed(seed)

    # Generate training data: x uniform in [-pi, pi]
    x_train = torch.rand(n_train, 1) * 2 * math.pi - math.pi
    y_train = target_function(x_train, a, b, f, phi)

    # Generate eval data: uniformly spaced in [-pi, pi]
    x_eval = torch.linspace(-math.pi, math.pi, n_eval).unsqueeze(1)
    y_eval = target_function(x_eval, a, b, f, phi)

    # Model and optimizer
    model = FunctionMLP(hidden=hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if not quiet and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={loss.item():.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_eval = model(x_eval)
        eval_mse = criterion(y_pred_eval, y_eval).item()

    if not quiet:
        print(f"  Final eval MSE: {eval_mse:.6f}")

    return model, eval_mse


def main():
    args = parse_args()

    if not args.quiet:
        print(f"Training MLP for condition: a={args.a:.2f}, b={args.b:.2f}, "
              f"f={args.f:.2f}, phi={args.phi:.2f}")

    model, eval_mse = train_mlp(
        a=args.a, b=args.b, f=args.f, phi=args.phi,
        hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        n_train=args.n_train, n_eval=args.n_eval, seed=args.seed,
        quiet=args.quiet,
    )

    # Quality gate
    if eval_mse > args.max_mse:
        print(f"WARNING: Eval MSE {eval_mse:.6f} exceeds threshold {args.max_mse}. "
              f"Checkpoint NOT saved.")
        return None

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    filename = make_filename(args.a, args.b, args.f, args.phi)
    save_path = os.path.join(args.output_dir, filename)
    torch.save(model.state_dict(), save_path)

    if not args.quiet:
        print(f"Saved: {save_path}")

    return save_path


if __name__ == "__main__":
    main()
