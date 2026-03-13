"""
Evaluate a generated MLP checkpoint against the ground-truth target function.

Usage:
    python test.py <checkpoint_path>
    python test.py experiments/mlp_regression/test_generated/generated_model.pth --condition 1.0 0.5 2.0 1.57
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import torch

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from experiments.mlp_regression.model import FunctionMLP, target_function


def parse_condition_from_filename(filename: str) -> dict:
    """
    Extract condition parameters from filename like func_a1.50_b0.30_f2.00_p1.57.pth.

    Returns:
        dict with keys: a, b, f, phi
    """
    basename = os.path.basename(filename)
    pattern = r"func_a([\d.-]+)_b([\d.-]+)_f([\d.-]+)_p([\d.-]+)\.pth"
    match = re.match(pattern, basename)
    if match:
        return {
            "a": float(match.group(1)),
            "b": float(match.group(2)),
            "f": float(match.group(3)),
            "phi": float(match.group(4)),
        }
    raise ValueError(f"Cannot parse condition from filename: {basename}")


def test_mlp(checkpoint_path: str, a: float, b: float, f: float, phi: float,
             hidden: int = 64, n_eval: int = 1000) -> float:
    """
    Load a checkpoint and evaluate against ground truth.

    Returns:
        eval_mse: MSE on uniformly-spaced evaluation points
    """
    # Load model
    model = FunctionMLP(hidden=hidden)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluation points
    x_eval = torch.linspace(-math.pi, math.pi, n_eval).unsqueeze(1)
    y_true = target_function(x_eval, a, b, f, phi)

    with torch.no_grad():
        y_pred = model(x_eval)
        mse = torch.mean((y_pred - y_true) ** 2).item()

    return mse


def main():
    parser = argparse.ArgumentParser(description="Test a FunctionMLP checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pth file")
    parser.add_argument("--condition", type=float, nargs=4, default=None,
                        metavar=("A", "B", "F", "PHI"),
                        help="Condition vector (a, b, f, phi). If not given, parsed from filename.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_eval", type=int, default=1000)
    args = parser.parse_args()

    # Get condition
    if args.condition is not None:
        a, b, f, phi = args.condition
    else:
        try:
            params = parse_condition_from_filename(args.checkpoint)
            a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]
        except ValueError:
            print("ERROR: Cannot determine condition. Use --condition A B F PHI")
            sys.exit(1)

    print(f"Testing: a={a:.2f}, b={b:.2f}, f={f:.2f}, phi={phi:.2f}")
    print(f"Checkpoint: {args.checkpoint}")

    mse = test_mlp(args.checkpoint, a, b, f, phi, hidden=args.hidden, n_eval=args.n_eval)

    print(f"\nResults:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {math.sqrt(mse):.6f}")

    return mse


if __name__ == "__main__":
    main()
