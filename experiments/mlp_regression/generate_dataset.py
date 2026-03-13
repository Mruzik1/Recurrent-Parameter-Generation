"""
Generate the full MLP regression dataset.

Samples 400 condition vectors using a mixed strategy (grid anchors + Latin
Hypercube Sampling), trains one FunctionMLP per condition, applies a quality
gate, and saves a manifest.json with metadata.

Usage:
    python generate_dataset.py [--num_conditions 400] [--output_dir ./dataset]
"""
import argparse
import itertools
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from model import FunctionMLP, target_function
from train_single import train_mlp, make_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Parameter ranges from the experiment plan
PARAM_RANGES = {
    "a": (0.2, 2.0),    # Sine amplitude
    "b": (-1.0, 1.0),   # Linear slope
    "f": (0.5, 3.0),    # Frequency
    "phi": (0.0, 2 * math.pi),  # Phase
}
PARAM_NAMES = list(PARAM_RANGES.keys())


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def sample_lhs(num_samples: int, seed: int = 42) -> list:
    """
    Latin Hypercube Sampling for 4D condition space.

    Divides each dimension into num_samples strata with exactly one sample
    per stratum, giving excellent space-filling coverage.
    """
    rng = np.random.default_rng(seed)
    n_dims = len(PARAM_RANGES)
    bounds = list(PARAM_RANGES.values())

    # Generate LHS in [0,1]^d
    result = np.zeros((num_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(num_samples)
        for i in range(num_samples):
            result[perm[i], d] = (i + rng.uniform()) / num_samples

    # Scale to actual parameter ranges
    samples = []
    for i in range(num_samples):
        params = {}
        for d, name in enumerate(PARAM_NAMES):
            lo, hi = bounds[d]
            params[name] = float(lo + result[i, d] * (hi - lo))
        samples.append(params)

    return samples


def sample_grid() -> list:
    """
    Generate structured grid anchor points covering key regions.

    Includes: center, all 16 corners, axis midpoints (25%/75%), pairwise extremes.
    """
    bounds = dict(PARAM_RANGES)
    center = {k: (lo + hi) / 2.0 for k, (lo, hi) in bounds.items()}
    samples = []

    # 1) Center point
    samples.append(dict(center))

    # 2) All 2^4 = 16 corners
    extremes = {n: list(bounds[n]) for n in PARAM_NAMES}
    for combo in itertools.product(*[extremes[n] for n in PARAM_NAMES]):
        samples.append({PARAM_NAMES[i]: combo[i] for i in range(len(PARAM_NAMES))})

    # 3) Axis midpoints: vary one param at 25%/75%, rest at center
    for name in PARAM_NAMES:
        lo, hi = bounds[name]
        for frac in [0.25, 0.75]:
            point = dict(center)
            point[name] = lo + frac * (hi - lo)
            samples.append(point)

    # 4) Pairwise extremes  
    for i in range(len(PARAM_NAMES)):
        for j in range(i + 1, len(PARAM_NAMES)):
            for vi in bounds[PARAM_NAMES[i]]:
                for vj in bounds[PARAM_NAMES[j]]:
                    point = dict(center)
                    point[PARAM_NAMES[i]] = vi
                    point[PARAM_NAMES[j]] = vj
                    samples.append(point)

    # De-duplicate
    unique = []
    seen = set()
    for s in samples:
        key = tuple(round(s[n], 6) for n in PARAM_NAMES)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def generate_conditions(num_samples: int, method: str = "mixed", seed: int = 42) -> list:
    """
    Generate condition vectors using the chosen sampling strategy.

    Args:
        num_samples: Total number of unique conditions.
        method: "lhs", "grid", or "mixed" (grid anchors + LHS fill).
        seed: Random seed.

    Returns:
        List of condition dicts with keys: a, b, f, phi.
    """
    np.random.seed(seed)

    if method == "lhs":
        return sample_lhs(num_samples, seed=seed)
    elif method == "grid":
        return sample_grid()[:num_samples]
    elif method == "mixed":
        grid = sample_grid()
        if len(grid) >= num_samples:
            return grid[:num_samples]
        remaining = num_samples - len(grid)
        lhs = sample_lhs(remaining, seed=seed)
        return grid + lhs
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def train_worker(task: tuple) -> dict:
    """
    Worker function to train one MLP.

    Args:
        task: (index, params_dict, output_dir, hidden, epochs, lr, n_train,
               n_eval, seed, max_mse)

    Returns:
        dict with: success, filename, mse, params, message
    """
    (idx, params, output_dir, hidden, epochs, lr, n_train, n_eval, seed, max_mse) = task

    a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]
    filename = make_filename(a, b, f, phi)
    save_path = os.path.join(output_dir, filename)

    # Skip if already exists
    if os.path.exists(save_path):
        return {
            "success": True,
            "filename": filename,
            "mse": None,
            "params": params,
            "message": "already exists",
        }

    try:
        model, eval_mse = train_mlp(
            a=a, b=b, f=f, phi=phi,
            hidden=hidden, epochs=epochs, lr=lr,
            n_train=n_train, n_eval=n_eval, seed=seed,
            quiet=True,
        )

        if eval_mse > max_mse:
            return {
                "success": False,
                "filename": filename,
                "mse": eval_mse,
                "params": params,
                "message": f"MSE {eval_mse:.6f} > threshold {max_mse}",
            }

        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)

        return {
            "success": True,
            "filename": filename,
            "mse": eval_mse,
            "params": params,
            "message": "ok",
        }

    except Exception as e:
        return {
            "success": False,
            "filename": filename,
            "mse": None,
            "params": params,
            "message": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate MLP regression dataset")
    parser.add_argument("--num_conditions", type=int, default=400, help="Total conditions")
    parser.add_argument("--sampling_method", type=str, default="mixed",
                        choices=["lhs", "grid", "mixed"])
    parser.add_argument("--output_dir", type=str,
                        default="./experiments/mlp_regression/dataset")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_mse", type=float, default=0.01, help="Quality gate threshold")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate conditions
    logger.info(f"Generating {args.num_conditions} conditions (method={args.sampling_method})")
    conditions = generate_conditions(args.num_conditions, method=args.sampling_method, seed=args.seed)
    logger.info(f"Generated {len(conditions)} unique conditions "
                f"(grid anchors + LHS fill)")

    # Prepare tasks
    tasks = [
        (i, cond, args.output_dir, args.hidden, args.epochs, args.lr,
         args.n_train, args.n_eval, args.seed, args.max_mse)
        for i, cond in enumerate(conditions)
    ]

    # Train in parallel
    manifest = []
    success_count = 0
    fail_count = 0

    logger.info(f"Training {len(tasks)} MLPs with {args.num_workers} workers...")

    if args.num_workers <= 1:
        # Sequential for debugging
        results = []
        for task in tqdm(tasks, desc="Training MLPs"):
            results.append(train_worker(task))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(train_worker, t): i for i, t in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Training MLPs"):
                results.append(future.result())

    # Process results
    for result in results:
        if result["success"]:
            success_count += 1
            manifest.append({
                "filename": result["filename"],
                "params": result["params"],
                "mse": result["mse"],
            })
        else:
            fail_count += 1
            logger.warning(f"Failed: {result['filename']} - {result['message']}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "..", "manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump({
            "num_conditions": len(conditions),
            "num_success": success_count,
            "num_failed": fail_count,
            "hidden_size": args.hidden,
            "epochs": args.epochs,
            "param_ranges": {k: list(v) for k, v in PARAM_RANGES.items()},
            "checkpoints": manifest,
        }, fp, indent=2)

    logger.info(f"Done! Success: {success_count}, Failed: {fail_count}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main()
