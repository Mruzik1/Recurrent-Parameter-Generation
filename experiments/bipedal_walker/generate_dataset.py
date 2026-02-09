"""
Generate a dataset of BipedalWalker policy checkpoints with randomized environment parameters.

Improvements over the original:
  - Structured sampling: Latin Hypercube Sampling (LHS) + grid anchors for
    systematic 4D coverage instead of pure uniform random.
  - Multiple seeds per condition: reduces initialization variance.
  - Quality gate: post-training evaluation filters out junk policies.
  - Manifest: JSON metadata file records condition, seed, reward for every run.
  - Better defaults: 300k timesteps, entropy bonus, LR annealing.
"""
import argparse
import itertools
import json
import logging
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from env_utils import make_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Parameter ranges (from EXPERIMENTS.md)
PARAM_RANGES = {
    "leg_length": (0.5, 2.0),
    "leg_width": (0.1, 0.5),
    "gravity": (-20.0, -5.0),
    "friction": (0.1, 5.0),
}

PARAM_NAMES = list(PARAM_RANGES.keys())


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------


def sample_parameters_random() -> dict:
    """Sample random environment parameters within defined ranges (legacy)."""
    return {
        name: random.uniform(*bounds) for name, bounds in PARAM_RANGES.items()
    }


def sample_parameters_lhs(num_samples: int, seed: int = 42) -> list:
    """
    Latin Hypercube Sampling for better coverage of 4D parameter space.

    LHS divides each dimension into *num_samples* equal strata and places
    exactly one sample in each stratum, then shuffles across dimensions.
    This guarantees no two samples share the same "row" in any dimension,
    giving far better space-filling properties than pure random sampling.
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


def sample_parameters_grid() -> list:
    """
    Generate structured anchor points that cover important regions.

    Includes:
      - Default/center point
      - All 16 corner combinations (parameter extremes)
      - Mid-points along each axis (25% and 75% quantiles, others at center)
      - Pairwise extreme combinations (all pairs of dimensions at lo/hi)
    """
    bounds = dict(PARAM_RANGES)
    center = {k: (lo + hi) / 2.0 for k, (lo, hi) in bounds.items()}
    samples = []

    # 1) Center / default point
    samples.append(dict(center))

    # 2) All 2^4 = 16 corners
    extremes = {n: list(bounds[n]) for n in PARAM_NAMES}
    for combo in itertools.product(*[extremes[n] for n in PARAM_NAMES]):
        samples.append({PARAM_NAMES[i]: combo[i] for i in range(len(PARAM_NAMES))})

    # 3) Axis mid-points: vary one param at 25%/75%, rest at center
    for name in PARAM_NAMES:
        lo, hi = bounds[name]
        for frac in [0.25, 0.75]:
            point = dict(center)
            point[name] = lo + frac * (hi - lo)
            samples.append(point)

    # 4) Pairwise extremes: for each pair, set both to lo/hi combos
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
    Generate a list of condition dicts using the chosen sampling strategy.

    Args:
        num_samples: Total number of unique conditions to generate.
        method: "random", "lhs", "grid", or "mixed" (grid anchors + LHS fill).
        seed: Random seed for reproducibility.

    Returns:
        List of parameter dictionaries, length <= num_samples.
    """
    random.seed(seed)
    np.random.seed(seed)

    if method == "random":
        return [sample_parameters_random() for _ in range(num_samples)]

    elif method == "lhs":
        return sample_parameters_lhs(num_samples, seed=seed)

    elif method == "grid":
        grid = sample_parameters_grid()
        if len(grid) >= num_samples:
            return grid[:num_samples]
        remaining = num_samples - len(grid)
        lhs = sample_parameters_lhs(remaining, seed=seed + 1)
        return grid + lhs

    elif method == "mixed":
        grid = sample_parameters_grid()
        if len(grid) >= num_samples:
            return grid[:num_samples]
        remaining = num_samples - len(grid)
        lhs = sample_parameters_lhs(remaining, seed=seed)
        return grid + lhs

    else:
        raise ValueError(f"Unknown sampling method: {method}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def train_worker(task: tuple) -> dict:
    """
    Worker function for training a single agent.

    Args:
        task: (index, params_dict, output_dir, total_timesteps, seed,
               min_reward, eval_episodes)

    Returns:
        dict with keys: success, filename, reward, rejected, message,
                        params, seed
    """
    idx, params, output_dir, total_timesteps, seed, min_reward, eval_episodes = task

    filename = make_filename(**params)
    # Append seed suffix to allow multiple seeds per condition
    base, ext = os.path.splitext(filename)
    filename_with_seed = f"{base}_s{seed}{ext}"
    output_path = os.path.join(output_dir, filename_with_seed)

    # Skip if already exists
    if os.path.exists(output_path):
        return {
            "success": True,
            "filename": filename_with_seed,
            "reward": None,
            "rejected": False,
            "message": f"Skipped {filename_with_seed} (exists)",
            "params": params,
            "seed": seed,
        }

    # Build command
    script_path = Path(__file__).parent / "train_single.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--leg_length", str(params["leg_length"]),
        "--leg_width", str(params["leg_width"]),
        "--gravity", str(params["gravity"]),
        "--friction", str(params["friction"]),
        "--output_path", output_path,
        "--total_timesteps", str(total_timesteps),
        "--seed", str(seed),
        "--min_reward", str(min_reward),
        "--eval_episodes", str(eval_episodes),
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
        )

        # Parse evaluation reward and quality gate from stdout
        reward = None
        rejected = False
        for line in result.stdout.split("\n"):
            if line.startswith("EVAL_REWARD="):
                try:
                    reward = float(line.split("=")[1])
                except (ValueError, IndexError):
                    pass
            if line.strip() == "QUALITY_GATE=REJECTED":
                rejected = True

        return {
            "success": True,
            "filename": filename_with_seed,
            "reward": reward,
            "rejected": rejected,
            "message": f"{filename_with_seed} reward={reward}",
            "params": params,
            "seed": seed,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "filename": filename_with_seed,
            "reward": None,
            "rejected": False,
            "message": f"Timeout: {filename_with_seed}",
            "params": params,
            "seed": seed,
        }
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr[:300] if e.stderr else "Unknown error"
        return {
            "success": False,
            "filename": filename_with_seed,
            "reward": None,
            "rejected": False,
            "message": f"Failed {filename_with_seed}: {stderr_msg}",
            "params": params,
            "seed": seed,
        }


# ---------------------------------------------------------------------------
# Manifest & statistics
# ---------------------------------------------------------------------------


def save_manifest(manifest: list, output_dir: str):
    """Save a JSON manifest with metadata for every training run."""
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {path} ({len(manifest)} entries)")


def print_statistics(manifest: list):
    """Print summary statistics about the generated dataset."""
    total = len(manifest)
    successful = [
        m for m in manifest
        if m["success"] and not m.get("rejected", False)
        and not m.get("message", "").startswith("Skipped")
    ]
    rejected = [m for m in manifest if m.get("rejected", False)]
    failed = [m for m in manifest if not m["success"]]
    skipped = [
        m for m in manifest if m.get("message", "").startswith("Skipped")
    ]

    rewards = [m["reward"] for m in successful if m["reward"] is not None]

    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"  Total tasks:       {total}")
    logger.info(f"  Saved (accepted):  {len(successful)}")
    logger.info(f"  Rejected (low-q):  {len(rejected)}")
    logger.info(f"  Failed/timeout:    {len(failed)}")
    logger.info(f"  Skipped (existed): {len(skipped)}")

    if rewards:
        r = np.array(rewards)
        logger.info("  Reward stats:")
        logger.info(f"    Mean:   {r.mean():.1f}")
        logger.info(f"    Std:    {r.std():.1f}")
        logger.info(f"    Min:    {r.min():.1f}")
        logger.info(f"    Max:    {r.max():.1f}")
        logger.info(f"    Median: {np.median(r):.1f}")

        # Reward distribution buckets
        buckets = [
            (-300, -200), (-200, -100), (-100, 0),
            (0, 100), (100, 200), (200, 350),
        ]
        logger.info("  Reward distribution:")
        for lo, hi in buckets:
            count = int(((r >= lo) & (r < hi)).sum())
            pct = 100 * count / len(r)
            bar = "#" * int(pct / 2)
            logger.info(f"    [{lo:>4},{hi:>4}): {count:>5} ({pct:5.1f}%) {bar}")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate BipedalWalker dataset")
    parser.add_argument(
        "--num_conditions", type=int, default=3000,
        help="Number of unique morphology conditions to sample",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=3,
        help="Number of random seeds to train per condition",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="experiments/bipedal_walker/dataset",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Parallel workers",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=300_000,
        help="Training steps per model",
    )
    parser.add_argument(
        "--min_reward", type=float, default=-200.0,
        help="Minimum eval reward to accept a model (-inf to disable)",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10,
        help="Evaluation episodes per trained model",
    )
    parser.add_argument(
        "--sampling_method", type=str, default="mixed",
        choices=["random", "lhs", "grid", "mixed"],
        help="Condition sampling strategy",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate conditions
    conditions = generate_conditions(
        args.num_conditions, method=args.sampling_method, seed=args.seed,
    )
    logger.info(
        f"Generated {len(conditions)} unique conditions "
        f"using '{args.sampling_method}' sampling"
    )

    # Expand conditions x seeds into tasks
    tasks = []
    for cond_idx, params in enumerate(conditions):
        for seed_offset in range(args.num_seeds):
            seed = args.seed + cond_idx * args.num_seeds + seed_offset
            tasks.append((
                len(tasks), params, args.output_dir, args.total_timesteps,
                seed, args.min_reward, args.eval_episodes,
            ))

    total_models = len(tasks)
    logger.info(
        f"Total training tasks: {total_models} "
        f"({len(conditions)} conditions x {args.num_seeds} seeds)"
    )
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Timesteps per model: {args.total_timesteps}")
    logger.info(f"Quality threshold: {args.min_reward}")

    # Track results
    manifest = []
    success_count = 0
    reject_count = 0
    skip_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(train_worker, task): task for task in tasks}

        with tqdm(total=len(futures), desc="Training Models", unit="model") as pbar:
            for future in as_completed(futures):
                result = future.result()
                manifest.append(result)

                if result["success"]:
                    if "Skipped" in result.get("message", ""):
                        skip_count += 1
                    elif result.get("rejected", False):
                        reject_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    logger.warning(result["message"])

                pbar.update(1)
                pbar.set_postfix(
                    ok=success_count, rej=reject_count,
                    skip=skip_count, fail=fail_count,
                )

    # Save manifest and print statistics
    save_manifest(manifest, args.output_dir)
    print_statistics(manifest)


if __name__ == "__main__":
    main()
