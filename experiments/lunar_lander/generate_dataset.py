"""
Generate a dataset of LunarLander policy checkpoints with varied reward weights.
Implements the "Skill Composer" data collection strategy from EXPERIMENTS.md.
Uses Dirichlet distribution to sample diverse weight combinations.
"""
import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def sample_weights_dirichlet(alpha: float = 1.0) -> dict:
    """
    Sample reward weights from a Dirichlet distribution.

    The Dirichlet distribution ensures weights sum to 1.0 and provides
    good coverage of the weight space.

    Args:
        alpha: Concentration parameter. Lower values (e.g., 0.5) create more
               extreme distributions, higher values (e.g., 2.0) create more
               uniform distributions. Default 1.0 gives uniform prior.

    Returns:
        Dictionary with keys: w_landing, w_fuel, w_time, w_smoothness
    """
    weights = np.random.dirichlet([alpha] * 4)
    return {
        "w_landing": float(weights[0]),
        "w_fuel": float(weights[1]),
        "w_time": float(weights[2]),
        "w_smoothness": float(weights[3]),
    }


def sample_weights_grid(num_samples: int) -> list:
    """
    Generate grid-based weight combinations for more systematic coverage.

    Returns:
        List of weight dictionaries
    """
    weights_list = []

    # Corner cases (one weight dominates)
    weights_list.extend([
        {"w_landing": 1.0, "w_fuel": 0.0, "w_time": 0.0, "w_smoothness": 0.0},
        {"w_landing": 0.0, "w_fuel": 1.0, "w_time": 0.0, "w_smoothness": 0.0},
        {"w_landing": 0.0, "w_fuel": 0.0, "w_time": 1.0, "w_smoothness": 0.0},
        {"w_landing": 0.0, "w_fuel": 0.0, "w_time": 0.0, "w_smoothness": 1.0},
    ])

    # Uniform case
    weights_list.append({"w_landing": 0.25, "w_fuel": 0.25, "w_time": 0.25, "w_smoothness": 0.25})

    # Two-way splits
    for w1 in [0.5, 0.7, 0.9]:
        w2 = 1.0 - w1
        weights_list.extend([
            {"w_landing": w1, "w_fuel": w2, "w_time": 0.0, "w_smoothness": 0.0},
            {"w_landing": w1, "w_fuel": 0.0, "w_time": w2, "w_smoothness": 0.0},
            {"w_landing": w1, "w_fuel": 0.0, "w_time": 0.0, "w_smoothness": w2},
            {"w_landing": 0.0, "w_fuel": w1, "w_time": w2, "w_smoothness": 0.0},
            {"w_landing": 0.0, "w_fuel": w1, "w_time": 0.0, "w_smoothness": w2},
            {"w_landing": 0.0, "w_fuel": 0.0, "w_time": w1, "w_smoothness": w2},
        ])

    # Fill remaining with Dirichlet samples
    while len(weights_list) < num_samples:
        weights_list.append(sample_weights_dirichlet(alpha=1.0))

    return weights_list[:num_samples]


def get_existing_filenames(output_dir: str) -> set:
    """Get set of all existing checkpoint filenames."""
    if not os.path.exists(output_dir):
        return set()
    return {f for f in os.listdir(output_dir) if f.endswith('.pth')}


def train_worker(task: tuple) -> tuple:
    """
    Worker function for training a single agent.

    Args:
        task: (index, weights_dict, output_dir, total_timesteps, seed)

    Returns:
        (success: bool, message: str)
    """
    idx, weights, output_dir, total_timesteps, seed = task

    # Build command
    script_path = Path(__file__).parent / "train_single.py"
    cmd = [
        sys.executable, str(script_path),
        "--w_landing", str(weights["w_landing"]),
        "--w_fuel", str(weights["w_fuel"]),
        "--w_time", str(weights["w_time"]),
        "--w_smoothness", str(weights["w_smoothness"]),
        "--output_dir", output_dir,
        "--total_timesteps", str(total_timesteps),
        "--seed", str(seed),
        "--quiet",  # Suppress progress bars in workers
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout
        )

        # Extract filename from output
        for line in result.stdout.split('\n'):
            if 'Model saved to' in line:
                filename = os.path.basename(line.split()[-1])
                return (True, filename)

        return (True, f"Task {idx} completed")

    except subprocess.TimeoutExpired:
        return (False, f"Timeout: Task {idx}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[:300] if e.stderr else "Unknown error"
        return (False, f"Failed Task {idx}: {error_msg}")
    except Exception as e:
        return (False, f"Exception in Task {idx}: {str(e)[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Generate LunarLander dataset")
    parser.add_argument("--num_samples", type=int, default=2000,
                       help="Number of models to train")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/lunar_lander/checkpoint",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--total_timesteps", type=int, default=500_000,
                       help="Training steps per model")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for weight sampling")
    parser.add_argument("--sampling_method", type=str, default="dirichlet",
                       choices=["dirichlet", "grid", "mixed"],
                       help="Method for sampling weights")
    args = parser.parse_args()

    # Seed for reproducible weight sampling
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate weight combinations
    logger.info(f"Sampling {args.num_samples} weight combinations using {args.sampling_method} method")

    if args.sampling_method == "dirichlet":
        weights_list = [sample_weights_dirichlet(alpha=1.0) for _ in range(args.num_samples)]
    elif args.sampling_method == "grid":
        weights_list = sample_weights_grid(args.num_samples)
    else:  # mixed
        # Use grid for first 25%, then fill with Dirichlet
        grid_count = max(25, args.num_samples // 4)
        weights_list = sample_weights_grid(grid_count)
        remaining = args.num_samples - len(weights_list)
        weights_list.extend([sample_weights_dirichlet(alpha=1.0) for _ in range(remaining)])

    # Create tasks with unique seeds
    tasks = []
    for i, weights in enumerate(weights_list):
        seed = args.seed + i
        tasks.append((i, weights, args.output_dir, args.total_timesteps, seed))

    logger.info(f"Generating {args.num_samples} samples with {args.num_workers} workers")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Timesteps per model: {args.total_timesteps}")

    # Track statistics
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(train_worker, task): task for task in tasks}

        with tqdm(total=len(futures), desc="Training Models", unit="model") as pbar:
            for future in as_completed(futures):
                success, message = future.result()

                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.warning(message)

                pbar.update(1)
                pbar.set_postfix(ok=success_count, fail=fail_count)

    # Summary
    logger.info("=" * 60)
    logger.info(f"Dataset generation complete!")
    logger.info(f"  Successfully trained: {success_count}")
    logger.info(f"  Failed:              {fail_count}")
    logger.info(f"  Total models:        {success_count} in {args.output_dir}")
    logger.info("=" * 60)

    # Check for duplicate weight combinations
    existing = get_existing_filenames(args.output_dir)
    weight_combos = {}
    for filename in existing:
        # Extract weights from filename (without sample number)
        parts = filename.rsplit('_n', 1)[0]  # Remove _n{num}.pth
        if parts in weight_combos:
            weight_combos[parts] += 1
        else:
            weight_combos[parts] = 1

    duplicates = {k: v for k, v in weight_combos.items() if v > 1}
    if duplicates:
        logger.info(f"Found {len(duplicates)} weight combinations with multiple samples:")
        for combo, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {combo}: {count} samples")


if __name__ == "__main__":
    main()
