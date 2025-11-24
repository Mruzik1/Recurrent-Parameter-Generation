"""
Generate a dataset of BipedalWalker policy checkpoints with randomized environment parameters.
Uses multiprocessing for parallel training.
"""
import argparse
import logging
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from env_utils import make_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Parameter ranges (from EXPERIMENTS.md)
PARAM_RANGES = {
    "leg_length": (0.5, 2.0),
    "leg_width": (0.1, 0.5),
    "gravity": (-20.0, -5.0),
    "friction": (0.1, 5.0),
}


def sample_parameters() -> dict:
    """Sample random environment parameters within defined ranges."""
    return {
        name: random.uniform(*bounds)
        for name, bounds in PARAM_RANGES.items()
    }


def train_worker(task: tuple) -> tuple:
    """
    Worker function for training a single agent.
    
    Args:
        task: (index, params_dict, output_dir, total_timesteps)
    
    Returns:
        (success: bool, message: str)
    """
    idx, params, output_dir, total_timesteps = task
    
    filename = make_filename(**params)
    output_path = os.path.join(output_dir, filename)
    
    # Skip if already exists
    if os.path.exists(output_path):
        return (True, f"Skipped {filename} (exists)")
    
    # Build command
    script_path = Path(__file__).parent / "train_single.py"
    cmd = [
        sys.executable, str(script_path),
        "--leg_length", str(params["leg_length"]),
        "--leg_width", str(params["leg_width"]),
        "--gravity", str(params["gravity"]),
        "--friction", str(params["friction"]),
        "--output_path", output_path,
        "--total_timesteps", str(total_timesteps),
        "--seed", str(idx),
        "--quiet",  # Suppress progress bars in workers
    ]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=600,  # 10 minute timeout
        )
        return (True, filename)
    except subprocess.TimeoutExpired:
        return (False, f"Timeout: {filename}")
    except subprocess.CalledProcessError as e:
        return (False, f"Failed {filename}: {e.stderr.decode()[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Generate BipedalWalker dataset")
    parser.add_argument("--num_samples", type=int, default=10_000, help="Number of models to train")
    parser.add_argument("--output_dir", type=str, default="experiments/bipedal_walker/dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--total_timesteps", type=int, default=100_000, help="Training steps per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter sampling")
    args = parser.parse_args()
    
    # Seed for reproducible parameter sampling
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate tasks
    tasks = []
    for i in range(args.num_samples):
        params = sample_parameters()
        tasks.append((i, params, args.output_dir, args.total_timesteps))
    
    logger.info(f"Generating {args.num_samples} samples with {args.num_workers} workers")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Timesteps per model: {args.total_timesteps}")
    
    # Track statistics
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(train_worker, task): task for task in tasks}
        
        with tqdm(total=len(futures), desc="Training Models", unit="model") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                
                if success:
                    if "Skipped" in message:
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    logger.warning(message)
                
                pbar.update(1)
                pbar.set_postfix(ok=success_count, skip=skip_count, fail=fail_count)
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"Dataset generation complete!")
    logger.info(f"  Trained: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Failed:  {fail_count}")
    logger.info(f"  Total:   {success_count + skip_count} models in {args.output_dir}")


if __name__ == "__main__":
    main()
