"""
Split the MLP regression dataset into train/val sets.

Moves checkpoint files into dataset_train/ and dataset_val/ subdirectories.
Uses a deterministic 90/10 split so the RPG model is trained only on the
training conditions, and analysis uses held-out val conditions.

Usage:
    python split_dataset.py [--val_fraction 0.1] [--seed 42]
"""
import argparse
import json
import os
import shutil
import re

import numpy as np


def parse_condition_from_filename(filename: str) -> dict:
    """Extract (a, b, f, phi) from filename."""
    pattern = r"func_a([\d.-]+)_b([\d.-]+)_f([\d.-]+)_p([\d.-]+)\.pth"
    match = re.match(pattern, os.path.basename(filename))
    if not match:
        raise ValueError(f"Cannot parse: {filename}")
    return {
        "a": float(match.group(1)),
        "b": float(match.group(2)),
        "f": float(match.group(3)),
        "phi": float(match.group(4)),
    }


def main():
    parser = argparse.ArgumentParser(description="Split MLP dataset into train/val")
    parser.add_argument("--dataset_dir", type=str,
                        default="./dataset",
                        help="Directory containing all .pth checkpoints")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of conditions for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Get all checkpoint files
    all_files = sorted([f for f in os.listdir(args.dataset_dir) if f.endswith(".pth")])
    n_total = len(all_files)
    if n_total == 0:
        print(f"ERROR: No .pth files found in {args.dataset_dir}")
        return

    # Deterministic shuffle and split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val

    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    # Create output directories
    train_dir = os.path.join(args.dataset_dir + "_train")
    val_dir = os.path.join(args.dataset_dir + "_val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Move files
    train_conditions = []
    val_conditions = []

    for i, fname in enumerate(all_files):
        src = os.path.join(args.dataset_dir, fname)
        if i in val_indices:
            dst = os.path.join(val_dir, fname)
            val_conditions.append(parse_condition_from_filename(fname))
        else:
            dst = os.path.join(train_dir, fname)
            train_conditions.append(parse_condition_from_filename(fname))
        shutil.move(src, dst)

    # Save split metadata
    split_info = {
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "train_conditions": train_conditions,
        "val_conditions": val_conditions,
    }

    split_path = os.path.join(os.path.dirname(args.dataset_dir), "split_info.json")
    with open(split_path, "w") as fp:
        json.dump(split_info, fp, indent=2)

    print(f"Split complete:")
    print(f"  Total:      {n_total}")
    print(f"  Train:      {n_train} -> {train_dir}")
    print(f"  Validation: {n_val} -> {val_dir}")
    print(f"  Metadata:   {split_path}")


if __name__ == "__main__":
    main()
