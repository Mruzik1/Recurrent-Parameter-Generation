"""
Generate BipedalWalker policies for specific morphologies using trained RPG model.
"""
import sys, os

root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)

import argparse
import numpy as np
import torch

from model import MambaDiffusion as Model
from model import DDPMSampler
from dataset import BipedalWalker_PPO as Dataset
from experiments.bipedal_walker.env_utils import make_filename


def load_model(checkpoint_path: str):
    """Load trained RPG model from checkpoint (no dataset needed)."""
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract architecture info from checkpoint
    num_permutation = state_dict["to_permutation_state.weight"].shape[0]
    sequence_length = state_dict["model.pe"].shape[1]  # [1, seq_len, d_model]
    positional_embedding = state_dict["model.pe"][0]  # Remove batch dim: [1, seq, dim] -> [seq, dim]

    print(f"Checkpoint info:")
    print(f"  - Permutation states: {num_permutation}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Embedding dim: {positional_embedding.shape[1]}")

    # Model config (must match training)
    config = {
        "num_permutation": num_permutation,
        "d_condition": 4,
        "d_model": 2048,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        "diffusion_batch": 256,
        "layer_channels": [1, 16, 32, 64, 32, 16, 1],
        "model_dim": 2048,
        "condition_dim": 2048,
        "kernel_size": 7,
        "beta": (0.0001, 0.02),
        "T": 500,
        "sample_mode": DDPMSampler,
        "forward_once": True,
    }

    Model.config = config
    model = Model(
        sequence_length=sequence_length,
        positional_embedding=positional_embedding
    )

    # Load checkpoint state
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    print(f"Using device: {device}")

    return model


def generate_policy(model, leg_length, leg_width, gravity, friction, output_dir="./experiments/bipedal_walker/test_generated"):
    """Generate a policy for specific morphology."""
    device = next(model.parameters()).device

    # Create condition vector
    condition = torch.tensor(
        [[leg_length, leg_width, gravity, friction]],
        dtype=torch.float32,
        device=device
    )

    # Generate parameters
    print(f"\nGenerating policy for: L={leg_length:.2f}, W={leg_width:.2f}, g={gravity:.2f}, μ={friction:.2f}")
    with torch.no_grad():
        params = model(sample=True, condition=condition)

    # Save checkpoint - need to load dataset just for save_params
    # (this reconstructs the network structure from tokenized params)
    from dataset import BipedalWalker_PPO as Dataset
    dataset = Dataset(dim_per_token=1024)

    filename = make_filename(leg_length, leg_width, gravity, friction)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    dataset.save_params(params, save_path=save_path)
    print(f"Saved to: {save_path}")

    return save_path


def generate_sweep(model, param_name, start, end, num_steps, output_dir):
    """Generate policies for a parameter sweep."""
    # Default parameters
    defaults = {"leg_length": 1.13, "leg_width": 0.26, "gravity": -10.0, "friction": 2.5}

    print(f"\n==> Sweeping {param_name} from {start} to {end} ({num_steps} steps)")

    values = np.linspace(start, end, num_steps)
    paths = []

    for val in values:
        params = defaults.copy()
        params[param_name] = val
        path = generate_policy(model, output_dir=output_dir, **params)
        paths.append(path)

    print(f"\n==> Generated {len(paths)} policies")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate BipedalWalker policies")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/bipedal_walker_morphology_adapter.pth",
                        help="Path to trained RPG model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./experiments/bipedal_walker/test_generated",
                        help="Output directory for generated policies")

    # Single generation
    parser.add_argument("--leg_length", type=float, default=None)
    parser.add_argument("--leg_width", type=float, default=None)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--friction", type=float, default=None)

    # Sweep generation
    parser.add_argument("--sweep", type=str, choices=["leg_length", "leg_width", "gravity", "friction"],
                        help="Parameter to sweep")
    parser.add_argument("--start", type=float, help="Sweep start value")
    parser.add_argument("--end", type=float, help="Sweep end value")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps in sweep")

    args = parser.parse_args()

    # Load model (no dataset needed for inference!)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train the model first using: bash workspace/launch.sh workspace/bipedal_walker/train.py '0'")
        return

    print("==> Loading model from checkpoint...")
    model = load_model(args.checkpoint)

    # Generate
    if args.sweep:
        # Sweep mode
        if args.start is None or args.end is None:
            print("Error: --start and --end required for sweep mode")
            return
        generate_sweep(model, args.sweep, args.start, args.end, args.num_steps, args.output_dir)

    elif all(v is not None for v in [args.leg_length, args.leg_width, args.gravity, args.friction]):
        # Single generation mode
        generate_policy(model, args.leg_length, args.leg_width, args.gravity, args.friction, args.output_dir)

    else:
        # Random generation (use model's default behavior)
        print("\n==> Generating random policy...")
        with torch.no_grad():
            params = model(sample=True)

        # Load dataset just for saving
        dataset = Dataset(dim_per_token=1024)
        save_path = os.path.join(args.output_dir, "generated_walker_random.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        dataset.save_params(params, save_path=save_path)
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
