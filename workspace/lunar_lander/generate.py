"""
Generate LunarLander policies for specific reward weight combinations using trained RPG model.
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
from dataset import LunarLander_PPO as Dataset
from experiments.lunar_lander.train_single import make_filename


def load_model(checkpoint_path: str):
    """Load trained RPG model from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    num_permutation = state_dict["to_permutation_state.weight"].shape[0]
    sequence_length = state_dict["model.pe"].shape[1]
    positional_embedding = state_dict["model.pe"][0]

    print(f"Checkpoint info:")
    print(f"  - Permutation states: {num_permutation}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Embedding dim: {positional_embedding.shape[1]}")

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

    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    print(f"Using device: {device}")

    return model


def generate_policy(model, w_landing, w_fuel, w_time, w_smoothness,
                    output_dir="./experiments/lunar_lander/test_generated"):
    """Generate a policy for specific reward weights."""
    device = next(model.parameters()).device

    condition = torch.tensor(
        [[w_landing, w_fuel, w_time, w_smoothness]],
        dtype=torch.float32,
        device=device
    )

    print(f"\nGenerating policy for: wl={w_landing:.3f}, wf={w_fuel:.3f}, wt={w_time:.3f}, ws={w_smoothness:.3f}")
    with torch.no_grad():
        params = model(sample=True, condition=condition)

    dataset = Dataset(dim_per_token=2048)

    filename = make_filename(w_landing, w_fuel, w_time, w_smoothness)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    dataset.save_params(params, save_path=save_path)
    print(f"Saved to: {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Generate LunarLander policies using trained RPG")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/lunar_lander_reward_adapter.pth",
                       help="Path to trained RPG checkpoint")
    parser.add_argument("--w_landing", type=float, default=0.4, help="Landing weight")
    parser.add_argument("--w_fuel", type=float, default=0.3, help="Fuel efficiency weight")
    parser.add_argument("--w_time", type=float, default=0.2, help="Time efficiency weight")
    parser.add_argument("--w_smoothness", type=float, default=0.1, help="Smoothness weight")
    parser.add_argument("--output_dir", type=str, default="./experiments/lunar_lander/test_generated")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of random samples to generate")
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    if args.num_samples == 1:
        generate_policy(model, args.w_landing, args.w_fuel, args.w_time, args.w_smoothness, args.output_dir)
    else:
        for i in range(args.num_samples):
            weights = np.random.dirichlet([1.0] * 4)
            generate_policy(model, weights[0], weights[1], weights[2], weights[3], args.output_dir)


if __name__ == "__main__":
    main()
