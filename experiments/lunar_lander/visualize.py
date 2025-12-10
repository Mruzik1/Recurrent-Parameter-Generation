"""
Visualize and compare LunarLander policies with different reward weights.
Can run multiple policies side-by-side or sequentially.
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    """
    Policy network for LunarLanderContinuous-v3.
    Architecture: [8 (State) -> 128 -> 128 -> 2 (Main/Side Engines)]
    """

    def __init__(self, obs_dim: int = 8, act_dim: int = 2):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )

    def get_action(self, x):
        """Get action from observation (deterministic for evaluation)."""
        return self.actor_mean(x)


def parse_weights_from_filename(filename: str) -> dict:
    """Parse reward weights from checkpoint filename."""
    pattern = r"lander_wl([\d.]+)_wf([\d.]+)_wt([\d.]+)_ws([\d.]+)_n(\d+)\.pth"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename '{filename}' doesn't match expected pattern")

    return {
        "w_landing": float(match.group(1)),
        "w_fuel": float(match.group(2)),
        "w_time": float(match.group(3)),
        "w_smoothness": float(match.group(4)),
        "sample_num": int(match.group(5)),
    }


def load_policy(checkpoint_path: str, device):
    """Load a policy from checkpoint."""
    policy = PolicyNetwork(obs_dim=8, act_dim=2).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def get_policy_label(weights: dict) -> str:
    """Generate a short label for a policy based on its weights."""
    # Find dominant weight
    weight_items = [
        ("Landing", weights["w_landing"]),
        ("Fuel", weights["w_fuel"]),
        ("Time", weights["w_time"]),
        ("Smooth", weights["w_smoothness"]),
    ]
    weight_items.sort(key=lambda x: -x[1])

    # If one weight dominates (>0.6), use that as label
    if weight_items[0][1] > 0.6:
        return f"{weight_items[0][0]}-focused"
    # If two weights dominate, show both
    elif weight_items[0][1] + weight_items[1][1] > 0.8:
        return f"{weight_items[0][0]}+{weight_items[1][0]}"
    else:
        return "Balanced"


def visualize_single(checkpoint_path: str, num_episodes: int = 3):
    """Visualize a single policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy
    policy = load_policy(checkpoint_path, device)

    # Parse weights
    filename = os.path.basename(checkpoint_path)
    weights = parse_weights_from_filename(filename)
    label = get_policy_label(weights)

    print(f"Visualizing policy: {label}")
    print(f"Weights: Landing={weights['w_landing']:.2f}, Fuel={weights['w_fuel']:.2f}, "
          f"Time={weights['w_time']:.2f}, Smoothness={weights['w_smoothness']:.2f}")

    # Create environment with rendering
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # Run episodes
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        print(f"\nEpisode {ep + 1}:")
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy.get_action(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"  Reward: {total_reward:.1f}, Steps: {steps}")

    env.close()


def compare_policies(checkpoint_paths: list, num_episodes: int = 3):
    """Compare multiple policies by running them sequentially."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policies = []
    labels = []

    # Load all policies
    for path in checkpoint_paths:
        policy = load_policy(path, device)
        filename = os.path.basename(path)
        weights = parse_weights_from_filename(filename)
        label = get_policy_label(weights)

        policies.append(policy)
        labels.append(label)

        print(f"\nPolicy {len(policies)}: {label}")
        print(f"  Weights: L={weights['w_landing']:.2f}, F={weights['w_fuel']:.2f}, "
              f"T={weights['w_time']:.2f}, S={weights['w_smoothness']:.2f}")

    # Run comparison
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    for policy_idx, (policy, label) in enumerate(zip(policies, labels)):
        print(f"\n{'='*60}")
        print(f"Testing Policy {policy_idx + 1}: {label}")
        print('='*60)

        episode_rewards = []

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = policy.get_action(obs_tensor)
                obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            print(f"  Episode {ep + 1}: Reward={total_reward:.1f}, Steps={steps}")

        avg_reward = np.mean(episode_rewards)
        print(f"  Average: {avg_reward:.1f}")

    env.close()


def benchmark_policies(checkpoint_dir: str, num_episodes: int = 10):
    """
    Benchmark all policies in a directory without rendering.
    Creates a visualization of performance vs. weight distribution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all checkpoints
    checkpoint_paths = list(Path(checkpoint_dir).glob("lander_*.pth"))

    if not checkpoint_paths:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints. Benchmarking...")

    results = []

    for path in checkpoint_paths:
        policy = load_policy(str(path), device)
        filename = path.name
        weights = parse_weights_from_filename(filename)

        # Test policy
        env = gym.make("LunarLanderContinuous-v3")
        episode_rewards = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = policy.get_action(obs_tensor)
                obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)

        env.close()

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        results.append({
            "filename": filename,
            "weights": weights,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
        })

        print(f"{filename}: {avg_reward:.1f} ± {std_reward:.1f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Policy Performance vs. Reward Weights", fontsize=16)

    weight_names = ["w_landing", "w_fuel", "w_time", "w_smoothness"]
    weight_labels = ["Landing", "Fuel Efficiency", "Time", "Smoothness"]

    for idx, (weight_name, weight_label) in enumerate(zip(weight_names, weight_labels)):
        ax = axes[idx // 2, idx % 2]

        x = [r["weights"][weight_name] for r in results]
        y = [r["avg_reward"] for r in results]
        yerr = [r["std_reward"] for r in results]

        ax.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.6)
        ax.set_xlabel(f"{weight_label} Weight", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(f"Performance vs. {weight_label} Weight")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=150)
    print(f"\nVisualization saved to policy_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize LunarLander policies")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to visualize")
    parser.add_argument("--compare", type=str, nargs="+", help="Multiple checkpoints to compare")
    parser.add_argument("--benchmark", type=str, help="Directory of checkpoints to benchmark")
    parser.add_argument("--num_episodes", type=int, default=3, help="Episodes per policy")
    args = parser.parse_args()

    if args.checkpoint:
        visualize_single(args.checkpoint, args.num_episodes)
    elif args.compare:
        compare_policies(args.compare, args.num_episodes)
    elif args.benchmark:
        benchmark_policies(args.benchmark, num_episodes=10)
    else:
        print("Error: Must specify --checkpoint, --compare, or --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()
