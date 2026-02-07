"""
Evaluate a LunarLander policy checkpoint.
Tests the trained agent over multiple episodes and reports statistics.
"""
import argparse
import os
import re
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
    Policy network for LunarLanderContinuous-v2.
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
    """
    Parse reward weights from checkpoint filename.

    Expected format: lander_wl{w_landing}_wf{w_fuel}_wt{w_time}_ws{w_smoothness}_n{sample_num}.pth

    Args:
        filename: The checkpoint filename (not full path)

    Returns:
        Dictionary with keys: w_landing, w_fuel, w_time, w_smoothness, sample_num

    Raises:
        ValueError: If filename doesn't match expected format
    """
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


def test_policy(checkpoint_path: str, num_episodes: int = 10, render: bool = False):
    """Test a policy checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy network
    policy = PolicyNetwork(obs_dim=8, act_dim=2).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Parse weights from filename
    filename = os.path.basename(checkpoint_path)
    try:
        weights = parse_weights_from_filename(filename)
        print(f"Testing policy with weights: Landing={weights['w_landing']:.3f}, "
              f"Fuel={weights['w_fuel']:.3f}, Time={weights['w_time']:.3f}, "
              f"Smoothness={weights['w_smoothness']:.3f} (Sample #{weights['sample_num']})")
    except ValueError:
        print("Could not parse weights from filename, using default environment")

    # Create environment
    render_mode = "human" if render else None
    env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)

    # Run evaluation
    rewards = []
    for ep in range(num_episodes):
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

        rewards.append(total_reward)
        print(f"Episode {ep+1}: {total_reward:.2f}")

    env.close()

    # Report statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Mean: {mean_reward:.2f}")
    print(f"  Std:  {std_reward:.2f}")
    print(f"  Min:  {np.min(rewards):.2f}")
    print(f"  Max:  {np.max(rewards):.2f}")

    return mean_reward


def main():
    parser = argparse.ArgumentParser(description="Test LunarLander policy")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    test_policy(args.checkpoint, args.num_episodes, args.render)


if __name__ == "__main__":
    main()
