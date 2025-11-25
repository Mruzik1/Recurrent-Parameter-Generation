"""
Evaluate a BipedalWalker policy checkpoint.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from env_utils import make_custom_env, parse_params_from_filename


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-Critic PPO Agent."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_action(self, x):
        """Get action from observation (deterministic for evaluation)."""
        return self.actor_mean(x)


def test_policy(checkpoint_path: str, num_episodes: int = 10, render: bool = False):
    """Test a policy checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load agent
    agent = Agent(obs_dim=24, act_dim=4).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    agent.eval()

    # Parse environment parameters from filename
    filename = os.path.basename(checkpoint_path)
    try:
        params = parse_params_from_filename(filename)
        print(f"Testing with: L={params['leg_length']:.2f}, W={params['leg_width']:.2f}, "
              f"g={params['gravity']:.2f}, μ={params['friction']:.2f}")
    except ValueError:
        # Use default parameters if filename doesn't match expected format
        params = {"leg_length": 1.13, "leg_width": 0.26, "gravity": -10.0, "friction": 2.5}
        print("Using default environment parameters")

    # Create environment
    render_mode = "human" if render else None
    env = make_custom_env(**params, render_mode=render_mode)

    # Run evaluation
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_tensor)
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
    parser = argparse.ArgumentParser(description="Test BipedalWalker policy")
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
