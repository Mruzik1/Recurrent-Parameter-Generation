"""
Train a single PPO agent on LunarLanderContinuous-v3 with custom reward weights.
Implements the "Skill Composer" experiment from EXPERIMENTS.md.
"""
import argparse
import logging
import os
import random
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LunarLander PPO agent with reward shaping")
    # Reward weights (must sum to 1.0)
    parser.add_argument("--w_landing", type=float, default=0.4, help="Weight for safe landing")
    parser.add_argument("--w_fuel", type=float, default=0.3, help="Weight for fuel efficiency")
    parser.add_argument("--w_time", type=float, default=0.2, help="Weight for speed")
    parser.add_argument("--w_smoothness", type=float, default=0.1, help="Weight for smooth control")
    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=256, help="Steps per rollout per env")
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--update_epochs", type=int, default=10)
    # Output
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/lunar_lander/checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Actor-Critic PPO Agent for LunarLanderContinuous-v3.
    Architecture: [8 (State) -> 128 -> 128 -> 2 (Main/Side Engines)]
    """

    def __init__(self, obs_dim: int = 8, act_dim: int = 2):
        super().__init__()
        # Critic network (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        # Actor network (policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


class RewardShaper:
    """
    Shapes the reward based on weighted objectives.
    R = w_landing * R_landing + w_fuel * R_fuel + w_time * R_time + w_smoothness * R_smoothness
    """

    def __init__(self, w_landing: float, w_fuel: float, w_time: float, w_smoothness: float):
        # Normalize weights to sum to 1.0
        total = w_landing + w_fuel + w_time + w_smoothness
        self.w_landing = w_landing / total
        self.w_fuel = w_fuel / total
        self.w_time = w_time / total
        self.w_smoothness = w_smoothness / total

        self.prev_action = None
        self.episode_steps = 0

        logger.info(f"Reward weights: Landing={self.w_landing:.3f}, Fuel={self.w_fuel:.3f}, "
                   f"Time={self.w_time:.3f}, Smoothness={self.w_smoothness:.3f}")

    def reset(self):
        """Reset episode-specific tracking."""
        self.prev_action = None
        self.episode_steps = 0

    def shape_reward(self, obs: np.ndarray, action: np.ndarray, reward: float,
                     terminated: bool, truncated: bool) -> float:
        """
        Apply reward shaping based on observation and action.

        LunarLander obs: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
        LunarLander action: [main_engine (-1 to 1), side_engine (-1 to 1)]
        """
        self.episode_steps += 1

        # Component 1: Landing quality (original reward already contains landing bonus)
        r_landing = reward  # Original reward includes +100 for landing, -100 for crash

        # Component 2: Fuel efficiency (penalize engine use)
        engine_power = np.abs(action[0]) + np.abs(action[1])  # Total thrust
        r_fuel = -engine_power * 0.3  # Penalty for using engines

        # Component 3: Time efficiency (penalize long episodes)
        r_time = -0.01  # Small penalty per timestep

        # Component 4: Smoothness (penalize abrupt control changes)
        if self.prev_action is not None:
            action_delta = np.abs(action - self.prev_action).sum()
            r_smoothness = -action_delta * 0.5  # Penalty for jerky movements
        else:
            r_smoothness = 0.0

        self.prev_action = action.copy()

        # Weighted combination
        shaped_reward = (
            self.w_landing * r_landing +
            self.w_fuel * r_fuel +
            self.w_time * r_time +
            self.w_smoothness * r_smoothness
        )

        return shaped_reward


def make_filename(w_landing: float, w_fuel: float, w_time: float, w_smoothness: float,
                  sample_num: int = 0) -> str:
    """
    Generate standardized filename for a checkpoint.
    Format: lander_wl{w_landing}_wf{w_fuel}_wt{w_time}_ws{w_smoothness}_n{sample_num}.pth
    """
    return (f"lander_wl{w_landing:.3f}_wf{w_fuel:.3f}_wt{w_time:.3f}_"
            f"ws{w_smoothness:.3f}_n{sample_num}.pth")


def get_next_available_filename(output_dir: str, w_landing: float, w_fuel: float,
                                w_time: float, w_smoothness: float) -> str:
    """
    Find next available filename by checking existing files.
    Increments sample_num until an unused filename is found.
    """
    sample_num = 0
    while True:
        filename = make_filename(w_landing, w_fuel, w_time, w_smoothness, sample_num)
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        sample_num += 1


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Create reward shaper
    shaper = RewardShaper(args.w_landing, args.w_fuel, args.w_time, args.w_smoothness)

    # Environment setup - vectorized for parallel execution
    def make_env():
        env = gym.make("LunarLanderContinuous-v3")
        return env

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(args.num_envs)])

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    num_envs = args.num_envs

    # Agent setup
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Hyperparameters
    num_steps = args.num_steps
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Pre-allocate storage tensors
    obs_storage = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_storage = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs_storage = torch.zeros((num_steps, num_envs), device=device)
    rewards_storage = torch.zeros((num_steps, num_envs), device=device)
    dones_storage = torch.zeros((num_steps, num_envs), device=device)
    values_storage = torch.zeros((num_steps, num_envs), device=device)

    # Initialize
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(next_obs).float().to(device)
    next_done = torch.zeros(num_envs, device=device)

    # Progress tracking
    episode_rewards = []
    episode_reward_buffer = np.zeros(num_envs)
    shapers = [RewardShaper(args.w_landing, args.w_fuel, args.w_time, args.w_smoothness)
               for _ in range(num_envs)]
    start_time = time.time()

    # Optional tqdm
    if not args.quiet:
        try:
            from tqdm import tqdm
            pbar = tqdm(range(num_updates), desc="Training", unit="update")
        except ImportError:
            pbar = range(num_updates)
    else:
        pbar = range(num_updates)

    for update in pbar:
        # Collect rollout
        for step in range(num_steps):
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            actions_storage[step] = action
            logprobs_storage[step] = logprob
            values_storage[step] = value.squeeze(-1)

            # Step all environments
            next_obs_np, raw_rewards, terminateds, truncateds, infos = envs.step(action.cpu().numpy())

            # Apply reward shaping per environment
            shaped_rewards = []
            for i in range(num_envs):
                shaped_r = shapers[i].shape_reward(
                    next_obs_np[i],
                    action[i].cpu().numpy(),
                    raw_rewards[i],
                    terminateds[i],
                    truncateds[i]
                )
                shaped_rewards.append(shaped_r)

                if terminateds[i] or truncateds[i]:
                    shapers[i].reset()

            shaped_rewards = np.array(shaped_rewards)

            # Convert to tensors
            rewards_storage[step] = torch.from_numpy(shaped_rewards).to(device)
            next_obs = torch.from_numpy(next_obs_np).float().to(device)
            next_done = torch.from_numpy(np.logical_or(terminateds, truncateds)).float().to(device)

            # Track episode rewards (using raw rewards for monitoring)
            episode_reward_buffer += raw_rewards
            for i, (term, trunc) in enumerate(zip(terminateds, truncateds)):
                if term or trunc:
                    episode_rewards.append(episode_reward_buffer[i])
                    episode_reward_buffer[i] = 0.0

        # Compute GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros((num_steps, num_envs), device=device)
            lastgaelam = torch.zeros(num_envs, device=device)

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    nextvalues = values_storage[t + 1]

                delta = rewards_storage[t] + gamma * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values_storage

        # Optimize policy
        b_obs = obs_storage.reshape((-1, obs_dim))
        b_actions = actions_storage.reshape((-1, act_dim))
        b_logprobs = logprobs_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue.squeeze() - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # Logging
        if not args.quiet and hasattr(pbar, 'set_postfix') and episode_rewards:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                avg_ret=f"{avg_reward:.1f}",
                eps=len(episode_rewards)
            )

    envs.close()

    # Determine output path with collision handling
    if args.output_path is None:
        output_path = get_next_available_filename(
            args.output_dir, args.w_landing, args.w_fuel, args.w_time, args.w_smoothness
        )
    else:
        output_path = args.output_path

    # Save model (only actor, not critic)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save only the actor network for the RPG dataset
    policy_state_dict = {
        'actor_mean.0.weight': agent.actor_mean[0].weight,
        'actor_mean.0.bias': agent.actor_mean[0].bias,
        'actor_mean.2.weight': agent.actor_mean[2].weight,
        'actor_mean.2.bias': agent.actor_mean[2].bias,
        'actor_mean.4.weight': agent.actor_mean[4].weight,
        'actor_mean.4.bias': agent.actor_mean[4].bias,
    }

    torch.save(policy_state_dict, output_path)

    elapsed = time.time() - start_time
    final_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
    logger.info(f"Training complete in {elapsed:.1f}s | Episodes: {len(episode_rewards)} | Avg Reward: {final_reward:.1f}")
    logger.info(f"Model saved to {output_path}")

    return agent, episode_rewards


if __name__ == "__main__":
    args = parse_args()
    train(args)
