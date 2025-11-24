"""
Train a single PPO agent on BipedalWalker with custom environment parameters.
Optimized for speed with pre-allocated tensors and minimal Python overhead.
"""
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from env_utils import make_custom_env, make_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BipedalWalker PPO agent")
    # Environment parameters
    parser.add_argument("--leg_length", type=float, default=1.13)
    parser.add_argument("--leg_width", type=float, default=0.26)
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--friction", type=float, default=2.5)
    # Training parameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--update_epochs", type=int, default=10)
    # Output
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args()


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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


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
    
    # Environment setup
    env = make_custom_env(args.leg_length, args.leg_width, args.gravity, args.friction)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    
    # Agent setup
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Hyperparameters
    num_steps = args.num_steps
    batch_size = num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // num_steps
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Pre-allocate storage tensors (faster than lists)
    obs_storage = torch.zeros((num_steps, obs_dim), device=device)
    actions_storage = torch.zeros((num_steps, act_dim), device=device)
    logprobs_storage = torch.zeros(num_steps, device=device)
    rewards_storage = torch.zeros(num_steps, device=device)
    dones_storage = torch.zeros(num_steps, device=device)
    values_storage = torch.zeros(num_steps, device=device)
    
    # Initialize
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(1, device=device)
    
    # Progress tracking
    episode_rewards = []
    current_episode_reward = 0.0
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
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
            
            actions_storage[step] = action.squeeze(0)
            logprobs_storage[step] = logprob
            values_storage[step] = value.squeeze()
            
            # Step environment
            next_obs_np, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
            done = terminated or truncated
            
            rewards_storage[step] = reward
            current_episode_reward += reward
            
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(float(done), device=device)
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                next_obs_np, _ = env.reset()
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        
        # Compute GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).squeeze()
            advantages = torch.zeros(num_steps, device=device)
            lastgaelam = 0.0
            
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
        b_inds = np.arange(batch_size)
        
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_storage[mb_inds], actions_storage[mb_inds]
                )
                logratio = newlogprob - logprobs_storage[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue.squeeze() - returns[mb_inds]) ** 2).mean()
                
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
    
    env.close()
    
    # Determine output path
    if args.output_path is None:
        filename = make_filename(args.leg_length, args.leg_width, args.gravity, args.friction)
        output_path = os.path.join(args.output_dir, filename)
    else:
        output_path = args.output_path
    
    # Save model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(agent.state_dict(), output_path)
    
    elapsed = time.time() - start_time
    final_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
    logger.info(f"Training complete in {elapsed:.1f}s | Episodes: {len(episode_rewards)} | Avg Reward: {final_reward:.1f}")
    logger.info(f"Model saved to {output_path}")
    
    # Optional visualization
    if args.visualize:
        visualize_agent(agent, args, device)
    
    return agent, episode_rewards


def visualize_agent(agent, args, device):
    """Run visualization of trained agent."""
    logger.info("Starting visualization...")
    env = make_custom_env(args.leg_length, args.leg_width, args.gravity, args.friction, render_mode="human")
    
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
            done = terminated or truncated
            total_reward += reward
            env.render()
        
        logger.info(f"Episode {ep + 1}: Reward = {total_reward:.1f}")
    
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
