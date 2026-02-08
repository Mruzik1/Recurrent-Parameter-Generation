"""
Reward-weighted self-training utilities for BipedalWalker RPG.

Provides:
- RewardBuffer: stores generated policies with their environment rewards
- evaluate_policy_reward: in-process policy evaluation (no subprocess)
- condition_to_env_params: convert condition tensor to env parameter dict
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Resolve project paths for env_utils import
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
_bw_dir = os.path.join(_project_root, 'experiments', 'bipedal_walker')
if _bw_dir not in sys.path:
    sys.path.insert(0, _bw_dir)


# ─── Agent definition (matches training architecture) ───

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Actor-Critic agent for BipedalWalker (obs=24, act=4)."""
    def __init__(self, obs_dim=24, act_dim=4):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, 64)), nn.Tanh(),
            _layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_action(self, x):
        return self.actor_mean(x)


# ─── Reward Buffer ───

class RewardBuffer:
    """
    Fixed-capacity buffer storing generated policies with their environment rewards.
    On overflow, the lowest-reward entry is evicted (keeps the best policies).
    """

    def __init__(self, capacity=50, temperature=1.0):
        """
        Args:
            capacity: Maximum number of entries to store.
            temperature: Controls sigmoid steepness for reward weighting.
                         Higher = sharper distinction between good/bad.
        """
        self.capacity = capacity
        self.temperature = temperature
        self.buffer = []  # list of (params_tensor, condition_tensor, reward_float)
        self._baseline = 0.0
        self._std = 1.0

    def add(self, params, condition, reward):
        """Add a generated policy and its evaluation reward."""
        entry = (
            params.detach().cpu().clone(),
            condition.detach().cpu().clone(),
            float(reward),
        )
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            # Evict lowest-reward entry
            self.buffer.sort(key=lambda x: x[2])
            self.buffer.pop(0)
        self._update_stats()

    def _update_stats(self):
        if not self.buffer:
            return
        rewards = [r for _, _, r in self.buffer]
        self._baseline = float(np.mean(rewards))
        self._std = max(float(np.std(rewards)), 1.0)

    @property
    def reward_baseline(self):
        return self._baseline

    @property
    def reward_std(self):
        return self._std

    def sample(self, device=None):
        """Sample a random entry. Returns (params, condition, reward)."""
        idx = np.random.randint(len(self.buffer))
        params, condition, reward = self.buffer[idx]
        if device is not None:
            params = params.to(device)
            condition = condition.to(device)
        return params, condition, reward

    def sample_best(self, device=None):
        """Sample the highest-reward entry."""
        best_idx = max(range(len(self.buffer)), key=lambda i: self.buffer[i][2])
        params, condition, reward = self.buffer[best_idx]
        if device is not None:
            params = params.to(device)
            condition = condition.to(device)
        return params, condition, reward

    def compute_weight(self, reward):
        """
        Compute reward weight via sigmoid on normalized advantage.
        Returns value in (0, 1):
          - advantage > 0 (above baseline) → weight > 0.5
          - advantage < 0 (below baseline) → weight < 0.5
          - advantage == 0 → weight = 0.5
        """
        advantage = (reward - self._baseline) / self._std
        weight = 1.0 / (1.0 + np.exp(-self.temperature * advantage))
        return float(weight)

    def __len__(self):
        return len(self.buffer)

    def stats(self):
        """Return summary statistics of the buffer."""
        if not self.buffer:
            return {"size": 0, "baseline": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}
        rewards = [r for _, _, r in self.buffer]
        return {
            "size": len(self.buffer),
            "baseline": self._baseline,
            "std": self._std,
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        }


# ─── Environment evaluation ───

def condition_to_env_params(condition):
    """Convert a [4]-shaped condition tensor to environment parameter dict."""
    if isinstance(condition, torch.Tensor):
        condition = condition.detach().cpu()
    return {
        "leg_length": float(condition[0]),
        "leg_width": float(condition[1]),
        "gravity": float(condition[2]),
        "friction": float(condition[3]),
    }


def evaluate_policy_reward(state_dict, env_params, num_episodes=3):
    """
    Evaluate a policy in BipedalWalker in-process (no subprocess).

    Args:
        state_dict: Agent state dict (from postprocess or torch.load).
        env_params: Dict with keys leg_length, leg_width, gravity, friction.
        num_episodes: Number of episodes to average over.

    Returns:
        Mean reward (float). Returns -200.0 on any failure.
    """
    from env_utils import make_custom_env

    try:
        agent = Agent(obs_dim=24, act_dim=4)
        # Handle tensors that may still be nn.Parameters
        clean_dict = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                      for k, v in state_dict.items()}
        agent.load_state_dict(clean_dict)
        agent.eval()

        env = make_custom_env(**env_params, render_mode=None)

        rewards = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done and steps < 1600:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = agent.get_action(obs_t)
                obs, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy().squeeze(0)
                )
                total_reward += reward
                done = terminated or truncated
                steps += 1
            rewards.append(total_reward)

        env.close()
        return float(np.mean(rewards))

    except Exception as e:
        print(f"  [reward_utils] Evaluation failed: {e}")
        return -200.0


def evaluate_generated_tensor(prediction, train_set, condition, num_episodes=3):
    """
    Full pipeline: generated tensor → postprocess → state_dict → evaluate.

    Args:
        prediction: Model output tensor, shape [1, seq_len, dim_per_token].
        train_set: Dataset instance (for postprocess).
        condition: Condition tensor, shape [4] or [1, 4].

    Returns:
        (state_dict, reward) tuple.
    """
    # Convert to state dict
    params_flat = prediction.squeeze(0).cpu().to(torch.float32)
    state_dict = train_set.postprocess(params_flat)

    # Get env params
    cond = condition.squeeze(0) if condition.dim() > 1 else condition
    env_params = condition_to_env_params(cond)

    # Evaluate
    reward = evaluate_policy_reward(state_dict, env_params, num_episodes=num_episodes)
    return state_dict, reward
