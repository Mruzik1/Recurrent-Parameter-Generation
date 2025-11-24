"""
Visualize a trained BipedalWalker agent.
Automatically infers environment parameters from the checkpoint filename.
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from env_utils import make_custom_env, parse_params_from_filename, DEFAULT_LEG_LENGTH, DEFAULT_LEG_WIDTH, DEFAULT_GRAVITY, DEFAULT_FRICTION
from train_single import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained BipedalWalker agent")
    parser.add_argument("model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("--leg_length", type=float, default=None, help="Override leg length")
    parser.add_argument("--leg_width", type=float, default=None, help="Override leg width")
    parser.add_argument("--gravity", type=float, default=None, help="Override gravity")
    parser.add_argument("--friction", type=float, default=None, help="Override friction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Try to infer parameters from filename
    filename = os.path.basename(args.model_path)
    
    try:
        inferred_params = parse_params_from_filename(filename)
        logger.info(f"Inferred parameters from filename: {inferred_params}")
    except ValueError:
        logger.info("Could not infer parameters from filename, using defaults")
        inferred_params = {
            "leg_length": DEFAULT_LEG_LENGTH,
            "leg_width": DEFAULT_LEG_WIDTH,
            "gravity": DEFAULT_GRAVITY,
            "friction": DEFAULT_FRICTION,
        }
    
    # Apply any command-line overrides
    params = {
        "leg_length": args.leg_length if args.leg_length is not None else inferred_params["leg_length"],
        "leg_width": args.leg_width if args.leg_width is not None else inferred_params["leg_width"],
        "gravity": args.gravity if args.gravity is not None else inferred_params["gravity"],
        "friction": args.friction if args.friction is not None else inferred_params["friction"],
    }
    
    logger.info(f"Environment: L={params['leg_length']:.2f}, W={params['leg_width']:.2f}, "
                f"g={params['gravity']:.2f}, μ={params['friction']:.2f}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment for dimension info
    env = make_custom_env(**params, render_mode="human")
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    
    # Load agent
    agent = Agent(obs_dim, act_dim).to(device)
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(state_dict)
        logger.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        env.close()
        return 1
    
    agent.eval()
    
    # Run episodes
    all_rewards = []
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze(0))
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            env.render()
        
        all_rewards.append(total_reward)
        logger.info(f"Episode {ep + 1}/{args.episodes}: Steps={steps}, Reward={total_reward:.1f}")
    
    env.close()
    
    # Summary
    logger.info(f"Average Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
