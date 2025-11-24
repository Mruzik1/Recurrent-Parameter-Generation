"""
Environment utilities for creating custom BipedalWalker environments.
"""
import gymnasium as gym
import gymnasium.envs.box2d.bipedal_walker as bw_module
from Box2D.b2 import polygonShape


# Default BipedalWalker constants (for reference)
DEFAULT_LEG_LENGTH = 1.13
DEFAULT_LEG_WIDTH = 0.26
DEFAULT_GRAVITY = -10.0
DEFAULT_FRICTION = 2.5

# Store original values to compute dependent constants
_ORIGINAL_LEG_H = 34 / 30  # Original LEG_H = 34/SCALE from gymnasium
_ORIGINAL_LEG_W = 8 / 30   # Original LEG_W = 8/SCALE from gymnasium
_ORIGINAL_LEG_DOWN = -8 / 30  # Original LEG_DOWN = -8/SCALE


def make_custom_env(
    leg_length: float = DEFAULT_LEG_LENGTH,
    leg_width: float = DEFAULT_LEG_WIDTH,
    gravity: float = DEFAULT_GRAVITY,
    friction: float = DEFAULT_FRICTION,
    render_mode: str = None
) -> gym.Env:
    """
    Creates a BipedalWalker-v3 environment with custom physical parameters.
    
    Args:
        leg_length: Length of the walker's legs (default: 1.13)
        leg_width: Width of the walker's legs (default: 0.26)
        gravity: Y-component of gravity, should be negative (default: -10.0)
        friction: Ground friction coefficient (default: 2.5)
        render_mode: 'human' for visualization, 'rgb_array' for frames, None for headless
    
    Returns:
        Configured gymnasium environment
    """
    # Modify module-level constants before environment creation
    # These constants are used when creating the Box2D bodies in reset()
    leg_h = float(leg_length)
    leg_w = float(leg_width)
    
    bw_module.LEG_H = leg_h
    bw_module.LEG_W = leg_w
    bw_module.FRICTION = float(friction)
    
    # LEG_DOWN controls the vertical offset from hull to hip joint
    # Scale proportionally with leg length to maintain proper positioning
    leg_scale = leg_length / _ORIGINAL_LEG_H
    bw_module.LEG_DOWN = _ORIGINAL_LEG_DOWN * leg_scale
    
    # CRITICAL: Update the fixture definitions' shapes!
    # LEG_FD and LOWER_FD are created at module import time with original dimensions.
    # We must update their shape attribute to use the new leg dimensions.
    # The shape is a box defined by half-width and half-height.
    bw_module.LEG_FD.shape = polygonShape(box=(leg_w / 2, leg_h / 2))
    bw_module.LOWER_FD.shape = polygonShape(box=(0.8 * leg_w / 2, leg_h / 2))
    
    # Create environment
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    
    # Modify gravity (world is created in __init__)
    if hasattr(env.unwrapped, "world"):
        env.unwrapped.world.gravity = (0, float(gravity))
    
    # Force a reset to create bodies with new parameters
    env.reset()
    
    return env


def parse_params_from_filename(filename: str) -> dict:
    """
    Parse environment parameters from a checkpoint filename.
    
    Expected format: walker_L{leg_length}_W{leg_width}_g{gravity}_mu{friction}.pth
    
    Args:
        filename: The checkpoint filename (not full path)
    
    Returns:
        Dictionary with keys: leg_length, leg_width, gravity, friction
        
    Raises:
        ValueError: If filename doesn't match expected format
    """
    if not filename.startswith("walker_") or not filename.endswith(".pth"):
        raise ValueError(f"Filename '{filename}' doesn't match expected pattern")
    
    # Remove prefix and suffix: walker_L1.23_W0.34_g-12.34_mu2.34.pth -> L1.23_W0.34_g-12.34_mu2.34
    params_str = filename[7:-4]
    parts = params_str.split("_")
    
    if len(parts) != 4:
        raise ValueError(f"Expected 4 parameter parts, got {len(parts)}")
    
    return {
        "leg_length": float(parts[0][1:]),   # L{value}
        "leg_width": float(parts[1][1:]),    # W{value}
        "gravity": float(parts[2][1:]),      # g{value}
        "friction": float(parts[3][2:]),     # mu{value}
    }


def make_filename(leg_length: float, leg_width: float, gravity: float, friction: float) -> str:
    """Generate a standardized filename for a checkpoint."""
    return f"walker_L{leg_length:.2f}_W{leg_width:.2f}_g{gravity:.2f}_mu{friction:.2f}.pth"
