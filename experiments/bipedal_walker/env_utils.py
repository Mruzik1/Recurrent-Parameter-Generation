"""
Environment utilities for creating custom BipedalWalker environments.
"""
import gymnasium as gym
import gymnasium.envs.box2d.bipedal_walker as bw_module


# Default BipedalWalker constants (for reference)
DEFAULT_LEG_LENGTH = 1.13
DEFAULT_LEG_WIDTH = 0.26
DEFAULT_GRAVITY = -10.0
DEFAULT_FRICTION = 2.5


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
    bw_module.LEG_H = float(leg_length)
    bw_module.LEG_W = float(leg_width)
    bw_module.FRICTION = float(friction)
    
    # Create environment
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    
    # Modify gravity (world is created in __init__)
    if hasattr(env.unwrapped, "world"):
        env.unwrapped.world.gravity = (0, float(gravity))
    
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
