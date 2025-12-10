"""
Train 4 test agents with distinct characteristics to verify the system.

This script trains:
1. Landing-focused agent (prioritizes safe landing)
2. Fuel-efficient agent (minimizes engine use)
3. Speed-focused agent (fast landing)
4. Smooth agent (gentle control inputs)
"""
import os
import subprocess
import sys
from pathlib import Path

# Define 4 distinct agent profiles
TEST_AGENTS = [
    {
        "name": "Landing-Focused",
        "weights": {"w_landing": 0.7, "w_fuel": 0.1, "w_time": 0.1, "w_smoothness": 0.1},
        "description": "Prioritizes safe landing above all else"
    },
    {
        "name": "Fuel-Efficient",
        "weights": {"w_landing": 0.2, "w_fuel": 0.6, "w_time": 0.1, "w_smoothness": 0.1},
        "description": "Minimizes engine usage (fuel saving)"
    },
    {
        "name": "Speed-Focused",
        "weights": {"w_landing": 0.2, "w_fuel": 0.1, "w_time": 0.6, "w_smoothness": 0.1},
        "description": "Lands as quickly as possible"
    },
    {
        "name": "Smooth-Controller",
        "weights": {"w_landing": 0.2, "w_fuel": 0.1, "w_time": 0.1, "w_smoothness": 0.6},
        "description": "Makes gentle, smooth control inputs"
    },
]


def train_agent(agent_config: dict, output_dir: str, timesteps: int = 300_000) -> tuple:
    """
    Train a single agent with specified configuration.

    Returns:
        (success: bool, checkpoint_path: str)
    """
    name = agent_config["name"]
    weights = agent_config["weights"]

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Description: {agent_config['description']}")
    print(f"Weights: {weights}")
    print('='*60)

    script_path = Path(__file__).parent / "train_single.py"
    cmd = [
        sys.executable, str(script_path),
        "--w_landing", str(weights["w_landing"]),
        "--w_fuel", str(weights["w_fuel"]),
        "--w_time", str(weights["w_time"]),
        "--w_smoothness", str(weights["w_smoothness"]),
        "--output_dir", output_dir,
        "--total_timesteps", str(timesteps),
        "--seed", str(hash(name) % 10000),  # Deterministic but unique seed
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Extract checkpoint path from output
        for line in result.stdout.split('\n'):
            if 'Model saved to' in line:
                checkpoint_path = line.split()[-1]
                print(f"✅ Training complete: {checkpoint_path}")
                return (True, checkpoint_path)

        print(f"✅ Training complete (path not found in output)")
        return (True, None)

    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed:")
        print(e.stderr)
        return (False, None)


def test_agent(checkpoint_path: str, num_episodes: int = 5) -> dict:
    """
    Test a trained agent and return performance metrics.

    Returns:
        dict with keys: mean_reward, std_reward, success
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"success": False, "mean_reward": 0.0, "std_reward": 0.0}

    print(f"  Testing checkpoint: {os.path.basename(checkpoint_path)}")

    script_path = Path(__file__).parent / "test.py"
    cmd = [
        sys.executable, str(script_path),
        checkpoint_path,
        "--num_episodes", str(num_episodes),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)

        # Parse results from output
        for line in result.stdout.split('\n'):
            if "Mean:" in line:
                mean_reward = float(line.split("Mean:")[1].strip())
            if "Std:" in line:
                std_reward = float(line.split("Std:")[1].strip())

        print(f"  Performance: {mean_reward:.1f} ± {std_reward:.1f}")
        return {"success": True, "mean_reward": mean_reward, "std_reward": std_reward}

    except Exception as e:
        print(f"  ❌ Testing failed: {e}")
        return {"success": False, "mean_reward": 0.0, "std_reward": 0.0}


def main():
    output_dir = "experiments/lunar_lander/checkpoint"
    os.makedirs(output_dir, exist_ok=True)

    # Reduced timesteps for quick testing (increase for production)
    timesteps = 300_000

    print("="*60)
    print("LUNAR LANDER: Training 4 Test Agents")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Training timesteps per agent: {timesteps:,}")
    print()

    results = []

    # Train and test each agent
    for agent_config in TEST_AGENTS:
        success, checkpoint_path = train_agent(agent_config, output_dir, timesteps)

        if success and checkpoint_path:
            test_results = test_agent(checkpoint_path, num_episodes=5)
            results.append({
                "name": agent_config["name"],
                "checkpoint": checkpoint_path,
                "performance": test_results
            })
        else:
            print(f"  ⚠️ Skipping test due to training failure")
            results.append({
                "name": agent_config["name"],
                "checkpoint": None,
                "performance": {"success": False}
            })

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for i, result in enumerate(results, 1):
        name = result["name"]
        perf = result["performance"]

        if perf["success"]:
            print(f"{i}. {name:20s} | Reward: {perf['mean_reward']:6.1f} ± {perf['std_reward']:5.1f}")
        else:
            print(f"{i}. {name:20s} | ❌ Failed")

    # Check for successful training
    successful = [r for r in results if r["performance"]["success"]]

    if len(successful) >= 3:
        print(f"\n✅ SUCCESS: {len(successful)}/4 agents trained successfully!")
        print("\nYou can now:")
        print(f"  1. Visualize individual policies:")
        print(f"     python experiments/lunar_lander/visualize.py --checkpoint <path>")
        print(f"  2. Compare policies:")
        checkpoints = [r["checkpoint"] for r in successful[:2]]
        print(f"     python experiments/lunar_lander/visualize.py --compare {' '.join(checkpoints)}")
        print(f"  3. Generate full dataset:")
        print(f"     python experiments/lunar_lander/generate_dataset.py --num_samples 100")
    else:
        print(f"\n⚠️ Only {len(successful)}/4 agents trained successfully")

    print("="*60)


if __name__ == "__main__":
    main()
