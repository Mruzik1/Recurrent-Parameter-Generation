"""
Comprehensive analysis of RPG-generated MLP checkpoints.

Covers:
  §6.1: Basic Generation Quality — MSE comparison vs trained baselines
  §6.3: Conditioning Fidelity — parameter sweeps and interpolation
  §6.4: Generalization — held-out conditions and extrapolation
  §6.5: Ensemble & Best-of-N — multi-sample generation

Usage:
    python analyze.py --rpg_checkpoint ../../checkpoint/mlp_regression_adapter.pth \
                      --train_dir ./dataset_train \
                      --val_dir ./dataset_val \
                      --output_dir ./results
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from experiments.mlp_regression.model import FunctionMLP, target_function
from experiments.mlp_regression.test import test_mlp, parse_condition_from_filename


# ---------------------------------------------------------------------------
# RPG model loading
# ---------------------------------------------------------------------------

def load_rpg_model(checkpoint_path: str):
    """Load the trained RPG model."""
    from model import MambaDiffusion as Model
    from model.diffusion import DDPMSampler
    from dataset import MLP_Regression as Dataset

    # Load dataset for structure/postprocessing
    dataset = Dataset(dim_per_token=2048)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract architecture info
    num_permutation = state_dict["to_permutation_state.weight"].shape[0]
    sequence_length = state_dict["model.pe"].shape[1]
    positional_embedding = state_dict["model.pe"][0]

    # Model config (must match training)
    config = {
        "num_permutation": num_permutation,
        "d_condition": 4,
        "d_model": 2048,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        "diffusion_batch": 512,
        "layer_channels": [1, 16, 32, 64, 32, 16, 1],
        "model_dim": 2048,
        "condition_dim": 2048,
        "kernel_size": 7,
        "beta": (0.0001, 0.02),
        "T": 500,
        "sample_mode": DDPMSampler,
        "forward_once": True,
    }

    Model.config = config
    model = Model(sequence_length=sequence_length, positional_embedding=positional_embedding)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, dataset, device


def generate_mlp(model, dataset, device, a, b, f, phi, save_dir=None):
    """Generate an MLP for a given condition and optionally save it."""
    condition = torch.tensor([[a, b, f, phi]], dtype=torch.float32, device=device)

    with torch.no_grad():
        params = model(sample=True, condition=condition)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"gen_a{a:.2f}_b{b:.2f}_f{f:.2f}_p{phi:.2f}.pth")
        dataset.save_params(params, save_path=save_path)
        return save_path
    else:
        # Reconstruct state dict in memory
        state_dict = dataset.postprocess(params.cpu().to(torch.float32))
        return state_dict


def eval_state_dict(state_dict, a, b, f, phi, hidden=64, n_eval=1000):
    """Evaluate a state dict against ground truth."""
    mlp = FunctionMLP(hidden=hidden)
    mlp.load_state_dict(state_dict)
    mlp.eval()

    x_eval = torch.linspace(-math.pi, math.pi, n_eval).unsqueeze(1)
    y_true = target_function(x_eval, a, b, f, phi)

    with torch.no_grad():
        y_pred = mlp(x_eval)
        mse = torch.mean((y_pred - y_true) ** 2).item()

    return mse, y_pred.squeeze().numpy(), y_true.squeeze().numpy(), x_eval.squeeze().numpy()


# ---------------------------------------------------------------------------
# §6.1: Basic Generation Quality
# ---------------------------------------------------------------------------

def analyze_basic_quality(model, dataset, device, train_dir, val_dir, output_dir, n_conditions=100):
    """Compare RPG-generated MLPs vs individually-trained MLPs.
    
    Uses train_dir checkpoints to measure quality on conditions RPG saw,
    and val_dir checkpoints for held-out conditions.
    """
    print("\n" + "="*60)
    print("§6.1: BASIC GENERATION QUALITY")
    print("="*60)

    # Use BOTH train and val checkpoints for comprehensive quality measurement
    # but label which are in-distribution vs held-out
    all_results_by_split = {}
    for split_name, dataset_dir in [("train (in-dist)", train_dir), ("val (held-out)", val_dir)]:
        checkpoint_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pth")]
        np.random.seed(42)
        selected = np.random.choice(checkpoint_files, size=min(n_conditions, len(checkpoint_files)), replace=False)

        trained_mses = []
        generated_mses = []

        for fname in selected:
            params = parse_condition_from_filename(fname)
            a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]
            trained_mse = test_mlp(os.path.join(dataset_dir, fname), a, b, f, phi)
            trained_mses.append(trained_mse)
            state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
            gen_mse, _, _, _ = eval_state_dict(state_dict, a, b, f, phi)
            generated_mses.append(gen_mse)

        trained_mses = np.array(trained_mses)
        generated_mses = np.array(generated_mses)
        all_results_by_split[split_name] = {
            "trained_mses": trained_mses,
            "generated_mses": generated_mses,
            "selected": selected,
            "dataset_dir": dataset_dir,
        }
        print(f"\n  [{split_name}] n={len(selected)}")
        print(f"    Trained MLP mean MSE:    {trained_mses.mean():.6f} ± {trained_mses.std():.6f}")
        print(f"    RPG-generated mean MSE:  {generated_mses.mean():.6f} ± {generated_mses.std():.6f}")
        print(f"    Ratio: {generated_mses.mean()/max(trained_mses.mean(), 1e-10):.2f}x")

    # For backward-compatible plots, use val set as the primary evaluation
    dataset_dir = val_dir
    checkpoint_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pth")]
    np.random.seed(42)
    selected = np.random.choice(checkpoint_files, size=min(n_conditions, len(checkpoint_files)), replace=False)
    trained_mses = all_results_by_split["val (held-out)"]["trained_mses"]
    generated_mses = all_results_by_split["val (held-out)"]["generated_mses"]

    trained_mses = []
    generated_mses = []
    conditions_used = []

    for fname in selected:
        params = parse_condition_from_filename(fname)
        a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]

        # Trained MSE
        trained_mse = test_mlp(os.path.join(dataset_dir, fname), a, b, f, phi)
        trained_mses.append(trained_mse)

        # RPG-generated MSE
        state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
        gen_mse, _, _, _ = eval_state_dict(state_dict, a, b, f, phi)
        generated_mses.append(gen_mse)
        conditions_used.append(params)

    trained_mses = np.array(trained_mses)
    generated_mses = np.array(generated_mses)

    print(f"\nTrained MLP mean MSE:    {trained_mses.mean():.6f} ± {trained_mses.std():.6f}")
    print(f"RPG-generated mean MSE:  {generated_mses.mean():.6f} ± {generated_mses.std():.6f}")
    print(f"Ratio (generated/trained): {generated_mses.mean()/max(trained_mses.mean(), 1e-10):.2f}x")

    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(trained_mses, generated_mses, alpha=0.6, s=20)
    max_val = max(trained_mses.max(), generated_mses.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x (perfect)')
    ax.set_xlabel("Trained MLP MSE")
    ax.set_ylabel("RPG-Generated MLP MSE")
    ax.set_title("Generation Quality: Trained vs RPG-Generated")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "basic_quality_scatter.png"), dpi=150)
    plt.close()

    # Function overlay plots for 10 random conditions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for idx, ax in enumerate(axes.flat):
        if idx >= min(10, len(selected)):
            ax.axis("off")
            continue
        fname = selected[idx]
        params = parse_condition_from_filename(fname)
        a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]

        # Ground truth
        x_eval = torch.linspace(-math.pi, math.pi, 500).unsqueeze(1)
        y_true = target_function(x_eval, a, b, f, phi).squeeze().numpy()
        x_np = x_eval.squeeze().numpy()

        # Trained
        mlp_trained = FunctionMLP(hidden=64)
        mlp_trained.load_state_dict(torch.load(os.path.join(dataset_dir, fname), map_location="cpu", weights_only=True))
        mlp_trained.eval()
        with torch.no_grad():
            y_trained = mlp_trained(x_eval).squeeze().numpy()

        # RPG-generated
        state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
        mlp_gen = FunctionMLP(hidden=64)
        mlp_gen.load_state_dict(state_dict)
        mlp_gen.eval()
        with torch.no_grad():
            y_gen = mlp_gen(x_eval).squeeze().numpy()

        ax.plot(x_np, y_true, 'k-', linewidth=2, label='True')
        ax.plot(x_np, y_trained, 'b--', linewidth=1, label='Trained')
        ax.plot(x_np, y_gen, 'r:', linewidth=1, label='RPG')
        ax.set_title(f"a={a:.1f} b={b:.1f} f={f:.1f}", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Function Approximation: True vs Trained vs RPG-Generated", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "basic_quality_overlays.png"), dpi=150)
    plt.close()

    # Save results
    results = {}
    for split_name, data in all_results_by_split.items():
        key = split_name.split(" ")[0]  # "train" or "val"
        results[key] = {
            "trained_mse_mean": float(data["trained_mses"].mean()),
            "trained_mse_std": float(data["trained_mses"].std()),
            "generated_mse_mean": float(data["generated_mses"].mean()),
            "generated_mse_std": float(data["generated_mses"].std()),
            "ratio": float(data["generated_mses"].mean() / max(data["trained_mses"].mean(), 1e-10)),
            "n_conditions": len(data["selected"]),
        }
    with open(os.path.join(output_dir, "basic_quality.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    return results


# ---------------------------------------------------------------------------
# §6.3: Conditioning Fidelity
# ---------------------------------------------------------------------------

def analyze_conditioning(model, dataset, device, output_dir, n_sweep=20):
    """Test whether conditions control the output via parameter sweeps."""
    print("\n" + "="*60)
    print("§6.3: CONDITIONING FIDELITY")
    print("="*60)

    # Default condition: a=1.0, b=0.0, f=1.0, phi=0.0
    defaults = {"a": 1.0, "b": 0.0, "f": 1.0, "phi": 0.0}

    sweep_ranges = {
        "a": np.linspace(0.2, 2.0, n_sweep),
        "b": np.linspace(-1.0, 1.0, n_sweep),
        "f": np.linspace(0.5, 3.0, n_sweep),
        "phi": np.linspace(0.0, 2 * math.pi, n_sweep),
    }

    fig_sweep, axes_sweep = plt.subplots(2, 2, figsize=(14, 10))
    fig_func, axes_func = plt.subplots(2, 2, figsize=(14, 10))
    x_eval = torch.linspace(-math.pi, math.pi, 500).unsqueeze(1)
    x_np = x_eval.squeeze().numpy()

    sweep_results = {}

    for ax_idx, (param_name, sweep_vals) in enumerate(sweep_ranges.items()):
        row, col = divmod(ax_idx, 2)
        ax_s = axes_sweep[row, col]
        ax_f = axes_func[row, col]

        mses = []
        for val in sweep_vals:
            cond = dict(defaults)
            cond[param_name] = float(val)
            a, b, f, phi = cond["a"], cond["b"], cond["f"], cond["phi"]

            state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
            mse, y_pred, y_true, _ = eval_state_dict(state_dict, a, b, f, phi)
            mses.append(mse)

            # Plot function for this sweep value
            color = plt.cm.viridis(float((val - sweep_vals.min()) / (sweep_vals.max() - sweep_vals.min() + 1e-10)))
            ax_f.plot(x_np, y_pred, color=color, alpha=0.6, linewidth=1)

        mses = np.array(mses)
        ax_s.plot(sweep_vals, mses, 'bo-', markersize=3)
        ax_s.set_xlabel(param_name)
        ax_s.set_ylabel("MSE")
        ax_s.set_title(f"MSE vs {param_name}")

        ax_f.set_xlabel("x")
        ax_f.set_ylabel("y")
        ax_f.set_title(f"Generated functions sweeping {param_name}")

        sweep_results[param_name] = {
            "values": sweep_vals.tolist(),
            "mses": mses.tolist(),
            "mean_mse": float(mses.mean()),
        }

        print(f"  Sweep {param_name}: mean MSE = {mses.mean():.6f}, max MSE = {mses.max():.6f}")

    fig_sweep.suptitle("Conditioning Fidelity: MSE vs Swept Parameter", fontsize=14)
    fig_sweep.tight_layout()
    fig_sweep.savefig(os.path.join(output_dir, "conditioning_sweep_mse.png"), dpi=150)
    plt.close(fig_sweep)

    fig_func.suptitle("Conditioning Fidelity: Generated Functions", fontsize=14)
    fig_func.tight_layout()
    fig_func.savefig(os.path.join(output_dir, "conditioning_sweep_functions.png"), dpi=150)
    plt.close(fig_func)

    # Condition interpolation
    print("\n  Condition interpolation test...")
    c1 = {"a": 0.5, "b": -0.5, "f": 1.0, "phi": 0.0}
    c2 = {"a": 2.0, "b": 0.5, "f": 2.5, "phi": math.pi}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_interp = 11
    for i, t in enumerate(np.linspace(0, 1, n_interp)):
        cond = {k: (1 - t) * c1[k] + t * c2[k] for k in c1}
        a, b, f, phi = cond["a"], cond["b"], cond["f"], cond["phi"]

        state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
        mlp = FunctionMLP(hidden=64)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        with torch.no_grad():
            y_pred = mlp(x_eval).squeeze().numpy()

        color = plt.cm.coolwarm(t)
        ax.plot(x_np, y_pred, color=color, alpha=0.8, linewidth=1.5, label=f't={t:.1f}')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Condition Interpolation: c1 -> c2")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conditioning_interpolation.png"), dpi=150)
    plt.close()

    with open(os.path.join(output_dir, "conditioning.json"), "w") as fp:
        json.dump(sweep_results, fp, indent=2)

    return sweep_results


# ---------------------------------------------------------------------------
# §6.4: Generalization to Unseen Conditions
# ---------------------------------------------------------------------------

def analyze_generalization(model, dataset, device, train_dir, val_dir, output_dir):
    """Test generalization to held-out and out-of-distribution conditions.
    
    Uses the actual val split (conditions RPG never saw) instead of random
    holdout from the same directory.
    """
    print("\n" + "="*60)
    print("§6.4: GENERALIZATION")
    print("="*60)

    # --- True held-out conditions (val split, never seen during RPG training) ---
    holdout_files = sorted([f for f in os.listdir(val_dir) if f.endswith(".pth")])
    dataset_dir = val_dir

    holdout_trained_mses = []
    holdout_generated_mses = []

    print(f"\n  Held-out val conditions ({len(holdout_files)} conditions, never seen by RPG)...")
    for fname in holdout_files:
        params = parse_condition_from_filename(fname)
        a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]

        trained_mse = test_mlp(os.path.join(dataset_dir, fname), a, b, f, phi)
        holdout_trained_mses.append(trained_mse)

        state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
        gen_mse, _, _, _ = eval_state_dict(state_dict, a, b, f, phi)
        holdout_generated_mses.append(gen_mse)

    holdout_trained_mses = np.array(holdout_trained_mses)
    holdout_generated_mses = np.array(holdout_generated_mses)

    print(f"  Holdout trained MSE:    {holdout_trained_mses.mean():.6f} ± {holdout_trained_mses.std():.6f}")
    print(f"  Holdout generated MSE:  {holdout_generated_mses.mean():.6f} ± {holdout_generated_mses.std():.6f}")

    # --- Extrapolation: outside training range ---
    print(f"\n  Extrapolation test (out-of-range conditions)...")
    extrap_conditions = [
        {"a": 2.5, "b": 0.0, "f": 1.0, "phi": 0.0, "label": "a=2.5 (max 2.0)"},
        {"a": 1.0, "b": 0.0, "f": 4.0, "phi": 0.0, "label": "f=4.0 (max 3.0)"},
        {"a": 1.0, "b": 1.5, "f": 1.0, "phi": 0.0, "label": "b=1.5 (max 1.0)"},
        {"a": 2.0, "b": 1.0, "f": 3.0, "phi": 2*math.pi, "label": "all extreme"},
        {"a": 0.1, "b": 0.0, "f": 0.3, "phi": 0.0, "label": "a=0.1,f=0.3 (below range)"},
    ]

    extrap_mses = []
    for ec in extrap_conditions:
        a, b, f, phi = ec["a"], ec["b"], ec["f"], ec["phi"]
        state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
        mse, _, _, _ = eval_state_dict(state_dict, a, b, f, phi)
        extrap_mses.append(mse)
        print(f"    {ec['label']}: MSE = {mse:.6f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Holdout scatter
    axes[0].scatter(holdout_trained_mses, holdout_generated_mses, alpha=0.6, s=20)
    max_val = max(holdout_trained_mses.max(), holdout_generated_mses.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--')
    axes[0].set_xlabel("Trained MSE")
    axes[0].set_ylabel("Generated MSE")
    axes[0].set_title("Holdout Generalization")

    # Extrapolation bar chart
    labels = [ec["label"] for ec in extrap_conditions]
    axes[1].barh(labels, extrap_mses, color='steelblue')
    axes[1].set_xlabel("MSE")
    axes[1].set_title("Extrapolation (Out-of-Range)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generalization.png"), dpi=150)
    plt.close()

    results = {
        "holdout": {
            "trained_mse_mean": float(holdout_trained_mses.mean()),
            "generated_mse_mean": float(holdout_generated_mses.mean()),
            "ratio": float(holdout_generated_mses.mean() / max(holdout_trained_mses.mean(), 1e-10)),
        },
        "extrapolation": {
            ec["label"]: float(mse) for ec, mse in zip(extrap_conditions, extrap_mses)
        },
    }
    with open(os.path.join(output_dir, "generalization.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    return results


# ---------------------------------------------------------------------------
# §6.5: Ensemble & Best-of-N
# ---------------------------------------------------------------------------

def analyze_ensemble(model, dataset, device, val_dir, output_dir, n_models=20, n_conditions=10):
    """Test ensemble and best-of-N generation strategies on held-out conditions."""
    print("\n" + "="*60)
    print("§6.5: ENSEMBLE & BEST-OF-N")
    print("="*60)

    dataset_dir = val_dir
    checkpoint_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pth")]
    np.random.seed(77)
    selected = np.random.choice(checkpoint_files, size=min(n_conditions, len(checkpoint_files)), replace=False)

    single_mses = []
    ensemble_mses = []
    best_of_n_mses = []
    trained_mses = []

    x_eval = torch.linspace(-math.pi, math.pi, 1000).unsqueeze(1)

    for fname in selected:
        params = parse_condition_from_filename(fname)
        a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]

        y_true = target_function(x_eval, a, b, f, phi).squeeze().numpy()

        # Trained baseline
        trained_mse = test_mlp(os.path.join(dataset_dir, fname), a, b, f, phi)
        trained_mses.append(trained_mse)

        # Generate N models
        preds = []
        indiv_mses = []
        for _ in range(n_models):
            state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
            mlp = FunctionMLP(hidden=64)
            mlp.load_state_dict(state_dict)
            mlp.eval()
            with torch.no_grad():
                y_pred = mlp(x_eval).squeeze().numpy()
            preds.append(y_pred)
            indiv_mses.append(np.mean((y_pred - y_true) ** 2))

        preds = np.array(preds)
        indiv_mses = np.array(indiv_mses)

        # Single model (mean of individual MSEs)
        single_mses.append(indiv_mses.mean())

        # Ensemble: average predictions
        ensemble_pred = preds.mean(axis=0)
        ensemble_mse = np.mean((ensemble_pred - y_true) ** 2)
        ensemble_mses.append(ensemble_mse)

        # Best-of-N
        best_idx = indiv_mses.argmin()
        best_of_n_mses.append(indiv_mses[best_idx])

    trained_mses = np.array(trained_mses)
    single_mses = np.array(single_mses)
    ensemble_mses = np.array(ensemble_mses)
    best_of_n_mses = np.array(best_of_n_mses)

    print(f"\n  Trained baseline MSE:  {trained_mses.mean():.6f} ± {trained_mses.std():.6f}")
    print(f"  Single RPG MSE:        {single_mses.mean():.6f} ± {single_mses.std():.6f}")
    print(f"  Ensemble ({n_models}) MSE:    {ensemble_mses.mean():.6f} ± {ensemble_mses.std():.6f}")
    print(f"  Best-of-{n_models} MSE:       {best_of_n_mses.mean():.6f} ± {best_of_n_mses.std():.6f}")

    # Bar chart comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    methods = ["Trained", f"Single RPG", f"Ensemble-{n_models}", f"Best-of-{n_models}"]
    means = [trained_mses.mean(), single_mses.mean(), ensemble_mses.mean(), best_of_n_mses.mean()]
    stds = [trained_mses.std(), single_mses.std(), ensemble_mses.std(), best_of_n_mses.std()]
    colors = ['green', 'steelblue', 'orange', 'red']

    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel("MSE")
    ax.set_title("Ensemble & Best-of-N vs Baselines")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{mean:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_comparison.png"), dpi=150)
    plt.close()

    # Detailed overlay for one condition
    fname = selected[0]
    params = parse_condition_from_filename(fname)
    a, b, f, phi = params["a"], params["b"], params["f"], params["phi"]
    y_true = target_function(x_eval, a, b, f, phi).squeeze().numpy()
    x_np = x_eval.squeeze().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Generate N models and plot all
    all_preds = []
    for _ in range(n_models):
        state_dict = generate_mlp(model, dataset, device, a, b, f, phi)
        mlp = FunctionMLP(hidden=64)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        with torch.no_grad():
            y_pred = mlp(x_eval).squeeze().numpy()
        all_preds.append(y_pred)
        ax.plot(x_np, y_pred, 'b-', alpha=0.15, linewidth=0.5)

    ensemble_pred = np.mean(all_preds, axis=0)
    ax.plot(x_np, y_true, 'k-', linewidth=2, label='Ground Truth')
    ax.plot(x_np, ensemble_pred, 'r-', linewidth=2, label=f'Ensemble ({n_models})')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Ensemble Detail: a={a:.2f}, b={b:.2f}, f={f:.2f}, phi={phi:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_detail.png"), dpi=150)
    plt.close()

    results = {
        "n_models": n_models,
        "n_conditions": len(selected),
        "trained_mse": float(trained_mses.mean()),
        "single_mse": float(single_mses.mean()),
        "ensemble_mse": float(ensemble_mses.mean()),
        "best_of_n_mse": float(best_of_n_mses.mean()),
    }
    with open(os.path.join(output_dir, "ensemble.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze RPG-generated MLP checkpoints")
    parser.add_argument("--rpg_checkpoint", type=str, required=True,
                        help="Path to trained RPG model checkpoint")
    parser.add_argument("--train_dir", type=str,
                        default="./dataset_train",
                        help="Directory with training MLP checkpoints (seen by RPG)")
    parser.add_argument("--val_dir", type=str,
                        default="./dataset_val",
                        help="Directory with held-out MLP checkpoints (unseen by RPG)")
    parser.add_argument("--output_dir", type=str,
                        default="./results",
                        help="Output directory for plots and metrics")
    parser.add_argument("--sections", type=str, nargs="+",
                        default=["quality", "conditioning", "generalization", "ensemble"],
                        help="Analysis sections to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading RPG model from: {args.rpg_checkpoint}")
    model, rpg_dataset, device = load_rpg_model(args.rpg_checkpoint)
    print(f"Model loaded successfully. Device: {device}")
    print(f"Train dir (in-dist): {args.train_dir} ({len(os.listdir(args.train_dir))} files)")
    print(f"Val dir (held-out):  {args.val_dir} ({len(os.listdir(args.val_dir))} files)")

    all_results = {}

    if "quality" in args.sections:
        all_results["quality"] = analyze_basic_quality(
            model, rpg_dataset, device, args.train_dir, args.val_dir, args.output_dir
        )

    if "conditioning" in args.sections:
        all_results["conditioning"] = analyze_conditioning(
            model, rpg_dataset, device, args.output_dir
        )

    if "generalization" in args.sections:
        all_results["generalization"] = analyze_generalization(
            model, rpg_dataset, device, args.train_dir, args.val_dir, args.output_dir
        )

    if "ensemble" in args.sections:
        all_results["ensemble"] = analyze_ensemble(
            model, rpg_dataset, device, args.val_dir, args.output_dir
        )

    # Save combined results
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as fp:
        json.dump(all_results, fp, indent=2)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
