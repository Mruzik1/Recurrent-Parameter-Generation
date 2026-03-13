# MLP Experiment Plan: RPG for Simple Fully-Connected Networks

## 1. Goal

Demonstrate RPG's practical capabilities on a **simple, fully interpretable** architecture — a small MLP — where every aspect of the weight space can be thoroughly analyzed. Unlike large vision models where understanding what RPG learned is difficult, an MLP is small enough to:

- Visualize complete weight distributions layer-by-layer
- Measure exact deviation from training data distribution
- Test conditioning on meaningful, continuous task parameters
- Verify generalization to unseen conditions
- Compute ensembles and best-of-N with trivial cost
- Compare head-to-head against individually-trained baselines

---

## 2. Task: Regression on Synthetic Functions

**Why regression, not classification?**
- Regression loss (MSE) gives continuous, fine-grained quality signal — not just "93% vs 94% accuracy"
- Synthetic functions allow **exact ground truth** — we know the optimal weights analytically for simple cases
- We can **control task difficulty** via function parameters → natural conditioning variable
- Training each MLP takes seconds → can generate hundreds of checkpoints quickly
- Results are immediately interpretable: plot predicted vs true function

### 2.1 Task Family: Parameterized 1D Functions

Train MLPs to approximate a family of 1D functions parameterized by a condition vector **c = (a, b, f, φ)**:

$$y = a \cdot \sin(f \cdot x + \varphi) + b \cdot x$$

Where:
| Parameter | Meaning | Range | Effect |
|-----------|---------|-------|--------|
| `a` | Sine amplitude | [0.2, 2.0] | Controls nonlinearity strength |
| `b` | Linear slope | [-1.0, 1.0] | Adds linear trend |
| `f` | Frequency | [0.5, 3.0] | Controls oscillation speed |
| `φ` | Phase | [0, 2π] | Shifts the sine wave |

**Input**: x ∈ [-π, π], sampled uniformly (200 points per training run)
**Output**: y = a·sin(f·x + φ) + b·x

This is ideal because:
- 4D condition space (same as BipedalWalker — the infrastructure exists)
- Smooth function family → small perturbations in c → small changes in optimal weights
- Easy to visualize: plot the learned function vs ground truth
- Can test interpolation (generate for unseen c) and extrapolation (c outside training range)

### 2.2 Alternative: Multi-class Spiral Classification

If a classification task is preferred, use the **parameterized spiral dataset**:

$$r = \theta \cdot s, \quad \text{where } s \in [0.3, 1.5] \text{ controls spiral tightness}$$

- 2-5 spirals (class count as condition)
- Spiral tightness `s` and noise level `σ` as continuous conditions
- Condition vector: **c = (num_classes, tightness, noise)** — 3D
- Simple enough that an MLP can solve it; hard enough that quality varies

**Recommendation: Start with the regression task (2.1).** It's simpler, faster, and the quality metric (MSE) is more informative than classification accuracy for analyzing RPG.

---

## 3. MLP Architecture

```python
class FunctionMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),      # 1 → 64:   128 params (64 weights + 64 biases)
            nn.ReLU(),
            nn.Linear(hidden, hidden), # 64 → 64:  4160 params
            nn.ReLU(),
            nn.Linear(hidden, 1),      # 64 → 1:   65 params
        )                              # Total: ~4,353 params

    def forward(self, x):
        return self.net(x)
```

### Parameter Count Analysis

| Hidden Size | Total Params | Tokens (dim=1024) | Tokens (dim=2048) |
|-------------|-------------|-------------------|-------------------|
| 32 | 1,153 | 2 | 1 |
| 64 | 4,353 | 5 | 3 |
| 128 | 16,769 | 17 | 9 |

**Recommended: hidden=64 with dim_per_token=1024** → 5 tokens. This gives:
- Short enough sequence that RPG trains very fast (minutes, not hours)
- Long enough to have meaningful inter-token structure for Mamba to learn
- Small enough d_model (1024) that the RPG model itself is lightweight
- ~4K params is comparable to the BipedalWalker policy network (~10K params)

---

## 4. Dataset Generation

### 4.1 Sampling Strategy

Generate **400 condition vectors** using the same mixed strategy as BipedalWalker:

1. **Grid anchors** (~40 points): corners + axis midpoints of the 4D space
2. **Latin Hypercube Sampling** (~360 points): space-filling coverage

For each condition, train **1 MLP** (deterministic — same seed, same optimizer). This is fast:
- Each MLP trains in ~2-5 seconds on CPU
- Total: 400 × 5s = ~30 minutes for the entire dataset

### 4.2 Training Protocol per MLP

```
Optimizer: Adam, lr=1e-3
Epochs: 500
Data: 200 points sampled uniformly from x ∈ [-π, π]
Loss: MSE
Eval: 1000 points uniformly spaced in [-π, π]
```

### 4.3 Save Format

Each checkpoint: `func_a{a:.2f}_b{b:.2f}_f{f:.2f}_p{phi:.2f}.pth`

Example: `func_a1.50_b0.30_f2.00_p1.57.pth`

Save: `model.state_dict()` — just the raw weight dict, matching RPG's expected format.

### 4.4 Quality Gate

Discard any MLP with final MSE > 0.01 on the eval set (sanity check — most should converge well since the task is simple for a 2-layer MLP).

---

## 5. RPG Training Configuration

### 5.1 Config (for `workspace/mlp_regression/train.py`)

```python
config = {
    "dataset": MLP_Regression,
    "dim_per_token": 2048,
    "batch_size": 16,
    "total_steps": 100_000,
    "learning_rate": 5e-4,
    "save_every": 10_000,
    "model_config": {
        "d_condition": 4,           # (a, b, f, φ)
        "d_model": 2048,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        "diffusion_batch": 512,
        "layer_channels": [1, 16, 32, 64, 32, 16, 1],
        "T": 500,
        "sample_mode": DDPMSampler,
    },
    "tag": "mlp_regression",
}
```

### 5.2 RPG Model: `ClassConditionMambaDiffusionFull`

Use the conditional model (same as BipedalWalker) since we have a 4D continuous condition.

### 5.3 Estimated Training Cost

- Sequence length: ~5 tokens
- d_model: 2048
- Each forward pass processes 5 × 2048 = 10,240 values (vs ~11M for ResNet-18)
- **Expected: ~10-30 minutes on a single GPU for full training**

---

## 6. Comprehensive Analysis Plan

### 6.1 Basic Generation Quality

**What to measure:**
- Generate 100 MLPs for randomly sampled conditions
- For each: compute MSE on the target function (ground truth is known exactly)
- Compare to the MSE of individually-trained MLPs (the training data)

**Expected output:**
```
Trained MLP mean MSE:    0.0012 ± 0.0008
RPG-generated mean MSE:  0.0045 ± 0.0025  (4x worse but still very good)
```

**Visualization:**
- Scatter plot: trained MSE vs RPG-generated MSE (one dot per condition)
- For 10 randomly picked conditions: overlay plots of true function, trained MLP prediction, RPG-generated MLP prediction

### 6.2 Weight Distribution Analysis

**Per-layer comparison (RPG-generated vs training data):**
- Histogram of weight values: layer 1 weights, layer 1 biases, layer 2 weights, etc.
- Per-layer statistics: mean, std, min, max, kurtosis
- QQ-plot: generated weight distribution vs training data distribution

**Inter-parameter correlations:**
- Flatten all weights, compute cosine similarity matrix across N generated models
- Compare to cosine similarity matrix of the training checkpoints
- Expected: RPG-generated models should show similar diversity as training data

**Weight-space PCA** (possible here because the models are tiny!):
- Flatten each checkpoint (trained + generated) to a single vector
- PCA on the full set
- Plot trained vs generated in the first 2-3 PC dimensions
- Shows whether RPG samples from the same manifold or a different one

### 6.3 Conditioning Fidelity

**Does the condition actually control the output?**

For each condition parameter (a, b, f, φ), fix the other three and sweep one:
- Generate MLPs at 20 evenly spaced values of the swept parameter
- Plot MSE vs swept parameter value
- Plot the learned function at each sweep point → should show smooth variation

**Example sweep — varying amplitude `a`:**
- Fix b=0, f=1, φ=0
- Generate for a = 0.2, 0.4, 0.6, ..., 2.0
- Expected: each generated MLP should approximate y = a·sin(x) with increasing amplitude

**Condition interpolation:**
- Pick two training conditions c₁ and c₂
- Generate MLPs for c_t = (1-t)·c₁ + t·c₂ at t = 0, 0.1, ..., 1.0
- Plot the generated functions → should smoothly morph between f(c₁) and f(c₂)
- This directly tests whether RPG learned a smooth latent space

### 6.4 Generalization to Unseen Conditions

**In-distribution holdout:**
- Reserve 10% of conditions (40 out of 400) as test set — never seen during RPG training
- Generate MLPs for these held-out conditions
- Measure MSE compared to individually-trained MLPs for the same conditions

**Edge/extrapolation:**
- Generate for conditions slightly **outside** the training range:
  - a = 2.5 (training max: 2.0)
  - f = 4.0 (training max: 3.0)
  - b = 1.5 (training max: 1.0)
- Measure degradation vs in-distribution performance
- This directly tests RPG's limits

**Novel combinations:**
- If training used LHS (no condition appears in all 4 dimensions simultaneously at extreme), test with all-extreme conditions like a=2.0, b=1.0, f=3.0, φ=2π

### 6.5 Ensemble & Best-of-N

**Ensemble of generated MLPs:**
- For a single condition, generate N = 20 MLPs
- Ensemble prediction: average their outputs (function values at each x)
- Compare ensemble MSE vs single model MSE
- Plot: single model spread + ensemble prediction + ground truth

**Best-of-N selection:**
- For each condition, generate N = 20, evaluate each on 50 quick test points
- Pick the best one → compare to mean RPG model and to trained baseline

**Cost comparison:**
- Time to generate 20 MLPs: ~seconds
- Time to train 20 MLPs: ~100 seconds
- Speedup ratio (not as dramatic as for large models, but demonstrates the pattern)

### 6.6 Diversity Analysis

**Functional diversity:**
- For a fixed condition, generate 20 MLPs
- Plot all 20 functions overlaid → shows functional variance of RPG
- Compute pairwise MSE between generated models' outputs (not weights)
- Compare to pairwise MSE of 20 independently-trained MLPs with different seeds

**Weight diversity:**
- Cosine similarity between generated weight vectors
- L2 distance in weight space
- Expected: weight vectors are diverse (low cosine sim) but functions are similar (low output MSE)

### 6.7 Ablation: How Many Training Checkpoints Do You Need?

Train RPG with subsets of the dataset:
- 50, 100, 200, 400 checkpoints
- For each: measure generation quality (MSE) and conditioning fidelity (sweep smoothness)
- Plot quality vs dataset size → answers: "is 400 enough? would 1000 help?"

### 6.8 Comparison to Baselines

**Baseline 1: Directly train an MLP for the target condition** (the "just train it" approach)
- Time: ~5 seconds
- Quality: optimal (this is the ceiling)

**Baseline 2: Nearest-neighbor in condition space**
- For a target condition c*, find the nearest c in the training set
- Use that checkpoint directly (no RPG needed)
- Quality: depends on density of training conditions

**Baseline 3: Linear interpolation in weight space**
- Find K nearest conditions in training set
- Weighted average of their weights (inverse distance weighting)
- This is the simplest "model soup" approach
- How does it compare to RPG for interpolated conditions?

**Baseline 4: Random weights** (sanity check — must be much worse)

---

## 7. File Structure

```
experiments/mlp_regression/
├── EXPERIMENT_PLAN.md          # This file
├── model.py                    # FunctionMLP definition
├── train_single.py             # Train one MLP for a specific condition
├── generate_dataset.py         # Generate all 400 training checkpoints
├── test.py                     # Evaluate a generated MLP checkpoint
├── analyze.py                  # Comprehensive analysis (§6 above)
├── dataset/                    # Generated checkpoints (populated by generate_dataset.py)
│   ├── func_a0.50_b0.00_f1.00_p0.00.pth
│   ├── ...
│   └── manifest.json           # Metadata: condition → MSE → filename mapping
└── results/                    # Output plots and metrics (populated by analyze.py)

dataset/mlp_regression/
├── checkpoint/                 # Symlink or copy of experiments/mlp_regression/dataset/
├── generated/                  # RPG-generated checkpoints go here
├── model.py                    # Copy of MLP model (for RPG's import)
├── train.py                    # Train/test boilerplate (matches RPG conventions)
├── test.py                     # Test script matching RPG test convention
└── structure.cache             # Auto-generated by RPG's BaseDataset

workspace/mlp_regression/
└── train.py                    # RPG training config (§5 above)
```

---

## 8. Implementation Order

### Phase 1: Data Generation (~1 hour)
1. Implement `experiments/mlp_regression/model.py` — the FunctionMLP class
2. Implement `experiments/mlp_regression/train_single.py` — train one MLP
3. Implement `experiments/mlp_regression/generate_dataset.py` — batch training, LHS sampling, manifest
4. Run dataset generation → 400 checkpoints
5. Register `MLP_Regression` in `dataset/register.py` as a `ConditionalDataset`
6. Create `dataset/mlp_regression/` with the RPG-compatible train.py/test.py/model.py

### Phase 2: RPG Training (~30 min)
7. Create `workspace/mlp_regression/train.py` — RPG training config
8. Train RPG model: `bash launch.sh workspace/mlp_regression/train.py '0'`
9. Verify basic generation works (generates valid MLPs)

### Phase 3: Analysis (~2-3 hours to implement, minutes to run)
10. Implement `experiments/mlp_regression/analyze.py` covering all §6 sections
11. Run analysis, generate all plots and tables
12. Write up results summary

---

## 9. Key Questions This Experiment Answers

1. **Does RPG work on tiny models?** — Not just large vision models, but 4K-param MLPs
2. **Can RPG learn smooth conditioning?** — Sweep plots will show this directly
3. **Does RPG generalize to unseen conditions?** — Holdout + extrapolation tests
4. **Is RPG's weight space distribution realistic?** — PCA + QQ plots + histograms
5. **Do RPG ensembles help?** — Ensemble vs single vs trained baseline
6. **How does RPG compare to trivial baselines?** — Nearest-neighbor, weight interpolation
7. **What's the minimum dataset size?** — Ablation study

---

## 10. Expected Outcomes and What Would Be Convincing

### Success criteria (RPG shows practical value):
- Generated MLPs achieve **< 5x the MSE** of individually-trained ones (out of distribution)
- Conditioning sweeps show **smooth, monotonic** variation of the learned function
- Generalization to held-out conditions works with **< 10x MSE** degradation
- RPG beats the nearest-neighbor baseline for interpolated conditions
- Weight-space PCA shows generated models **occupy the same manifold** as trained ones

### Failure modes that would indicate RPG isn't adding value:
- Generated MLPs are no better than random → RPG didn't learn the distribution
- Conditioning has no effect → condition information is ignored
- Nearest-neighbor baseline matches or beats RPG → no need for generative model
- Weight PCA shows generated models in a completely separate cluster → mode collapse

### What would be especially impressive:
- Smooth interpolation in condition space → RPG learned a meaningful weight manifold
- Extrapolation works (even slightly beyond training range) → genuine generalization
- Ensemble of RPG models beats the best individually-trained model → practical value
