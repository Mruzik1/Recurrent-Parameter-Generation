"""Quick validation that all components are working."""
import sys, os
sys.path.append(".")
os.chdir(".")

import torch
from model import MambaDiffusion as Model
from model.diffusion import DDPMSampler
from dataset import BipedalWalker_PPO as Dataset

print("==> Loading dataset...")
dataset = Dataset(dim_per_token=1024)
print(f"  ✓ Dataset loaded: {dataset.real_length} checkpoints")
print(f"  ✓ Sequence length: {dataset.sequence_length}")

print("\n==> Testing data loading...")
param, condition = dataset[0]
print(f"  ✓ Param shape: {param.shape}")
print(f"  ✓ Condition: {condition}")

print("\n==> Building model...")
config = {
    "num_permutation": dataset.max_permutation_state,
    "d_condition": 4,
    "d_model": 1024,
    "d_state": 64,
    "d_conv": 4,
    "expand": 2,
    "num_layers": 2,
    "diffusion_batch": 256,
    "layer_channels": [1, 16, 32, 64, 32, 16, 1],
    "model_dim": 1024,
    "condition_dim": 1024,
    "kernel_size": 7,
    "sample_mode": DDPMSampler,
    "beta": (0.0001, 0.02),
    "T": 500,
    "forward_once": True,
}

Model.config = config
model = Model(
    sequence_length=dataset.sequence_length,
    positional_embedding=dataset.get_position_embedding(
        positional_embedding_dim=config["d_model"]
    )
)
print(f"  ✓ Model created")
print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Move to CUDA if available (Mamba requires CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    model = model.to(device)
    param = param.to(device)
    condition = condition.to(device)
    print(f"  ✓ Using device: {device}")

    print("\n==> Testing forward pass...")
    param_batch = param.unsqueeze(0)
    condition_batch = condition.unsqueeze(0)
    loss = model(output_shape=param_batch.shape, x_0=param_batch, condition=condition_batch)
    print(f"  ✓ Training loss: {loss.item():.4f}")

    print("\n==> Testing generation...")
    with torch.no_grad():
        generated = model(sample=True, condition=condition_batch)
    print(f"  ✓ Generated shape: {generated.shape}")
else:
    print(f"  ⚠ Skipping forward/generation tests (Mamba requires CUDA)")

print("\n✅ All components working! Ready to train.")
