import sys, os, json
import subprocess
import re
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)
with open("./workspace/config.json", "r") as f:
    additional_config = json.load(f)
USE_WANDB = additional_config["use_wandb"]

# Set global seed
import random
import numpy as np
import torch
seed = SEED = 999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Other imports
import warnings
from _thread import start_new_thread
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb

# Torch
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate.utils import DistributedDataParallelKwargs, AutocastKwargs
from accelerate import Accelerator

# Model
from model import MambaDiffusion as Model
from model.diffusion import DDPMSampler

# Dataset
from dataset import MLP_Regression_Train as Dataset
from torch.utils.data import DataLoader


config = {
    "seed": SEED,
    # Dataset setting
    "dataset": Dataset,
    "dim_per_token": 2048,
    "sequence_length": 'auto',
    # Train setting
    "batch_size": 16,
    "num_workers": 8,
    "total_steps": 100_000,
    "learning_rate": 5e-4,
    "weight_decay": 0.0,
    "save_every": 10_000,
    "print_every": 50,
    "autocast": lambda i: 5_000 < i < 80_000,
    "checkpoint_save_path": "./checkpoint",
    # Test setting
    "test_batch_size": 1,
    "generated_path": "./experiments/mlp_regression/test_generated/generated_model.pth",
    "test_command": f"python ./experiments/mlp_regression/test.py ./experiments/mlp_regression/test_generated/generated_model.pth",
    # Model config
    "model_config": {
        "num_permutation": 'auto',
        # Mamba config
        "d_condition": 4,           # (a, b, f, phi)
        "d_model": 2048,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        # Diffusion config
        "diffusion_batch": 512,
        "layer_channels": [1, 16, 32, 64, 32, 16, 1],
        "model_dim": "auto",
        "condition_dim": "auto",
        "kernel_size": 7,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 500,
        "forward_once": True,
    },
    "tag": "mlp_regression_adapter",
}


# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"])
print("Dataset length:", train_set.real_length)
print("Input shape:", train_set[0][0].shape)
if config["model_config"]["num_permutation"] == "auto":
    config["model_config"]["num_permutation"] = train_set.max_permutation_state
if config["model_config"]["condition_dim"] == "auto":
    config["model_config"]["condition_dim"] = config["model_config"]["d_model"]
if config["model_config"]["model_dim"] == "auto":
    config["model_config"]["model_dim"] = config["dim_per_token"]
if config["sequence_length"] == "auto":
    config["sequence_length"] = train_set.sequence_length
    print(f"Sequence length: {config['sequence_length']}")
else:
    assert train_set.sequence_length == config["sequence_length"]

train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    persistent_workers=True,
    drop_last=True,
    shuffle=True,
)

# Model
print('==> Building model..')
Model.config = config["model_config"]
model = Model(
    sequence_length=config["sequence_length"],
    positional_embedding=train_set.get_position_embedding(
        positional_embedding_dim=config["model_config"]["d_model"]
    )
)

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config["total_steps"],
)

# Accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# Wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key=additional_config["wandb_api_key"])
    wandb.init(project="Recurrent-Parameter-Generation", name=config['tag'], config=config)


# Training
print('==> Defining training..')
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> Start training..")
    model.train()

    # Create progress bar
    pbar = tqdm(enumerate(train_loader), total=config["total_steps"],
                desc="Training", unit="step", dynamic_ncols=True)

    for batch_idx, (param, condition, permutation_state) in pbar:
        optimizer.zero_grad()

        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            loss = model(output_shape=param.shape, x_0=param, condition=condition, permutation_state=permutation_state)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(batch_idx)

        # Update progress bar with metrics
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'lr': f'{current_lr:.2e}',
            'autocast': config["autocast"](batch_idx),
        })

        # Logging
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item()})
        elif not USE_WANDB:
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                avg_loss = train_loss/this_steps
                pbar.write(f'Step {batch_idx}: Avg Loss: {avg_loss:.6f}')
                this_steps = 0
                train_loss = 0

        # Save checkpoint
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = accelerator.unwrap_model(model).state_dict()
            torch.save(state, os.path.join(config["checkpoint_save_path"], config["tag"]+".pth"))
            generate(save_path=config["generated_path"], need_test=True)

        if batch_idx >= config["total_steps"]:
            break

    pbar.close()


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        # Sample a random condition from the training set
        random_idx = torch.randint(0, len(train_set), (1,)).item()
        _, condition, _ = train_set[random_idx]
        condition = condition.unsqueeze(0)  # Add batch dimension [1, d_condition]

        prediction = model(sample=True, condition=condition)
        generated_norm = prediction.abs().mean()
    print("Generated norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        evaluate_generated_checkpoint(save_path, condition.squeeze(0))
    model.train()
    return prediction


def evaluate_generated_checkpoint(checkpoint_path, condition):
    """Evaluate the generated MLP against its target function."""
    print(f"\n{'='*60}")
    print("EVALUATING GENERATED MLP")
    print(f"{'='*60}")

    try:
        a, b, f, phi = condition[0].item(), condition[1].item(), condition[2].item(), condition[3].item()
        result = subprocess.run(
            [sys.executable, "./experiments/mlp_regression/test.py",
             checkpoint_path, "--condition", str(a), str(b), str(f), str(phi)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout
        print(output)

        # Parse MSE from output
        mse_match = re.search(r'MSE:\s+([\d.]+)', output)
        if mse_match:
            mse = float(mse_match.group(1))
            print(f"Evaluation MSE: {mse:.6f}")
            if USE_WANDB:
                wandb.log({"eval/mse": mse})

            # Save to log
            eval_log_path = "./checkpoint/eval_log_mlp.txt"
            with open(eval_log_path, "a") as logf:
                logf.write(f"{checkpoint_path}: MSE={mse:.6f} (a={a:.2f},b={b:.2f},f={f:.2f},phi={phi:.2f})\n")
            return mse
        else:
            print("Could not parse MSE from output")
            return None

    except subprocess.TimeoutExpired:
        print("Evaluation timed out")
        return None
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None
    finally:
        print(f"{'='*60}\n")


if __name__ == '__main__':
    train()
    del train_loader
    print("Finished Training!")
    exit(0)
