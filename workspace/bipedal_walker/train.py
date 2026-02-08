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
from dataset import BipedalWalker_PPO as Dataset
from torch.utils.data import DataLoader

# Reward-weighted self-training
sys.path.insert(0, os.path.join(root, 'workspace', 'bipedal_walker'))
from reward_utils import (
    RewardBuffer, evaluate_generated_tensor, condition_to_env_params
)


config = {
    "seed": SEED,
    # Dataset setting
    "dataset": Dataset,
    "dim_per_token": 2048,  # Small policy network
    "sequence_length": 'auto',
    # Train setting
    "batch_size": 16,
    "num_workers": 8,
    "total_steps": 500_000,
    "learning_rate": 5e-4,
    "weight_decay": 0.0,
    "save_every": 5000,
    "print_every": 50,
    "autocast": lambda i: 50_000 < i < 400_000,
    "checkpoint_save_path": "./checkpoint",
    # Test setting
    "test_batch_size": 1,
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # Model config
    "model_config": {
        "num_permutation": 'auto',
        # Mamba config
        "d_condition": 4,  # [leg_length, leg_width, gravity, friction]
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
    "tag": "bipedal_walker_morphology_adapter",
    # Reward-weighted self-training
    "reward_loss_enabled": True,
    "reward_loss_weight": 0.1,       # lambda: auxiliary loss coefficient
    "eval_every": 1000,              # steps between generate-evaluate cycles
    "eval_episodes": 3,              # episodes per evaluation
    "reward_buffer_size": 50,        # max entries in replay buffer
    "reward_temperature": 1.0,       # sigmoid steepness for reward weighting
    "reward_loss_start": 20000,      # start using reward loss after model has learned basic structure
    "reward_weight_threshold": 0.5,  # only use samples with relative weight above this
    "reward_min_reward": -50.0,      # absolute floor: ignore policies worse than this
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

# Reward buffer
print('==> Initializing reward buffer..')
reward_buffer = RewardBuffer(
    capacity=config["reward_buffer_size"],
    temperature=config["reward_temperature"],
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
            # Standard diffusion loss on real training data
            base_loss = model(output_shape=param.shape, x_0=param, condition=condition, permutation_state=permutation_state)

            # Reward-weighted auxiliary loss from replay buffer
            aux_loss = torch.tensor(0.0, device=base_loss.device)
            reward_weight_used = 0.0
            if (config["reward_loss_enabled"]
                    and len(reward_buffer) > 0
                    and batch_idx >= config["reward_loss_start"]):
                buf_params, buf_cond, buf_reward = reward_buffer.sample(
                    device=base_loss.device
                )
                # Gate 1: absolute quality floor — skip garbage policies
                if buf_reward >= config["reward_min_reward"]:
                    reward_weight_used = reward_buffer.compute_weight(buf_reward)
                    # Gate 2: relative quality — only reinforce above-average
                    if reward_weight_used >= config["reward_weight_threshold"]:
                        raw_aux = reward_weight_used * model(
                            output_shape=buf_params.shape,
                            x_0=buf_params,
                            condition=buf_cond.unsqueeze(0),
                            permutation_state=None,  # generated with perm_state=False
                        )
                        # Clamp: aux_loss never exceeds base_loss magnitude
                        aux_loss = torch.min(raw_aux, base_loss.detach())

            loss = base_loss + config["reward_loss_weight"] * aux_loss

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(batch_idx)

        # Update progress bar with metrics
        current_lr = scheduler.get_last_lr()[0]
        postfix = {
            'loss': f'{loss.item():.6f}',
            'base': f'{base_loss.item():.6f}',
            'lr': f'{current_lr:.2e}',
        }
        if config["reward_loss_enabled"] and len(reward_buffer) > 0:
            postfix['aux'] = f'{aux_loss.item():.4f}' if isinstance(aux_loss, torch.Tensor) else '0'
            postfix['buf'] = len(reward_buffer)
        pbar.set_postfix(postfix)

        # Logging
        if USE_WANDB and accelerator.is_main_process:
            log_dict = {"train_loss": loss.item(), "base_loss": base_loss.item()}
            if isinstance(aux_loss, torch.Tensor) and aux_loss.item() > 0:
                log_dict["aux_loss"] = aux_loss.item()
                log_dict["reward_weight"] = reward_weight_used
            if len(reward_buffer) > 0:
                buf_stats = reward_buffer.stats()
                log_dict["buffer/size"] = buf_stats["size"]
                log_dict["buffer/baseline"] = buf_stats["baseline"]
                log_dict["buffer/best"] = buf_stats["max"]
            wandb.log(log_dict)
        elif not USE_WANDB:
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                avg_loss = train_loss/this_steps
                pbar.write(f'Step {batch_idx}: Avg Loss: {avg_loss:.6f}')
                this_steps = 0
                train_loss = 0

        # Periodic reward evaluation: generate → evaluate → store in buffer
        if (config["reward_loss_enabled"]
                and batch_idx % config["eval_every"] == 0
                and batch_idx > 0
                and accelerator.is_main_process):
            _reward_eval_step(batch_idx, pbar)

        # Save checkpoint
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            pbar.write(f'\n{"="*60}')
            pbar.write(f'💾 Saving checkpoint at step {batch_idx}')
            pbar.write(f'{"="*60}')
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = accelerator.unwrap_model(model).state_dict()
            torch.save(state, os.path.join(config["checkpoint_save_path"], config["tag"]+".pth"))
            generate(save_path=config["generated_path"], need_test=True)

        if batch_idx >= config["total_steps"]:
            break

    pbar.close()


def _reward_eval_step(step, pbar=None):
    """Generate a policy, evaluate in environment, store in reward buffer."""
    model.eval()
    with torch.no_grad():
        # Sample random condition from training set
        idx = torch.randint(0, len(train_set), (1,)).item()
        _, cond, _ = train_set[idx]
        cond_batch = cond.unsqueeze(0).to(next(model.parameters()).device)

        # Generate policy
        prediction = model(sample=True, condition=cond_batch, permutation_state=False)

    # Evaluate in environment (in-process, no subprocess)
    _, reward = evaluate_generated_tensor(
        prediction, train_set, cond,
        num_episodes=config["eval_episodes"],
    )

    # Store in buffer
    reward_buffer.add(prediction, cond, reward)
    buf_stats = reward_buffer.stats()

    msg = (f"  [Reward Eval @ step {step}] "
           f"reward={reward:.1f}  buf_size={buf_stats['size']}  "
           f"baseline={buf_stats['baseline']:.1f}  best={buf_stats['max']:.1f}")
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg)

    if USE_WANDB:
        wandb.log({
            "reward_eval/reward": reward,
            "reward_eval/buffer_baseline": buf_stats["baseline"],
            "reward_eval/buffer_best": buf_stats["max"],
        })

    model.train()
    return reward


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
        evaluate_generated_policy(save_path)
    model.train()
    return prediction


def evaluate_generated_policy(checkpoint_path, num_episodes=5):
    """Synchronously evaluate the generated policy and log results."""
    print(f"\n{'='*60}")
    print("📊 EVALUATING GENERATED POLICY")
    print(f"{'='*60}")

    try:
        # Run test.py and capture output
        result = subprocess.run(
            [sys.executable, "./experiments/bipedal_walker/test.py",
             checkpoint_path, "--num_episodes", str(num_episodes)],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        output = result.stdout
        print(output)

        # Parse mean reward from output
        mean_match = re.search(r'Mean:\s+([-\d.]+)', output)
        std_match = re.search(r'Std:\s+([-\d.]+)', output)

        if mean_match:
            mean_reward = float(mean_match.group(1))
            std_reward = float(std_match.group(1)) if std_match else 0.0

            print(f"\n✅ Evaluation complete: {mean_reward:.2f} ± {std_reward:.2f}")

            # Log to wandb
            if USE_WANDB:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                })

            # Save to file
            eval_log_path = "./checkpoint/eval_log.txt"
            with open(eval_log_path, "a") as f:
                f.write(f"{checkpoint_path}: {mean_reward:.2f} ± {std_reward:.2f}\n")

            return mean_reward
        else:
            print("⚠️  Could not parse evaluation results")
            return None

    except subprocess.TimeoutExpired:
        print("⚠️  Evaluation timed out")
        return None
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        return None
    finally:
        print(f"{'='*60}\n")


if __name__ == '__main__':
    train()
    del train_loader
    print("Finished Training!")
    exit(0)
