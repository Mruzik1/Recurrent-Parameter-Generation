#!/bin/bash
#SBATCH --job-name=gen_dataset        # Job name
#SBATCH --output=test_logs/test_%j.out   # Standard output log (%j expands to jobId)
#SBATCH --error=test_logs/test_%j.err    # Standard error log
#SBATCH --nodes=1                    # Run on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=32G                    # Memory limit
#SBATCH --time=48:00:00              # Time limit
#SBATCH --partition=dgx              # Submit to GPU partition

cd ~/Recurrent-Parameter-Generation
source .venv/bin/activate
mkdir -p test_logs

python experiments/bipedal_walker/generate_dataset.py --num_conditions 3333 --num_seeds 3 --output_dir test_data/bipedal_walker_data/bipedal_walker --num_workers 4 --total_timesteps 500000 --sampling_method mixed --min_reward -50 --eval_episodes 10