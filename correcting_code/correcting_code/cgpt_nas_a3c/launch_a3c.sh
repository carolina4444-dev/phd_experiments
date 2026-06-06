#!/bin/bash
#SBATCH --job-name=nas_a3c
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=122536
#SBATCH --time=04:00:00

#SBATCH --output=logs_a3c/%x_%j.out
#SBATCH --error=logs_a3c/%x_%j.err

# ensure log directory exists
mkdir -p logs_a3c

# activate environment
source .venv/bin/activate

# run script
python3 nats_refinement_dual_actor_critic.py