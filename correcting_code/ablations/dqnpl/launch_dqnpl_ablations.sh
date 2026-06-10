#!/bin/bash
#SBATCH --job-name=nas_dqnpl
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=122536
#SBATCH --time=168:00:00

#SBATCH --output=logs_dqnpl_ablations/%x_%j.out
#SBATCH --error=logs_dqnpl_ablations/%x_%j.err

# ensure log directory exists
mkdir -p logs_dqnpl_ablations

# activate environment
source .venv/bin/activate

# run script
python3 dqnpl_ablations_full.py