#!/bin/bash
#SBATCH --job-name=nas_dqnpl
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=122536
#SBATCH --time=04:00:00

#SBATCH --output=logs_dqnpl/%x_%j.out
#SBATCH --error=logs_dqnpl/%x_%j.err

# ensure log directory exists
mkdir -p logs_dqnpl

# activate environment
source .venv/bin/activate

# run script
python3 dqnpl_conv_nats_refinement_v2.py