#!/bin/bash
#SBATCH --job-name=nas_bench
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

#SBATCH --array=0-269

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source .venv/bin/activate

METHODS=("random" "greedy" "heuristic")
BUDGETS=(100 500 1000)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 90))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 90))

BUDGET_INDEX=$((REMAINDER / 30))
SEED=$((REMAINDER % 30))

METHOD=${METHODS[$METHOD_INDEX]}
BUDGET=${BUDGETS[$BUDGET_INDEX]}

python run_experiment.py \
    --method $METHOD \
    --seed $SEED \
    --budget $BUDGET