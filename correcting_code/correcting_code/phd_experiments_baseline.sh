#!/bin/bash
#SBATCH --job-name=baseline_imdb                      
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --mem=122536
#SBATCH --cpus-per-task=2                                                                                                                    


source .venv/bin/activate

python3 darts_baseline_v4.py