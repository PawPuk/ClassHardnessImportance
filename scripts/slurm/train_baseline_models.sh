#!/bin/bash
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load the modules required by our program
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

dataset_name=$1

python3 -m src.experiments.train_baseline_models --dataset_name "$dataset_name"

