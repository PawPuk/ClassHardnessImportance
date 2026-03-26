#!/bin/bash
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

# Load the modules required by our program
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

pruning_rate=$1
dataset_name=$2
oversampling_strategy=$3

python3 -m src.experiments.case_study_1 \
        --dataset_name "$dataset_name" \
        --pruning_rate "$pruning_rate"  \
        --oversampling_strategy "$oversampling_strategy"
