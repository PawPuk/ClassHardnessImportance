#!/bin/bash
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

# Load the modules required by our program
module load Anaconda3/2022.05
module load CUDA/10.2.89-GCC-8.3.0
source activate pytorch

dataset_name=$1
oversampling_strategy=$2
alpha=$3

python3 -m src.experiments.case_study_2 \
           --dataset_name "$dataset_name" \
           --oversampling "$oversampling_strategy" \
           --undersampling easy --alpha "$alpha"
