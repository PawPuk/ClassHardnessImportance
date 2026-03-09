#!/bin/bash

set -e

echo "========================================"
echo "Running Case Study 2 Experiments"
echo "========================================"

echo "Training models on balanced dataset (the baseline)..."
python -m src.experiments.train_baseline_models --dataset_name CIFAR100

echo "Training models on resampled full dataset (case study 2)..."
# python -m src.experiments.case_study_2 --dataset_name CIFAR100 --oversampling random --undersampling easy --alpha 2
# python -m src.experiments.case_study_2 --dataset_name CIFAR100 --oversampling SMOTE --undersampling easy --alpha 2
# python -m src.experiments.case_study_2 --dataset_name CIFAR100 --oversampling rEDM --undersampling easy --alpha 2
# python -m src.experiments.case_study_2 --dataset_name CIFAR100 --oversampling aEDM --undersampling easy --alpha 2
python -m src.experiments.case_study_2 --dataset_name CIFAR100 --oversampling hEDM --undersampling easy --alpha 2

echo ""
python -m src.experimtnes.visualize_case_study_2_results --dataset_name CIFAR100

echo "========================================"
echo "Case Study 2 experiments completed."
echo "========================================"