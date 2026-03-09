#!/bin/bash

set -e

echo "========================================"
echo "Running Case Study 1 Experiments"
echo "========================================"

echo "Training models on balanced dataset (the baseline)..."
python -m src.experiments.train_baseline_models --dataset_name CIFAR100

echo "Training models on pruned subdatasets (case study 1)..."
python -m src.experiments.case_study_1 --dataset_name CIFAR100 --pruning_rate 42 --oversampling_strategy holdout
python -m src.experiments.case_study_1 --dataset_name CIFAR100 --pruning_rate 58 --oversampling_strategy holdout

echo ""
python -m src.experimtnes.visualize_case_study_1_results --dataset_name CIFAR100

echo "========================================"
echo "Case Study 1 experiments completed."
echo "========================================"
