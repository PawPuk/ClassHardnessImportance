#!/bin/bash

set -e

echo "========================================"
echo "Running Case Study 1 Experiments"
echo "========================================"

echo "Training models on balanced dataset (the baseline)..."

dataset_names=('CIFAR10' 'CIFAR100')
baseline_job_ids=()


for dataset_name in "${dataset_names[@]}"
do
    if [ "$dataset_name" == "CIFAR10" ]; then
        dataset_code="10"
    else
        dataset_code="100"
    fi

    job_name="${dataset_code}base"
    log_file="Output/output_train_baseline_models_${dataset_name}.out"

    jid=$(sbatch \
        --job-name="$job_name" \
        --output="$log_file" \
        scripts/slurm/train_baseline_models.sh "$dataset_name" | awk '{print $4}')

    baseline_job_ids+=($jid)
done

baseline_dependency=$(IFS=:; echo "${baseline_job_ids[*]}")
echo "Baseline jobs submitted: ${baseline_job_ids[*]}"
echo "Case study jobs will wait for baseline jobs to finish."

echo "Now training models on pruned subdatasets (case study 1)..."

pruning_rates=(33 42 50 58 66 75)
oversampling_strategies=('random' 'SMOTE' 'holdout')
case_study_job_ids=()

for pruning_rate in "${pruning_rates[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
      	for oversampling_strategy in "${oversampling_strategies[@]}"
        do
            if [ "$dataset_name" == "CIFAR10" ]; then
                dataset_code="10"
            else
                dataset_code="100"
            fi

            if [ "$oversampling_strategy" == "SMOTE" ]; then
                oversampling_code="S"
            elif [ "$oversampling_strategy" == "random" ]; then
                oversampling_code="r"
            else
                oversampling_code="h"
            fi

            job_name="${dataset_code}${oversampling_code}clp${pruning_rate}"
            log_file="Output/output_case_study_1_${dataset_name}_${oversampling_strategy}${pruning_rate}.out"

            jid=$(sbatch \
                --dependency=afterok:"$baseline_dependency" \
                --job-name="$job_name" \
                --output="$log_file" \
                scripts/slurm/run_experiments_for_case_study_1.sh \
                "$pruning_rate" "$dataset_name" "$oversampling_strategy" | awk '{print $4}')

            case_study_job_ids+=($jid)
        done
    done
done

case_dependency=$(IFS=:; echo "${case_study_job_ids[*]}")
echo "Case study jobs submitted: ${case_study_job_ids[*]}"
echo "Visualization will wait for these jobs."

for dataset_name in "${dataset_names[@]}"
do
    if [ "$dataset_name" == "CIFAR10" ]; then
        dataset_code="10"
    else
        dataset_code="100"
    fi

    job_name="cs1${dataset_code}vis"
    log_file="Output/output_case_study_1_visualization_${dataset_name}.out"

    sbatch --dependency=afterok:"$case_dependency" \
           --job-name="$job_name" \
           --output="$log_file" \
           scripts/slurm/visualize_case_study_1_results.sh "$dataset_name"
    # sbatch --job-name="$job_name" --output="$log_file" scripts/slurm/visualize_case_study_1_results.sh "$dataset_name"

done

echo "========================================"
echo "Case Study 1 experiments completed."
echo "========================================"

