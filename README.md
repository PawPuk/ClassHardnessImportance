# Experiment 1

This is the first part of our experimental pipeline - training and ensemble of networks on the original datasets.
This program accepts two parameters ``dataset_name``, and ``remove_noise``. To obtain the results on CIFAR-10 please 
run the below.

```
python3 experiment3.py --dataset_name CIFAR10
```

Running the above will result in training an ensemble of 20 networks (see `dataset_configs[CIFAR10][num_models]` in `config.py`) and 
save it in `dataset_configs[CIFAR10][save_dir]`. On top of that the time taken for training will be recorder in 
`dataset_configs[CIFAR10][timings_dir]`, and the hardness estimates, produced via AUM and Forgetting, will be recorded 
in the directory specified by the `hardness_save_dir` variable in the `save_results` method of the `train_ensemle.py`.
*Therefore, please modify the `ROOT` variable from config.py.*

### Parameters

  - ``dataset_name`` - string that specifies the name of the dataset. Currently, only supports `CIFAR10` and `CIFAR100`.
  - ``remove_noise`` - boolean flag that, if called, removes a subset of the hardest samples that were identified as 
noise by AUM before training. In order to raise this flag you firstly need to run `experiment1.py` without raising this 
flag to compute the hardness estimates via AUM that are required for noise removal.

### Important

To reiterate, make sure to alter the `ROOT` variable from `config.py`, to determine the saving locations. This is also 
important for other programs in this pipeline.

# Verify Statistical Significance

The purpose of this part is to perform the robustness analysis of the trained ensemble with respect to the data pruning 
and resampling ratio computation.

```
python3 verify_statistical_significance.py --dataset_name CIFAR10
```

This program takes the same parameters as `experiment1.py`. Running it will produce the EL2N hardness estimates, as 
well as the parts of Fig. 7, and Fig. 8 which correspond to CIFAR-10.

# Experiment 2

This program is responsible for the data pruning case study (Sec. III-C).

### Parameters

  - ``dataset_name`` - same as in other programs
  - ``pruning_strategy`` - specifies the pruning strategy. Choose either `fclp`—fixed class-level pruning, which we call class-level pruning in our paper for simplicity—or `dlp`—dataset-level-pruning.
  - ``pruning_rate`` - specifies the percentage of data samples that will be removed during data pruning (please use integers). In our paper we used pruning rates of 10, 20, 30, 40, 50, 60, 70 (and 80 for CIFAR-10).
  - ``hardness_estimator`` - allows the used to select the hardness estimator used for pruning. Our experiments were performed for AUM.

### Example

Running 

```
python3 experiment2.py --dataset_name CIFAR10, --pruning_strategy dlp --pruning_rate 30 --hardness_esimator AUM
``` 

will result in training an ensemble of 20 models (see `dataset_configs[CIFAR10][num_models]` in `config.py`) that was
trained on a subset of CIFAR-10 obtained by pruning 30% of the easiest samples from the dataset, as specified by AUM.
This type of pruning will result in an introduction of data imbalance into the pruned subset, which is visualized in the 
Figure saved in `{ROOT}/Figures/{pruning_strategy}{pruning_rate}/{dataset_name}/sorted_class_level_sample_distribution.pdf`.

# Visualize Pruning Performance

After running `experiment2.py` for different pruning rates and pruning strategy you can visualize the result using 
`visualize_performance.py`. This program takes only `--dataset_name` as parameter, and creates:

  - Figure 4 from Supplementary Material (plot_pruned_percentages) - class-level vs dataset-level pruned percentages to 
show the imbalanced introduced by dataset-level pruning (DLP).
  - Figure 5 from Supplementary Material (plot_class_level_results) - recall averaged over models of ensembles 
trained on subsets of datasets obtained via DLP and class-level pruning (CLP). Not implemented for CIFAR-100 due to too 
many classes.
  - Figure 12 from the main text and 6 from Supplementary Material (compare_fclp_with_dlp) - recall averaged over 
classes for ensembles trained on subsets of datasets obtained via DLP and CLP.

### Important

Make sure you have run `experiment2.py` sufficient number of times producing enough results for the visualizations to 
make sense. To reduce computational complexity we suggest reducing the number of models in config.py.

# Experiment 3

This is the main part responsible for the resampling experiments. The program firstly resamples the specified dataset
using specified over- and undersampling techniques, and using the resampling ratios obtained using specified hardness
estimator and alpha.

### Parameters

  - `--dataset_name` - the same as in previous programs
  - `--oversampling` - string specifying the strategy used for oversampling. The allowed options are: 1) random; 2) 
easy; 3) hard; 4) SMOTE; and 5) none (indicating no oversampling, which is important for ablation study).
  - `--undersampling` - string specifying the strategy used for undersampling. The allowed options are: 1) easy; and 2) 
none (indicating no undersampling, which is important for ablation study).
  - `--hardness_estimator` - string specifying the hardness estimator used to compute the resampling ratios. The allowed
options are: 1) EL2N, 2) AUM; and 3) Forgetting. In our paper we report only results for AUM as it was found to be the 
most robust.
  - `--remove_noise` - the same as in previous programs.
  - `--alpha` - integer that controls the degree of introduced imbalance (see Equation 4 from the main text).

### Important

Make sure you have run `experiment1.py` before this one to ensure the hardness estimates have been computed (and 
`verify_statistical_siginificance` if you want to use EL2N as the hardness estimator).

# Visualize Resampling Performance

After running `experiment3.py` for different alphas you cna visualize the results using 
`visualise_resampling_effects.py`. This program takes only `--dataset_name` as parameter, and creates:

  - variant of Figure 9 with results for different alphas (plot_all_accuracies_sorted) - class-level metric values (F1, 
MCC, Recall, Precision, Accuracy, ...) with classes sorted based on their hardness (computed from `experiment1.py`).
  - Figure 11 (plot_metric_changes) - changes in class-level metric values due to resampling with classes sorted based
on their hardness (computed from `experiment1.py`).

### Important

Make sure you have run `experiment3.py` for at least one resampling strategy pair (oversampling, undersampling), one 
hardness estimator and one alpha. To reduce computational complexity we suggest reducing the number of models in 
config.py.

# Noise Removal

This program is being executed when you raise the `--remove_noise` flag in `experiment1.py` or `experiment3.py`. Its
purpose is to identify the label noise using AUM and remove it from the dataset. On top of that it also produces the
following:

  - Figure 5b (plot_cumulative_distribution) - the cumulative distribution of hardness across all data samples in the
dataset.
  - Figure 5a (plot_removed_samples_distribution) - the distribution of the removed samples (the ones that we 
identified as noise) across classes.
  - Figure 6 (visualize_lowest_aum_samples) - top 30 hardest samples, which are also the samples most likely to be 
mislabeled according to AUM.

# Contact

In case you have any questions regarding the code please contact *ppukowski1@sheffield.ac.uk*