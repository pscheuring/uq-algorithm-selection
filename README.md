<div align="center">

# *Master Thesis*
# Toward Human-Centered Automated Machine Learning: An Empirical Study on Deep Learning-based Uncertainty Quantification for Time Series Algorithm Selection
</div>

This repository contains the full code accompanying the master thesis *“Toward Human-Centered Automated Machine Learning: An Empirical Study on Deep Learning-based Uncertainty Quantification for Time Series Algorithm Selection”*. 

## Abstract
Selecting suitable time series algorithms is a crucial step in real-world applications that rely on the accurate analysis of complex temporal data. Because different algorithms excel at different tasks, the large variety of available methods requires extensive evaluation to identify the most suitable time series algorithm for a new task. Existing approaches to time series algorithm selection address this challenge by using a meta-model trained on the characteristics of previous tasks to predict the best algorithm for a new task. However, they rarely account for different forms of uncertainty. Such predictions inherently involve epistemic uncertainty arising from limited knowledge of the meta-model and aleatoric uncertainty due to noise in the data. As a result, users do not know when to trust an algorithm recommendation and when to treat it with caution. This study addresses this issue by investigating how epistemic and aleatoric uncertainty can be integrated into time series algorithm selection systems. By identifying key characteristics of such systems, the work introduces an experimental framework that enables their systematic investigation in controlled experiments and reveals how these properties shape epistemic and aleatoric uncertainty. The results show that existing uncertainty quantification methods can handle the multidimensional nature of the data used to train meta-models, but sufficient data coverage is essential for obtaining reliable estimates of aleatoric uncertainty. Moreover, the quality of epistemic uncertainty estimates depends on the chosen method, and the evaluation metrics should be selected according to the intended purpose of the uncertainty assessment. These findings provide a basis for extending existing time series algorithm selection systems so that they not only achieve high predictive accuracy but also communicate uncertainty reliably, thereby moving toward human-centered automated machine learning systems.


## Overview of the Five Experiments

**Experiment 1 — Unique Samples (*N*<sub>unique</sub>):**  
This experiment investigates how the number of available unique training instances influences predictive performance and the resulting uncertainty components. Here, a unique training instance refers to a distinct meta-feature configuration, each associated with a single target value. This setup corresponds to a benchmarking scenario in which each time series algorithm is evaluated only once per dataset, yielding a single observed performance value for each meta-feature configuration.

**Experiment 2 — Repeated Observations (*N*<sub>rep</sub>):**  
In real meta-learning datasets for time series AS, multiple identical meta-feature inputs may be associated with varying target values, which we denote as repeated observations. This experiment examines how different numbers of repeated observations affect uncertainty estimates, with a particular focus on aleatoric uncertainty.

**Experiment 3 — In-Between OOD:**  
In-between OOD refers to inputs that fall within the nominal input range but in regions that were never observed during training, typically due to gaps between disjoint intervals of available data. Thus, it represents an OOD condition that arises inside the nominal domain rather than beyond its boundaries. This experiment investigates how models behave when exposed to in-between OOD conditions with a focus on epistemic uncertainty.

**Experiment 4 — Two-Dimensional Feature Space:**  
This experiment extends the setup of Experiment 1 to a two-dimensional input space to examine whether the uncertainty patterns observed in the one-dimensional case persist when additional features are incorporated. It evaluates how well uncertainty estimates generalize to higher-dimensional settings by varying the number of unique training instances.

**Experiment 5 — Multi-Target Prediction:**  
This experiment investigates how moving from single-target to multi-target prediction affects predictive behavior and uncertainty estimates, reflecting realistic settings in which multiple algorithm performances must be predicted simultaneously. By comparing single-target and two types of multi-target configurations, the experiment analyzes how inter-target relationships shape uncertainty.


## Repository Structure

```text
.
├── main.py                               # Main entry point
├── pyproject.toml                        # Project/build configuration
├── uv.lock                               # uv lockfile
├── LICENSE                               # License file
├── README.md                             # Project documentation
├── .gitignore                            # Git ignore rules

├── config/                               # Experiment configs
│   ├── exp1_n_unique.yaml
│   ├── exp2_n_repeat.yaml
│   ├── exp3_inbetween_ood.yaml
│   ├── exp4_two_feat.yaml
│   ├── exp5_multi_target_distinct.yaml
│   ├── exp5_multi_target_similar.yaml

├── data/                                 # Meta-learning data
│   ├── algorithm_performances/           # Time series algorithm benchmarks
│   └── meta_feature_values_norm.csv      # Normalized meta-feature values

├── images/                               # Generated figures and plots
│   ├── 01_exp1_n_unique/
│   ├── 02_exp2_n_repeat/
│   ├── 03_exp3_inbetween_ood/
│   ├── 04_exp4_two_feat/
│   ├── 05_exp5_multi_target/
│   └── 08_concept_figures/

├── logs/                                 # Experiment and run logs

├── notebooks/
│   ├── 01_exp1_n_unique.ipynb            # Exp 1 — results & figures
│   ├── 02_exp2_n_repeat.ipynb            # Exp 2 — results & figures
│   ├── 03_exp3_inbetween_ood.ipynb       # Exp 3 — results & figures
│   ├── 04_exp4_two_feat.ipynb            # Exp 4 — results & figures
│   ├── 05_exp5_multi_target.ipynb        # Exp 5 — results & figures
│   ├── 06_exp_eval_metrics.ipynb         # LaTeX-table eval metric tables
│   ├── 07_derivation_dgp.ipynb           # Derivation of the DGP
│   └── 08_hyperparameter_tuning.ipynb    # UQ model hyperparameter tuning
│   ├── 09_concept_figures.ipynb          # Thesis concept figures

├── results/                              # Final stored results

├── src/                                  # Source code
│   ├── models/                           # UQ model implementations
│   │   ├── base_model.py                  # Base NN architecture
│   │   ├── bbb.py                         # Bayes by Backprop
│   │   ├── ensemble.py                    # Deep Ensemble
│   │   ├── evidential.py                  # Depe evidential regression
│   │   ├── layers.py                      # Custom NN layers
│   │   ├── losses.py                      # Custom loss functions
│   │   └── mcdropout.py                   # MC Dropout

│   ├── utils/                             # Utility modules
│   │   ├── utils_derivation_dgp.py        # Candidate funcs for DGP
│   │   ├── utils_data.py                  # Train / Test sampling
│   │   ├── utils_logging.py               # Logging setup
│   │   ├── utils_models.py                # Model factory
│   │   ├── utils_pipeline.py              # Pipeline logic
│   │   ├── utils_results.py               # Results loading

│   ├── visualizations/                    # Experiment results visualizations

│   ├── constants.py                       # Global constants
│   ├── data_sampler.py                    # Data generation / sampling
│   └── evaluation.py                      # Metric computation & evaluation logic
```

## Reproduce Experiment Results

**Requirements:**  
- Python >=3.11
- [uv](https://docs.astral.sh/uv/) for dependency management

To reproduce any experiment, follow these steps:
1. **Install dependencies:**  
   ```bash
   uv sync
   ```
2. **Select the corresponding experiment config file**  
   In `main.py`, set the desired experiment by passing the appropriate config path, e.g.:

   ```python
   if __name__ == "__main__":
       main(experiment_path=EXP1_CONFIG_PATH)
    ```
   Available config files:
      - `EXP1_CONFIG_PATH`
      - `EXP2_CONFIG_PATH`
      - `EXP3_CONFIG_PATH`
      - `EXP4_CONFIG_PATH`
      - `EXP5_DISTINCT_CONFIG_PATH`
      - `EXP5_SIMILAR_CONFIG_PATH`

3. **Run the benchmark pipeline:**

   Run the pipeline from the project root:
   ```bash
   uv run main.py
   ```
  
4. **Results are saved automatically**  
   All outputs — including target predictions, epistemic / aleatoric uncertainty, losses, NLL values, and train/infer times — are stored in: `results/<experiment_name>/<model_name>/<full_job_name>/<timestamp>`
   Additionally, a new entry is appended to: 
   `results/benchmark_summary.csv`


5. **Generate metrics and figures for the thesis**  
Open the corresponding notebook for each experiment (e.g., `01_exp1_n_unique.ipynb`) and load the results by adjusting the paths inside  
`load_meta_model_benchmarking_results()`.

    **Example:**

    ```python
    bbb_instances_40 = load_meta_model_benchmarking_results(
        "exp1_n_unique/bbb/seed-42_fn-1_nz-1_tri-[-4.0,4.0]_tei-[-6.0,6.0]_trn-40x0_grid-1000_model-bbb/251031_1825"
    )
    ```
    Run the notebook `exp_eval_metrics.ipynb` to load all eval metrics and create .tex files for latex tables
    This workflow allows you to fully reproduce all results and figures presented in the thesis.

## Reproduce Data-Generating Process

To reproduce the synthetic data-generating process used throughout all experiments:

- Run the notebook `07_derivation_dgp.ipynb`.
- In the section **"print functions of all algorithms"**, the notebook will generate all DGP functions.
- These functions are the exact ones used inside `data_sampler.py` to sample training and test data for each experiment.

## Reproduce Hyperparameter Tuning

- Run the notebook `09_hyperparameter_tuning.ipynb` to reproduce the HPO process for all UQ models.
- The final hyperparemters used in this work are those in the experiment config files
