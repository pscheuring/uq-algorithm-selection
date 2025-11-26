# Deep Learning-based Uncertainty Quantification for Time Series Algorithm Selection

This repository contains the full code accompanying the master thesis *“Toward Human-Centered Automated Machine Learning: An Empirical Study on Deep Learning-based Uncertainty Quantification for Time Series Algorithm Selection”*. 

The project investigates how epistemic and aleatoric uncertainty can be modeled, quantified, and evaluated within meta-learning systems that recommend time series algorithms. With this repository, all results and figures of the conducted experiments presented in the thesis can be fully reproduced.


## Overview of the Five Experiments

1. **Experiment 1 — Unique Samples (*N*<sub>unique</sub>):**  
   Investigates how increasing the number of unique training instances affects epistemic and aleatoric uncertainty.

2. **Experiment 2 — Repeated Samples (*N*<sub>rep</sub>):**  
   Examines how repeated evaluations of the same instance influence specifically aleatoric uncertainty. More repetitions help models learn the inherent noise structure more reliably.

3. **Experiment 3 — In-Between OOD:**  
   Tests model behavior when predictions are required in regions between two well-covered training areas. This evaluates whether models meaningfully express increased epistemic uncertainty under distribution shift.

4. **Experiment 4 — Two-Dimensional Feature Space:**  
   Extends the experiments to a two-dimensional input space to evaluate how well uncertainty estimates generalize to more complex patterns.

5. **Experiment 5 — Multi-Target Prediction:**  
   Analyzes uncertainty when predicting multiple targets (e.g. algorithm performances) simultaneously, reflecting realistic meta-learning setups with correlated targets.


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
