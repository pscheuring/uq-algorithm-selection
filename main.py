import os
import time
from src.logging import logger

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import EXPERIMENT_FEATURES_PATH, SUMMARY_COLUMNS, SUMMARY_PATH
from src.utils import (
    build_model,
    create_full_job_name,
    create_train_test_data,
    generate_benchmark_jobs,
    job_already_done,
    save_results,
    set_global_seed,
)


def main(n_runs, experiment_path: str):
    logger.info("Benchmarking started.")
    benchmark_jobs = generate_benchmark_jobs(EXPERIMENT_FEATURES_PATH)

    # Initialize or load summary file
    if os.path.exists(SUMMARY_PATH):
        summary_df = pd.read_csv(SUMMARY_PATH)
    else:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        summary_df.to_csv(SUMMARY_PATH, index=False)

    # Run all benchmark_jobs
    with tqdm(
        total=len(benchmark_jobs), desc="Running benchmark_jobs", ncols=100
    ) as pbar:
        for job in benchmark_jobs:
            full_job_name = create_full_job_name(job)
            tqdm.write(f"Now running: {job['function']} | {job['model_name']}")

            if job_already_done(job):
                logger.debug(f"Skipping job (already exists): {full_job_name}")
                pbar.update(1)
                continue

            logger.debug(f"Started job: {full_job_name}")
            df_train, df_test = create_train_test_data(job)
            x_cols = df_train.columns[df_train.columns.str.startswith("x")]

            X_train = df_train.loc[:, x_cols].to_numpy(dtype=np.float32)
            X_test = df_test.loc[:, x_cols].to_numpy(dtype=np.float32)

            y_train = df_train["y"].to_numpy(dtype=np.float32)
            y_test = df_test["y"].to_numpy(dtype=np.float32)
            noise_test = df_test["noise"].to_numpy(dtype=np.float32)

            all_preds = []
            all_epistemic = []
            all_aleatoric = []
            train_times = []
            inference_times = []

            for run_idx in range(job["model_runs"]):
                logger.info(f"Run {run_idx + 1}/{n_runs}")
                seed = job["random_seed"] + run_idx
                set_global_seed(seed)
                model = build_model(job["model_name"], job["model_params"])

                # Training time
                t0 = time.time()
                model.fit(X_train, y_train)
                t1 = time.time()
                train_times.append(t1 - t0)

                # Inference time
                t2 = time.time()
                if job["model_name"] == "bnn":
                    y_pred, epistemic, aleatoric = model.predict_with_uncertainties(
                        X_test, n_mc_samples=job["model_params"]["n_mc_samples"]
                    )
                else:
                    y_pred, epistemic, aleatoric = model.predict_with_uncertainties(
                        X_test
                    )
                t3 = time.time()
                inference_times.append(t3 - t2)

                # save results for this run
                all_preds.append(y_pred)
                all_epistemic.append(epistemic)
                all_aleatoric.append(aleatoric)

            # Save all results
            save_results(
                preds_all=np.stack(all_preds),
                epistemic_all=np.stack(all_epistemic),
                aleatoric_all=np.stack(all_aleatoric),
                aleatoric_true=noise_test,
                X_test=X_test,
                X_train=X_train,
                y_test=y_test,
                y_train=y_train,
                train_times=train_times,
                infer_times=inference_times,
                job=job,
            )


if __name__ == "__main__":
    main(experiment_path=EXPERIMENT_FEATURES_PATH)
