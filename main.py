import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import (
    EXPERIMENT_CONFIG_PATH,
    RESULTS_DIR,
    SUMMARY_COLUMNS,
    SUMMARY_PATH,
)
from src.utils.utils_logging import logger
from src.utils.utils_data import create_train_test_data
from src.utils.utils_models import build_model
from src.utils.utils_pipeline import (
    append_summary,
    create_full_job_name,
    generate_benchmark_jobs,
    generate_result_path,
    job_already_done,
    save_results,
    set_global_seed,
)


def main(experiment_path: str):
    """Run all benchmark jobs defined in the given experiment config."""

    logger.info("Benchmarking started.")
    benchmark_jobs = generate_benchmark_jobs(experiment_path)

    # Initialize or load summary file
    if SUMMARY_PATH.exists():
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

            df_train, df_test, n_features = create_train_test_data(job)
            x_cols = df_train.columns[df_train.columns.str.startswith("x")]

            X_train = df_train.loc[:, x_cols].to_numpy(dtype=np.float32)
            X_test = df_test.loc[:, x_cols].to_numpy(dtype=np.float32)

            y_train = np.stack(df_train["y"].to_numpy(), axis=0).astype(
                np.float32
            )  # (N, d)
            y_test = np.stack(df_test["y"].to_numpy(), axis=0).astype(
                np.float32
            )  # (N, d)

            # Normalize y_train
            y_mean = y_train.mean(axis=0)
            y_std = y_train.std(axis=0)
            y_train_norm = (y_train - y_mean) / y_std

            sigma_test = np.stack(df_test["sigma"].to_numpy(), axis=0).astype(
                np.float32
            )  # (N, d)

            epoch_losses_all = []
            preds_all = []
            epistemic_all = []
            aleatoric_all = []
            nll_all = []
            cal_score_all = []
            train_time_all = []
            inference_time_all = []
            for run_idx in range(job["model_runs"]):
                logger.info(f"Run {run_idx + 1}/{job['model_runs']}")
                seed = job["seed"] + run_idx
                set_global_seed(seed)
                logger.debug(f"Model params: {job['model_params']}")
                model = build_model(
                    job["model_name"], job["model_params"], seed, n_features
                )
                # Training time
                t0 = time.time()
                epoch_losses = model.fit(X_train, y_train_norm)
                t1 = time.time()
                train_time_all.append(t1 - t0)

                # Inference time
                t2 = time.time()
                y_pred, epistemic, aleatoric, nll, cal_score = (
                    model.predict_with_uncertainties(
                        X_test, y_test, y_mean, y_std, job["test_interval"]
                    )
                )
                t3 = time.time()
                inference_time_all.append(t3 - t2)

                # save results for this run
                epoch_losses_all.append(epoch_losses)
                preds_all.append(y_pred)
                epistemic_all.append(epistemic)
                aleatoric_all.append(aleatoric)
                nll_all.append(nll)
                cal_score_all.append(cal_score)

            # Save all results
            results_dir = generate_result_path(RESULTS_DIR, job)
            save_results(
                epoch_losses_all=np.stack(epoch_losses_all),
                preds_all=np.stack(preds_all),
                epistemic_all=np.stack(epistemic_all),
                aleatoric_all=np.stack(aleatoric_all),
                nll_all=nll_all,
                cal_score_all=cal_score_all,
                aleatoric_true=sigma_test**2,
                X_test=X_test,
                X_train=X_train,
                y_test=y_test,
                y_train=y_train,
                train_time_all=train_time_all,
                infer_time_all=inference_time_all,
                job=job,
                results_dir=results_dir,
            )
            # Update summary
            append_summary(job, results_dir)

            pbar.update(1)


if __name__ == "__main__":
    main(experiment_path=EXPERIMENT_CONFIG_PATH)
