import logging
from datetime import datetime
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Subdirectories
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Timestamped log directory
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = BASE_DIR / "logs" / RUN_TIMESTAMP
LOG_PATH = LOG_DIR / "main.log"

# Experiment configuration paths
EXP1_CONFIG_PATH = CONFIG_DIR / "exp1_n_unique.yaml"
EXP2_CONFIG_PATH = CONFIG_DIR / "exp2_n_repeat.yaml"
EXP3_CONFIG_PATH = CONFIG_DIR / "exp3_inbetween_ood.yaml"
EXP4_CONFIG_PATH = CONFIG_DIR / "exp4_two_feat.yaml"
EXP5_DISTINCT_CONFIG_PATH = CONFIG_DIR / "exp5_multi_target_distinct.yaml"
EXP5_SIMILAR_CONFIG_PATH = CONFIG_DIR / "exp5_multi_target_similar.yaml"

# Matplotlib config path
STYLE_PATH = CONFIG_DIR / "visualizations.mplstyle"

# Summary and meta-data paths
SUMMARY_PATH = RESULTS_DIR / "benchmark_summary.csv"
META_FEATURES_PATH = DATA_DIR / "meta_feature_values_norm.csv"
ALGORITHM_PERFORMANCE_PATH = DATA_DIR / "algorithm_performances"

# Benchmark summary
SUMMARY_COLUMNS = [
    "experiment_name",
    "model_runs",
    "seed",
    "function",
    "noise",
    "train_interval",
    "train_instances",
    "train_repeats",
    "test_interval",
    "test_grid_length",
    "model_name",
    "model_params",
    "result_folder",
    "timestamp",
]

# Multi-target evaluation function for experiments 5
MULTI_TARGET_EVAL_FUNC = "f_cosine_hc2_1_feat"

### Logging ###
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PROJECT_LOGGER_NAME = "uq_algorithm_selection"
LOGGING_LEVEL = logging.WARN
