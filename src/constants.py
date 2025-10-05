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

# File paths
LOG_PATH = LOG_DIR / "main.log"
DEFAULT_EXPERIMENT_PATH = CONFIG_DIR / "default_experiment.yaml"
EXPERIMENT_FEATURES_PATH = CONFIG_DIR / "experiment_features.yaml"
EXPERIMENT_INSTANCES_PATH = CONFIG_DIR / "experiment_instances.yaml"
EXPERIMENT_REPEATS_PATH = CONFIG_DIR / "experiment_repeats.yaml"
SUMMARY_PATH = RESULTS_DIR / "benchmark_summary.csv"


### Benchmark summary ###
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

### Logging ###
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PROJECT_LOGGER_NAME = "uq_algorithm_selection"
LOGGING_LEVEL = logging.INFO
