from pathlib import Path
import os


# ============================================================
# U-Net Spatial Baseline Configuration
# ============================================================

# Set this environment variable on each machine:
#
# Windows PowerShell:
#   $env:WILDFIRE_DATA_ROOT="C:\path\to\Dataset"
#
# Linux/Mac:
#   export WILDFIRE_DATA_ROOT="/path/to/Dataset"
#
# Colab default:
#   /content/drive/MyDrive/Wildfire_Area_Regression /Dataset

DEFAULT_COLAB_DATA_ROOT = "/content/drive/MyDrive/Wildfire_Area_Regression /Dataset"

DATA_ROOT = Path(os.environ.get("WILDFIRE_DATA_ROOT", DEFAULT_COLAB_DATA_ROOT))

CACHE_ROOT = DATA_ROOT / "tensors_npy_cache"
UNET_WORKSPACE = DATA_ROOT / "unet_spatial_workspace"

STATS_PATH = UNET_WORKSPACE / "reports" / "module14_normalization_stats.json"


# ============================================================
# Dataset settings
# ============================================================

WINDOW_SIZE = 4
CROP_SIZE = 512
INPUT_CHANNELS = 40
OUTPUT_CHANNELS = 1

MIN_FIRE_PIXELS = 1

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"


# ============================================================
# Model settings
# ============================================================

BASE_CHANNELS = 16
MODEL_NAME = "UNetFire"


# ============================================================
# Training settings
# ============================================================

FULL_RUN_NAME = "unet_fireonly_normalized_firefiltered_crop512_base16_full_e10"

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

SELECTED_THRESHOLD = 0.65
THRESHOLD_FOR_EPOCH_METRICS = SELECTED_THRESHOLD

POSITIVE_CROP_PROB = 0.7
NUM_WORKERS = 0


# ============================================================
# Output folders
# ============================================================

FULL_RUN_DIR = UNET_WORKSPACE / FULL_RUN_NAME
CHECKPOINT_DIR = FULL_RUN_DIR / "checkpoints"
LOG_DIR = FULL_RUN_DIR / "logs"
REPORT_DIR = FULL_RUN_DIR / "reports"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{FULL_RUN_NAME}_best.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{FULL_RUN_NAME}_latest.pt"

TRAINING_LOG_CSV = LOG_DIR / "full_training_log.csv"


def create_output_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def print_config():
    print("DATA_ROOT:", DATA_ROOT)
    print("CACHE_ROOT:", CACHE_ROOT)
    print("UNET_WORKSPACE:", UNET_WORKSPACE)
    print("STATS_PATH:", STATS_PATH)
    print("FULL_RUN_DIR:", FULL_RUN_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("LOG_DIR:", LOG_DIR)
    print("REPORT_DIR:", REPORT_DIR)