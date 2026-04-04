import os

# Project root (parent of this app/ directory)
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_PATH     = os.environ.get("MODEL_PATH",     os.path.join(_BASE, "inference", "sleep_best_ckpt.keras"))
REG_MODEL_PATH = os.environ.get("REG_MODEL_PATH", os.path.join(_BASE, "regression_model.pkl"))

# Signal processing
TARGET_FS         = 100
EPOCH_SECONDS     = 30
SAMPLES_PER_EPOCH = TARGET_FS * EPOCH_SECONDS   # 3000
SEQ_LEN           = 5
SMOOTH_KERNEL     = 5

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Bandpass filter (matches training pipeline)
LOWCUT       = 0.5
HIGHCUT      = 30.0
FILTER_ORDER = 4

# Sleep quality score weights and thresholds
SE_MIN, SE_GOOD      = 0.70, 0.90
WASO_BAD_MIN         = 90.0
LATENCY_REF_MIN      = 45.0
N3_TARGET, REM_TARGET = 0.20, 0.22
FRAG_BAD_REF         = 0.12
W_SE, W_WASO, W_LATENCY, W_REST, W_FRAG = 0.30, 0.20, 0.15, 0.20, 0.15

# Input limits
MAX_SAMPLES = TARGET_FS * 3600 * 24   # 24 hours
MAX_EDF_MB  = 500

# External service keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_KEY           = os.environ.get("API_KEY", "")

# Static files
STATIC_DIR = os.path.join(_BASE, "static")