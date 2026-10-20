"""Configure fine-tuning, evaluation sizes, outputs, and release checks.

``MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS`` and ``MIN_BOUNDARY_CLASS_ROWS`` default to zero
because the label bands leave a deliberate gap around the flag threshold: each band keeps
a margin from the boundary, so the extreme bands carry the human/AI decision and the
40-60 boundary band can only be filled by evidence-scored rows whose measured detector
score happens to land there. Ask for those rows explicitly through the environment when a
repository pool actually contains them.
"""

import os
import time

from data.config import DEFAULT_CANDIDATE, DEFAULT_TEST, DEFAULT_TRAIN, DEFAULT_VAL
from data.validation import DEFAULT_MAX_LEN

from .config_validation import validate_config

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
MAX_LEN = DEFAULT_MAX_LEN
BASE_MODEL = os.environ.get(
    "HACKGRADER_BASE_MODEL", "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
)
RUN_ID = os.environ.get("HACKGRADER_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
MODELS_DIR = "models"
MERGE_DIR = (
    os.environ.get("HACKGRADER_MERGE_DIR")
    or f"{MODELS_DIR}/qwen3_ai_detector_merged-{RUN_ID}"
)
GGUF = (
    os.environ.get("HACKGRADER_GGUF")
    or f"{MODELS_DIR}/ai_detector_q8-{RUN_ID}.gguf"
)
ADAPTER_DIR = os.environ.get("HACKGRADER_ADAPTER_DIR") or os.path.join(
    "outputs", f"lora_adapter-{RUN_ID}"
)
TRAIN_FILE = os.environ.get("HACKGRADER_TRAIN_FILE", DEFAULT_TRAIN)
VAL_FILE = os.environ.get("HACKGRADER_VAL_FILE", DEFAULT_VAL)
TEST_FILE = os.environ.get("HACKGRADER_TEST_FILE", DEFAULT_TEST)
COMBINED_FILE = os.environ.get("HACKGRADER_DATASET_FILE", DEFAULT_CANDIDATE)
REPEAT = int(os.environ.get("HACKGRADER_REPEAT", "1"))
EPOCHS = int(os.environ.get("HACKGRADER_EPOCHS", "1"))
MAX_TRAINING_HOURS = 20.0
HOURS = float(os.environ.get("HACKGRADER_HOURS", "20"))
VALIDATION_SCORING_RESERVE_HOURS = min(4.0, HOURS / 8)
FINAL_TEST_RESERVE_HOURS = min(4.0, HOURS / 8)
MAX_GATE_CANDIDATES = int(os.environ.get("HACKGRADER_MAX_GATE_CANDIDATES", "4"))
MAX_STEPS_PER_ROUND = int(os.environ.get("HACKGRADER_MAX_STEPS_PER_ROUND", "500"))
MICRO_BATCH = int(os.environ.get("HACKGRADER_BATCH", "1"))
ACCUM = int(os.environ.get("HACKGRADER_ACCUM", "8"))
OPTIM = os.environ.get("HACKGRADER_OPTIM", "adamw_torch")
ASSISTANT_ONLY_LOSS = os.environ.get("HACKGRADER_ASSISTANT_ONLY_LOSS", "1") == "1"
CHECK_ROWS = int(os.environ.get("HACKGRADER_CHECK_ROWS", "100"))
MIN_TEST_ROWS = int(os.environ.get("HACKGRADER_MIN_TEST_ROWS", "50"))
MIN_TEST_CLASS_ROWS = int(os.environ.get("HACKGRADER_MIN_TEST_CLASS_ROWS", "10"))
EVAL_ROWS = int(os.environ.get("HACKGRADER_EVAL_ROWS", "48"))
EVAL_STEPS = int(os.environ.get("HACKGRADER_EVAL_STEPS", "50"))
MIN_TRAIN_ROWS = int(os.environ.get("HACKGRADER_MIN_TRAIN_ROWS", "1000"))
MIN_TRAIN_CLASS_ROWS = int(os.environ.get("HACKGRADER_MIN_TRAIN_CLASS_ROWS", "200"))
MIN_SCORE_BAND_ROWS = int(os.environ.get("HACKGRADER_MIN_SCORE_BAND_ROWS", "10"))
MIN_VALIDATION_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_VALIDATION_ROWS",
        str(max(300, CHECK_ROWS, EVAL_ROWS, MIN_TEST_ROWS)),
    )
)
MIN_VALIDATION_CLASS_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_VALIDATION_CLASS_ROWS",
        str(max(MIN_TEST_CLASS_ROWS, min(100, MIN_VALIDATION_ROWS // 2))),
    )
)
MIN_VALIDATION_SCORE_BAND_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_VALIDATION_SCORE_BAND_ROWS",
        str(max(MIN_SCORE_BAND_ROWS, (MIN_VALIDATION_ROWS + 2) // 3)),
    )
)
MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS = int(
    os.environ.get("HACKGRADER_MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS", "0")
)
MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS",
        os.environ.get("HACKGRADER_MIN_VALIDATION_SCORE_BAND_ROWS", "0"),
    )
)
MIN_GENERATED_TEST_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_DATA_TEST_ROWS", str(max(300, CHECK_ROWS, MIN_TEST_ROWS))
    )
)
MIN_GENERATED_TEST_CLASS_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_DATA_TEST_CLASS_ROWS",
        str(max(MIN_TEST_CLASS_ROWS, min(100, MIN_GENERATED_TEST_ROWS // 2))),
    )
)
MIN_GENERATED_TEST_SCORE_BAND_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_DATA_TEST_SCORE_BAND_ROWS",
        str(max(MIN_SCORE_BAND_ROWS, (MIN_GENERATED_TEST_ROWS + 2) // 3)),
    )
)
MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS = int(
    os.environ.get(
        "HACKGRADER_MIN_DATA_TEST_MIDDLE_SCORE_BAND_ROWS",
        os.environ.get("HACKGRADER_MIN_DATA_TEST_SCORE_BAND_ROWS", "0"),
    )
)
MIN_BOUNDARY_CLASS_ROWS = int(
    os.environ.get("HACKGRADER_MIN_BOUNDARY_CLASS_ROWS", "0")
)
MIN_VALIDATION_BOUNDARY_CLASS_ROWS = int(
    os.environ.get("HACKGRADER_MIN_VALIDATION_BOUNDARY_CLASS_ROWS", "0")
)
MIN_GENERATED_TEST_BOUNDARY_CLASS_ROWS = int(
    os.environ.get("HACKGRADER_MIN_DATA_TEST_BOUNDARY_CLASS_ROWS", "0")
)
LORA_LR = float(os.environ.get("HACKGRADER_LORA_LR", "1e-4"))
MIN_ADAPTIVE_LR = float(os.environ.get("HACKGRADER_MIN_ADAPTIVE_LR", str(LORA_LR / 10)))
SWEEP_CHECKPOINTS = os.environ.get("HACKGRADER_SWEEP_CHECKPOINTS", "1") == "1"
SWEEP_ROWS = int(os.environ.get("HACKGRADER_SWEEP_ROWS", "0"))
PATIENCE = int(os.environ.get("HACKGRADER_PATIENCE", "5"))
ASSUMED_SECONDS_PER_STEP = float(os.environ.get("HACKGRADER_SECONDS_PER_STEP", "22"))
BUDGET_SAFETY_FACTOR = float(os.environ.get("HACKGRADER_BUDGET_SAFETY_FACTOR", "1.25"))
MIN_HOLDOUT_SPREAD = float(os.environ.get("HACKGRADER_MIN_HOLDOUT_SPREAD", "4"))
HOLDOUT_MAE_LIMIT = float(os.environ.get("HACKGRADER_HOLDOUT_MAE_LIMIT", "20"))
HOLDOUT_MIN_PRECISION = float(os.environ.get("HACKGRADER_TEST_MIN_PRECISION", "0.80"))
HOLDOUT_MIN_RECALL = float(os.environ.get("HACKGRADER_TEST_MIN_RECALL", "0.80"))
HOLDOUT_MIN_SPECIFICITY = float(
    os.environ.get("HACKGRADER_TEST_MIN_SPECIFICITY", "0.80")
)
VRAM_FLOOR_GB = float(os.environ.get("HACKGRADER_VRAM_FLOOR_GB", "1.5"))
CACHE_LIMIT_GB = float(os.environ.get("HACKGRADER_CACHE_LIMIT_GB", "2"))
MIN_HOLDOUT_SEPARATION = float(
    os.environ.get("HACKGRADER_MIN_HOLDOUT_SEPARATION", "25")
)
_grad_checkpointing = os.environ.get("HACKGRADER_GRAD_CKPT", "true").lower()
GRAD_CHECKPOINTING: bool | str = {"true": True, "false": False}.get(
    _grad_checkpointing, _grad_checkpointing
)
RESUME_CHECKPOINT = os.environ.get("HACKGRADER_RESUME_CHECKPOINT")
RUN_DIR = os.environ.get("HACKGRADER_OUTPUT_DIR") or os.path.join("outputs", RUN_ID)
MIN_FREE_VRAM_GB = float(os.environ.get("HACKGRADER_MIN_FREE_VRAM_GB", "14"))

validate_config(
    {name: value for name, value in globals().copy().items() if name.isupper()}
)
