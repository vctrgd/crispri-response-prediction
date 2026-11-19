import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("KAGGLE_DATA_DIR", PROJECT_ROOT / "data"))
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

SEED = 8245

SCORED_LINES = ("K562", "RPE1", "Jurkat", "HepG2")
EXTRA_LINES = ("HCT116", "HEK293T")
ALL_LINES = SCORED_LINES + EXTRA_LINES

META_COLS = (
    "id",
    "line",
    "target_gene",
    "target_symbol",
    "n_cells",
    "fold",
    "extra_line",
)
