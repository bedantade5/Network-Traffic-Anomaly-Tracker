"""Read and validate the processed UNSW-NB15 train/test splits."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"id", "dur", "sbytes", "dbytes", "spkts", "dpkts", "proto", "service", "state"}
NUMERIC = ("id", "label", "dur", "sbytes", "dbytes", "spkts", "dpkts")


def load_flows(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Expected a .csv or .parquet file: {path}")
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after standardization")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    input_rows = len(frame)
    # Missing records are discarded, as specified in the synopsis; zero values remain.
    frame = frame.dropna().copy()
    for name in NUMERIC:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna().copy()
    frame = frame.replace([np.inf, -np.inf], 0)
    if not frame["id"].is_unique:
        raise ValueError(f"Connection id must be unique within each split: {path}")
    if "label" in frame and not frame["label"].isin([0, 1]).all():
        raise ValueError("label must contain only 0 (normal) and 1 (attack)")
    return frame.reset_index(drop=True), {"input_rows": input_rows, "retained_rows": len(frame), "dropped_rows": input_rows - len(frame)}
