import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

FEAT_COLS = [
    "log_OPS", "log_params", "log_acts", "stride_inv",
    "CIN", "COUT", "KERNEL_SIZE", "STRIDES",
]
NUM_BASE = ["HW", "KERNEL_SIZE", "CIN", "COUT", "STRIDES"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    if "Energy_mJ" in df.columns and "energy_mJ" not in df.columns:
        ren["Energy_mJ"] = "energy_mJ"
    if "op_name" in df.columns and "OP" not in df.columns:
        ren["op_name"] = "OP"
    if ren: df = df.rename(columns=ren)
    return df

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df.copy())
    for c in NUM_BASE:
        if c not in df.columns: raise ValueError(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    HW = df["HW"].astype(np.float32)
    df["OPS"]    = HW * (df["KERNEL_SIZE"]**2) * df["CIN"] * df["COUT"] / (df["STRIDES"]**2)
    df["params"] = (df["KERNEL_SIZE"]**2) * df["CIN"] * df["COUT"]
    df["acts"]   = HW * df["COUT"]
    df["stride_inv"] = 1.0 / df["STRIDES"]
    for col in ("OPS","params","acts"):
        df[f"log_{col}"] = np.log10(df[col].clip(lower=1e-12).astype(np.float32))
    return df

def get_feature_matrix(df: pd.DataFrame, cols: List[str] = None) -> np.ndarray:
    cols = cols or FEAT_COLS
    return df[cols].astype(np.float32).values
