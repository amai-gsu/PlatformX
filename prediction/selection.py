#!/usr/bin/env python3
import numpy as np, pandas as pd
from typing import List
from features import derive_features, FEAT_COLS, get_feature_matrix
from estimator import Predictor

def _normalize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu)/sd

def kcenter_select(df_pool: pd.DataFrame, k: int, feat_cols: List[str] = None, seed: int = 42) -> pd.DataFrame:
    df = derive_features(df_pool.copy())
    X = get_feature_matrix(df, cols=feat_cols or FEAT_COLS).astype(np.float32)
    Xn = _normalize(X); n = len(df)
    if k >= n: return df.copy()
    rng = np.random.default_rng(seed); start = int(rng.integers(0, n))
    chosen = [start]; dist = np.full(n, np.inf, dtype=np.float32); last = start
    for _ in range(1, k):
        d = np.linalg.norm(Xn - Xn[last], axis=1)
        dist = np.minimum(dist, d); last = int(np.argmax(dist)); chosen.append(last)
    return df.iloc[chosen].copy()

def select_next_uncertainty_diversity(df_pool: pd.DataFrame, measured_ids: pd.Series, predictor: Predictor,
                                      k: int, shortlist_factor: int = 3, id_col: str = "id") -> pd.DataFrame:
    remaining = df_pool[~df_pool[id_col].isin(set(measured_ids.tolist()))].copy()
    if remaining.empty: return remaining
    preds = predictor.predict_configs(remaining.copy())
    top = preds.nlargest(k*shortlist_factor, "pred_std_mJ", keep="all")
    return kcenter_select(top.copy(), k=k, feat_cols=FEAT_COLS)
