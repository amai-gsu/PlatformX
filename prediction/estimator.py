#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np, pandas as pd, torch, joblib
from sklearn.preprocessing import StandardScaler

from models import EnergyMLP
from features import derive_features, standardize_columns, FEAT_COLS, get_feature_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class OperatorEnsemble:
    def __init__(self, op_dir: Path):
        self.op_dir = op_dir
        self.seeds = sorted([p for p in op_dir.iterdir() if p.name.startswith("seed_") and p.suffix==".pth"])
        if not self.seeds: raise RuntimeError(f"No seed_*.pth under {op_dir}")
        self.scaler: StandardScaler = joblib.load(op_dir / "scaler.pkl")
        self.feats: List[str] = json.loads((op_dir / "feats.json").read_text())
        self.models = []
        for p in self.seeds:
            m = EnergyMLP(len(self.feats)).to(DEVICE)
            m.load_state_dict(torch.load(p, map_location=DEVICE)); m.eval(); self.models.append(m)

    @torch.inference_mode()
    def predict_array(self, X: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        Xn = self.scaler.transform(X.astype(np.float32))
        preds = []
        for m in self.models:
            log_e = m(torch.tensor(Xn, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            preds.append(10 ** log_e)
        P = np.stack(preds, axis=0)
        return P.mean(0), P.std(0)

class Predictor:
    def __init__(self, ckpt_root: str, device: str):
        self.dir = Path(ckpt_root) / device
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.ops = {}
        self.calib_path = self.dir / "calibration.json"
        self.calib = json.loads(self.calib_path.read_text()) if self.calib_path.exists() else {}

    def _get_op(self, op: str) -> OperatorEnsemble:
        if op not in self.ops: self.ops[op] = OperatorEnsemble(self.dir / op)
        return self.ops[op]

    def set_calibration(self, op: str, a: float, b: float):
        self.calib[op] = {"a": float(a), "b": float(b)}
    def save_calibration(self): self.calib_path.write_text(json.dumps(self.calib, indent=2))

    def _apply_calib(self, op: str, y: np.ndarray) -> np.ndarray:
        if op in self.calib:
            a,b = self.calib[op]["a"], self.calib[op]["b"]
            return a*y + b
        return y

    def predict_configs(self, df_cfg: pd.DataFrame) -> pd.DataFrame:
        df_cfg = derive_features(standardize_columns(df_cfg.copy()))
        rows = []
        for op, sub in df_cfg.groupby("OP", sort=False):
            ens = self._get_op(op)
            X = get_feature_matrix(sub, cols=ens.feats)
            mean, std = ens.predict_array(X)
            mean = self._apply_calib(op, mean)
            o = sub.copy(); o["pred_mean_mJ"] = mean; o["pred_std_mJ"] = std
            rows.append(o)
        return pd.concat(rows, ignore_index=True)
