#!/usr/bin/env python3
import argparse, json, os, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from models import EnergyMLP
from features import derive_features, standardize_columns, FEAT_COLS, get_feature_matrix

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

def build_dataset(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = False):
    X = get_feature_matrix(df)
    y = np.log10(df["energy_mJ"].astype(np.float32).values)
    if fit: scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X).astype(np.float32)
    return TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y)), scaler

def train_one(ds: TensorDataset, in_dim: int, epochs: int = 120) -> EnergyMLP:
    mdl = EnergyMLP(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), 1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    loader = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)
    best, wait = 1e9, 0
    for ep in range(epochs):
        mdl.train(); tot = 0.0
        for xb, yb in loader:
            if xb.size(0) < 2: continue
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(mdl(xb), yb); loss.backward(); opt.step()
            tot += loss.item()*len(xb)
        if tot < best-1e-4: best, wait = tot, 0
        else: wait += 1
        if wait == 25: break
    return mdl.cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--ckpt_root", default="pred_ckpts")
    ap.add_argument("--ensemble_size", type=int, default=5)
    args = ap.parse_args()

    out_root = Path(args.ckpt_root) / args.device
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = standardize_columns(df)
    df = derive_features(df)

    meta = {"ops": {}, "feats": FEAT_COLS, "ensemble_size": args.ensemble_size}
    for op_name, sub in df.groupby("OP", sort=False):
        sub = sub.dropna(subset=["energy_mJ"]).reset_index(drop=True)
        if len(sub) < 5: continue
        op_dir = out_root / op_name; op_dir.mkdir(parents=True, exist_ok=True)
        ds, scaler = build_dataset(sub, fit=True)
        joblib.dump(scaler, op_dir / "scaler.pkl")
        (op_dir / "feats.json").write_text(json.dumps(FEAT_COLS))
        saved = []
        for s in range(args.ensemble_size):
            torch.manual_seed(SEED + s)
            mdl = train_one(ds, in_dim=len(FEAT_COLS))
            p = op_dir / f"seed_{s}.pth"; torch.save(mdl.state_dict(), p); saved.append(p.name)
        meta["ops"][op_name] = {"n": int(len(sub)), "seeds": saved}
        print(f"saved ensemble for {op_name} ×{len(saved)}")
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(out_root)

if __name__ == "__main__":
    main()
