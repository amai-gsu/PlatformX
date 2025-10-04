#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from estimator import Predictor
from selection import kcenter_select, select_next_uncertainty_diversity
from calibrate import fit_affine
from features import standardize_columns

def evaluate(pred, test_df):
    preds = pred.predict_configs(test_df.copy())
    out = preds.merge(test_df[["id","OP","energy_mJ"]], on=["id","OP"], how="left")
    y = out["energy_mJ"].values; yhat = out["pred_mean_mJ"].values
    mae = float(np.mean(np.abs(y-yhat))); ybar = np.mean(y)
    r2 = float(1 - np.sum((y-yhat)**2)/np.sum((y-ybar)**2))
    return mae, r2, len(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulate","select"], required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--source_device", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--test_ids_csv", default=None)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    pred = Predictor(args.ckpt_root, args.source_device)
    df = pd.read_csv(args.target_csv); df = standardize_columns(df)
    if "id" not in df.columns: raise RuntimeError("needs 'id'")
    if args.test_ids_csv:
        test_ids = set(pd.read_csv(args.test_ids_csv)["id"].tolist())
    else:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_test = max(200, int(0.2*len(df)))
        test_ids = set(df["id"].head(n_test).tolist())
    test_df = df[df["id"].isin(test_ids)].copy()
    pool_df = df[~df["id"].isin(test_ids)].copy()

    fs = kcenter_select(pool_df.copy(), k=min(args.batch, len(pool_df)))
    fs_ids = set(fs["id"].tolist())
    print(f"[Round 0] select {len(fs)} coverage.")

    if args.mode == "select":
        fs[["id","OP"]].to_csv("selected_round0.csv", index=False)
        return

    if "energy_mJ" not in df.columns: raise RuntimeError("simulate requires energy_mJ")
    # calibrate on fs
    preds = pred.predict_configs(fs.copy())
    merged = preds.merge(fs[["id","OP","energy_mJ"]], on=["id","OP"], how="inner")
    for op, sub in merged.groupby("OP"):
        a,b = fit_affine(sub["pred_mean_mJ"].values, sub["energy_mJ"].values)
        pred.set_calibration(op, a, b)
    pred.save_calibration()

    mae, r2, n = evaluate(pred, test_df.copy())
    print(f"[Round 0] MAE={mae:.4f} R2={r2:.4f} n={n}")

    measured_ids = fs["id"]
    for r in range(1, args.rounds+1):
        nxt = select_next_uncertainty_diversity(pool_df.copy(), measured_ids, pred, k=min(args.batch, len(pool_df)))
        if nxt.empty: break
        fs = pd.concat([fs, nxt], ignore_index=True)
        measured_ids = fs["id"]
        preds = pred.predict_configs(fs.copy())
        merged = preds.merge(fs[["id","OP","energy_mJ"]], on=["id","OP"], how="inner")
        for op, sub in merged.groupby("OP"):
            a,b = fit_affine(sub["pred_mean_mJ"].values, sub["energy_mJ"].values)
            pred.set_calibration(op, a, b)
        pred.save_calibration()
        mae, r2, n = evaluate(pred, test_df.copy())
        print(f"[Round {r}] MAE={mae:.4f} R2={r2:.4f} n={n}")

if __name__ == "__main__":
    main()
