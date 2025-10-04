#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from estimator import Predictor

def fit_affine(y_pred: np.ndarray, y_true: np.ndarray):
    A = np.vstack([y_pred, np.ones_like(y_pred)]).T
    a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
    return float(a), float(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--fewshot_csv", required=True)
    ap.add_argument("--id_col", default="id")
    args = ap.parse_args()

    pred = Predictor(args.ckpt_root, args.device)
    fs = pd.read_csv(args.fewshot_csv)
    preds = pred.predict_configs(fs.copy())
    merged = preds.merge(fs[[args.id_col,"OP","energy_mJ"]], on=[args.id_col,"OP"], how="inner")
    for op, sub in merged.groupby("OP"):
        a,b = fit_affine(sub["pred_mean_mJ"].values, sub["energy_mJ"].values)
        pred.set_calibration(op, a, b)
        print(f"{op}: a={a:.4f}, b={b:.4f}, n={len(sub)}")
    pred.save_calibration()
    print(pred.calib_path)

if __name__ == "__main__":
    main()
