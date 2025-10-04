#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd, math

def mae(y,yhat): return float(np.mean(np.abs(y-yhat)))
def rmse(y,yhat): return float(np.sqrt(np.mean((y-yhat)**2)))
def r2(y,yhat):
    ybar = np.mean(y); ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-ybar)**2)
    return float(1 - ss_res/ss_tot) if ss_tot>0 else float("nan")
def p_abs_err(y,yhat,q): return float(np.quantile(np.abs(y-yhat), q))

def spearman(y,yhat):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(y,yhat).correlation)
    except Exception: return None
def kendall(y,yhat):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(y,yhat).correlation)
    except Exception: return None

def acc_at_k(df, true_col, pred_col, k_percent=10.0):
    n = len(df); 
    if n==0: return float("nan")
    k = max(1, int(math.ceil(n * (k_percent/100.0))))
    true_top = df.nsmallest(k, true_col).index
    pred_top = df.nsmallest(k, pred_col).index
    return float(len(set(true_top)&set(pred_top)))/float(k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["energy_mJ","pred_mean_mJ"])
    y = df["energy_mJ"].values.astype(float); yhat = df["pred_mean_mJ"].values.astype(float)
    out = {
        "count": int(len(df)),
        "MAE": mae(y,yhat),
        "RMSE": rmse(y,yhat),
        "R2": r2(y,yhat),
        "P50_abs_err": p_abs_err(y,yhat,0.5),
        "P90_abs_err": p_abs_err(y,yhat,0.9),
        "Acc@10": acc_at_k(df,"energy_mJ","pred_mean_mJ",10.0)
    }
    sp = spearman(y,yhat); kd = kendall(y,yhat)
    if sp is not None: out["Spearman"] = sp
    if kd is not None: out["Kendall"]  = kd
    with open(args.out,"w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
