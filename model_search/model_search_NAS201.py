# -*- coding: utf-8 -*-
"""
energynet_active_loop_with_metrics.py — EnergyNet 主动式搜索 + 真实测量闭环（含全面度量与报告）
- 仅针对 EnergyNet_*_*_* 的候选
- 初始 INIT_K → 每轮 NEXT_K 新测 → 增量更新
- 训练：多GPU + AMP + 早停
- 能耗：优先 Monsoon 真测；否则回退预测能耗
- 断点续跑：输出 OUT_DIR 下 CSV/图像；跳过已测模型；持续运行直到子集全部跑完
- 风险模型处理：精度过低或出现 NaN/Inf → 直接跳过并记录到 skipped_models.csv
- 报告过滤：三个“最佳”仅从 Acc_real ≥ REPORT_MIN_ACC 的样本中选出
- ★ 延迟（latency, ms）：仅用 inference_time.csv 的 duration_* 列做平均（自动单位换算）
- ★ 新增：accuracy–latency trade-off 最优模型 best_latency_tradeoff
- ★ 新增：候选排序采用 UCB（带不确定性上界） + k-center 多样性覆盖
- ★ 新增：FRESH_START 一键“从零开始”；REUSE_FROM_DIR 懒加载按需复用旧结果（不在启动时整批导入）
"""
from __future__ import annotations
import copy
import os, json, time, random, gc, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= 按工程路径调整 =========
from nas_201_api import NASBench201API as API
from model_generation.model_generater import EnergyNet

import onnx, tensorflow as tf
from onnx_tf.backend import prepare

# 可选：Monsoon 真实测量模块
try:
    from monsoon.energy_profiling import power_monitor_setup, run_inference, compute_energy
    HAVE_MONSOON = True
except Exception:
    HAVE_MONSOON = False

# =============== 全局配置 =================
CFG: Dict = {
    # 预测CSV
    "PRED_CSV": "/home/xiaolong/Downloads/GreenAuto/prediction_results/model_scores_delpoy_energy.csv",
    "NAME_COL": "Model Name",
    "SCORE_COL": "NWOT Score",
    "ENERGY_COL": "total conv energy",

    # 输出目录（本次运行）
    "OUT_DIR": "/home/xiaolong/Downloads/GreenAuto/model_search_results_global",

    # 一键“从零开始”（会清空 OUT_DIR 内的 subset/to_measure/measured 等）
    "FRESH_START": False,

    # 按需复用旧目录的测量结果（仅在被选中需要测量时才复用）
    "REUSE_FROM_DIR": "/home/xiaolong/Downloads/GreenAuto/model_search_results",

    # NB201 API
    "NB201_API_PTH": "/home/xiaolong/Downloads/GreenAuto/NAS-Bench-201-v1_0-e61699.pth",

    # 训练数据
    "DATASET": "cifar10",
    "DATA_ROOT": "/home/xiaolong/Downloads/GreenAuto/cifar10",
    "NUM_CLASSES": 10,
    "BATCH": 128,
    "EPOCHS": 12,
    "PATIENCE": 3,
    "LR": 1e-3,
    "WEIGHT_DECAY": 2e-4,
    "NUM_WORKERS": 8,

    # 设备
    "GPU_IDS": [0, 1],
    "AMP": True,

    # 采样与迭代
    "SUBSET_SIZE": 20000,
    "INIT_K": 100,
    "NEXT_K": 10,     
    "EXPLORE_FRAC": 0.25,

    # 选型权重
    "WEIGHT_ACC": 1.0,
    "WEIGHT_EN": 1.0,

    # 动态权重
    "WEIGHT_MODE": "auto-slope",  # fixed / auto-slope
    "WEIGHT_MIN": 0.5,
    "WEIGHT_MAX": 3.0,
    "AUTO_SLOPE_SMAX": 1.5,
    "AUTO_SLOPE_GAIN_LOW": 2.0,
    "AUTO_SLOPE_GAIN_HIGH": 0.7,

    # 可选固定日程
    "WEIGHT_SCHEDULE": None,

    # EnergyNet 配置空间
    "COUT_RANGE": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256],
    "KERNEL_SIZE_RANGE": [1, 3, 5, 7],
    "STRIDE_RANGE": [1, 2, 3, 4],

    # TFLite 导出
    "TFLITE_DIR": "/home/xiaolong/Downloads/GreenAuto/model_search_results/models",
    "BACKEND": "CPU",
    "PHONE_BENCHMARK_BIN_GPU": "/data/local/tmp/benchmark_model_gpu",
    "PHONE_BENCHMARK_BIN_CPU": "/data/local/tmp/benchmark_model",
    "KERNEL_CL_PATH": "/data/local/tmp/kernel.cl",
    "NUM_RUNS": 10,
    "WARMUP_RUNS": 10,

    # 度量
    "REF_ENERGY_Q": 0.95,

    # 跳过阈值
    "MIN_ACC_FOR_EXPORT": 0.12,

    # 报告门槛
    "REPORT_MIN_ACC": 0.80,

    # 预测延迟列（可选）
    "LATENCY_PRED_COL": None,

    # UCB 参数
    "UCB_ALPHA": 0.10,
    "UCB_BETA": 1.2,

    # 种子
    "SEED": 42,
}

# reproducibility + 速度
random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
torch.manual_seed(CFG["SEED"])
torch.backends.cudnn.benchmark = True

# 运行统计（全局）
RUN: Dict = {
    "start_ts": time.time(),
    "gpu_count": 0,
    "models_started": 0,
    "models_finished": 0,
    "tflite_ok": 0,
    "monsoon_ok": 0,
    "energy_pred_fallback": 0,
    "gpu_seconds": 0.0,
    "wall_seconds": 0.0,
    "iteration_stats": []
}

# =============== 基础工具 ===============
def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _coerce_numeric_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    before = len(df)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    nans = df[col].isna().sum()
    if nans:
        print(f"[clean] 列 '{col}' 非数值 → NaN: {nans}/{before}，将被丢弃")
    return df

def _clean_measured_energy(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    m["Energy_mJ_real"] = pd.to_numeric(m["Energy_mJ_real"], errors="coerce")
    # 把 <=0 的能耗视为无效（NaN），避免进入 Pareto/Trade-off/最佳选择
    m.loc[~np.isfinite(m["Energy_mJ_real"]) | (m["Energy_mJ_real"] <= 0), "Energy_mJ_real"] = np.nan
    return m

def load_pred_df() -> pd.DataFrame:
    df = pd.read_csv(CFG["PRED_CSV"], low_memory=False)
    df = df[df[CFG["NAME_COL"]].astype(str).str.startswith("EnergyNet_")].copy()
    need = [CFG["NAME_COL"], CFG["SCORE_COL"], CFG["ENERGY_COL"]]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"预测文件缺少列：{miss}")
    df = _coerce_numeric_col(df, CFG["SCORE_COL"])
    df = _coerce_numeric_col(df, CFG["ENERGY_COL"])
    if CFG.get("LATENCY_PRED_COL") and CFG["LATENCY_PRED_COL"] in df.columns:
        df = _coerce_numeric_col(df, CFG["LATENCY_PRED_COL"])
    df = df.dropna(subset=[CFG["SCORE_COL"], CFG["ENERGY_COL"]])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[CFG["SCORE_COL"], CFG["ENERGY_COL"]])
    df = df.drop_duplicates(subset=[CFG["NAME_COL"]])
    keep = [CFG["NAME_COL"], CFG["SCORE_COL"], CFG["ENERGY_COL"]]
    if CFG.get("LATENCY_PRED_COL") and CFG["LATENCY_PRED_COL"] in df.columns:
        keep.append(CFG["LATENCY_PRED_COL"])
    return df[keep].reset_index(drop=True)

def parse_energynet_name(name: str) -> Tuple[int, int]:
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"非法 EnergyNet 名称：{name}")
    nb201_id = int(parts[1]); config_id = int(parts[3])
    return nb201_id, config_id

def str2structure(arch_str: str):
    nodes = []
    for node_str in arch_str.split('+'):
        node_str = node_str.strip()
        if not node_str: continue
        edges = []
        for token in node_str.split('|'):
            token = token.strip()
            if not token: continue
            op, j = token.split('~')
            edges.append((op, int(j)))
        nodes.append(tuple(edges))
    return nodes

def generate_energy_configs(C: int, cout_range, kernel_size_range, stride_range, best_config=None):
    cfgs = []
    if best_config: cfgs.append(best_config)
    for cout in cout_range:
        for k in kernel_size_range:
            for s in stride_range:
                cfg = {'CIN': C, 'COUT': cout, 'KERNEL_SIZE': k, 'STRIDE': s, 'PADDING': 1}
                if best_config is None or cfg != best_config:
                    cfgs.append(cfg)
    return cfgs

class NB201Resolver:
    def __init__(self, api_pth: str):
        if not Path(api_pth).exists():
            raise RuntimeError(f"[nb201] 找不到 API 权重：{api_pth}")
        self.api = API(api_pth, verbose=False)
    def cfg_C_N_numcls(self, uid: int) -> Tuple[int, int, int]:
        cfg = self.api.get_net_config(uid, "cifar10-valid")
        return cfg["C"], cfg["N"], cfg["num_classes"]
    def arch_str(self, uid: int) -> str:
        arch = self.api.arch(uid)
        try: return arch.tostr()
        except Exception: return str(arch)

nb201 = NB201Resolver(CFG["NB201_API_PTH"])

def energy_config_from_id(C_in: int, config_id: int) -> Dict[str, int]:
    all_cfgs = generate_energy_configs(
        C_in, CFG["COUT_RANGE"], CFG["KERNEL_SIZE_RANGE"], CFG["STRIDE_RANGE"], best_config=None
    )
    idx = max(1, int(config_id)) - 1
    idx = min(idx, len(all_cfgs) - 1)
    return all_cfgs[idx]

def get_loaders() -> Tuple[DataLoader, DataLoader]:
    ds = CFG["DATASET"].lower()
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])
    if ds == "cifar10":
        train = datasets.CIFAR10(CFG["DATA_ROOT"], train=True, download=True, transform=tf_train)
        test  = datasets.CIFAR10(CFG["DATA_ROOT"], train=False, download=True, transform=tf_test)
        CFG["NUM_CLASSES"] = 10
    else:
        train = datasets.CIFAR100(CFG["DATA_ROOT"], train=True, download=True, transform=tf_train)
        test  = datasets.CIFAR100(CFG["DATA_ROOT"], train=False, download=True, transform=tf_test)
        CFG["NUM_CLASSES"] = 100
    train_loader = DataLoader(
        train, batch_size=CFG["BATCH"], shuffle=True,
        num_workers=CFG["NUM_WORKERS"], pin_memory=True, persistent_workers=CFG["NUM_WORKERS"]>0
    )
    test_loader  = DataLoader(
        test, batch_size=CFG["BATCH"], shuffle=False,
        num_workers=CFG["NUM_WORKERS"], pin_memory=True, persistent_workers=CFG["NUM_WORKERS"]>0
    )
    return train_loader, test_loader

# ---- 动态权重（auto-slope） ----
def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-9
    return (x - med) / mad

def _auto_slope_weights(measured: pd.DataFrame, base_acc: float, base_en: float) -> Tuple[float, float]:
    if measured is None or len(measured) < 5:
        return base_acc, base_en
    pf = pareto_front(measured, "Acc_real", "Energy_mJ_real")
    if len(pf) < 3:
        return base_acc, base_en
    E = np.log1p(pf["Energy_mJ_real"].astype(float).values)
    A = pf["Acc_real"].astype(float).values
    zE = _robust_z(E); zA = _robust_z(A)
    dE = np.diff(zE); dA = np.diff(zA)
    mask = np.abs(dE) > 1e-6
    if not np.any(mask):
        return base_acc, base_en
    slope = np.median(np.abs(dA[mask] / dE[mask]))
    s = float(np.clip(slope, 0.0, CFG["AUTO_SLOPE_SMAX"]))
    smax = CFG["AUTO_SLOPE_SMAX"]
    g_lo, g_hi = CFG["AUTO_SLOPE_GAIN_LOW"], CFG["AUTO_SLOPE_GAIN_HIGH"]
    ratio = g_lo + (g_hi - g_lo) * (s / smax)
    w_acc = base_acc
    w_en  = float(np.clip(base_en * ratio, CFG["WEIGHT_MIN"], CFG["WEIGHT_MAX"]))
    return w_acc, w_en

def get_iter_weights(measured: pd.DataFrame, iter_idx: int) -> Tuple[float, float]:
    acc_w, en_w = CFG["WEIGHT_ACC"], CFG["WEIGHT_EN"]
    schedule = CFG.get("WEIGHT_SCHEDULE")
    if schedule:
        for s in sorted(schedule, key=lambda x: x.get("from", 0)):
            if iter_idx >= int(s.get("from", 0)):
                acc_w, en_w = float(s.get("acc", acc_w)), float(s.get("en", en_w))
    mode = CFG.get("WEIGHT_MODE", "fixed").lower()
    if mode == "fixed" or iter_idx == 0:
        return acc_w, en_w
    elif mode == "auto-slope":
        return _auto_slope_weights(measured, acc_w, en_w)
    else:
        return acc_w, en_w

# ---- 分位分箱 ----
def _rank_bins(series: pd.Series, q: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    r = s.rank(pct=True, method="average")
    bins = np.floor(np.clip(r * q, 0, q - 1)).astype(int)
    return pd.Series(bins, index=series.index)

def pick_subset(df: pd.DataFrame, subset_size: int, seed: int = 42) -> pd.DataFrame:
    if subset_size <= 0 or subset_size >= len(df):
        return df.sample(frac=1.0, random_state=seed)
    df = df.copy()
    df[CFG["SCORE_COL"]] = pd.to_numeric(df[CFG["SCORE_COL"]], errors="coerce")
    df[CFG["ENERGY_COL"]] = pd.to_numeric(df[CFG["ENERGY_COL"]], errors="coerce")
    if CFG.get("LATENCY_PRED_COL") and CFG["LATENCY_PRED_COL"] in df.columns:
        df[CFG["LATENCY_PRED_COL"]] = pd.to_numeric(df[CFG["LATENCY_PRED_COL"]], errors="coerce")
    df = df.dropna(subset=[CFG["SCORE_COL"], CFG["ENERGY_COL"]])
    df["_logE"] = np.log1p(df[CFG["ENERGY_COL"]].astype(float))
    sbin = _rank_bins(df[CFG["SCORE_COL"]], 10)
    ebin = _rank_bins(df["_logE"],           10)
    df["_bin"] = sbin.astype(str) + "-" + ebin.astype(str)
    groups = df.groupby("_bin", dropna=True)
    per = max(1, subset_size // max(1, len(groups)))
    idxs = []
    for _, g in groups:
        take = min(per, len(g))
        idxs.append(g.sample(n=take, random_state=seed).index)
    chosen = np.concatenate(idxs) if idxs else df.sample(n=subset_size, random_state=seed).index
    out = df.loc[chosen].drop(columns=["_logE", "_bin"]).reset_index(drop=True)
    print(f"[data] 子集规模: {len(out)}")
    return out

# ---- Pareto & AUPEC ----
def pareto_front(df: pd.DataFrame, acc_col: str, en_col: str) -> pd.DataFrame:
    d = df.sort_values([en_col, acc_col], ascending=[True, False])
    chosen, best_acc = [], -1
    for _, r in d.iterrows():
        a = float(r[acc_col])
        if a > best_acc:
            chosen.append(r); best_acc = a
    return pd.DataFrame(chosen)

def aupec(measured: pd.DataFrame, ref_energy: float) -> float:
    if measured is None or len(measured) == 0:
        return 0.0
    pf = pareto_front(measured, "Acc_real", "Energy_mJ_real")
    if len(pf) < 2:
        return 0.0
    pf = pf.sort_values("Energy_mJ_real")
    e = pf["Energy_mJ_real"].to_numpy()
    a = pf["Acc_real"].to_numpy()
    e0 = float(np.nanmin(e))
    Eref = float(ref_energy)
    e = np.clip(e, e0, Eref)
    mask = (e >= e0) & (e <= Eref)
    e, a = e[mask], a[mask]
    if len(e) < 2 or Eref <= e0:
        return 0.0
    area = 0.0
    for i in range(len(e) - 1):
        area += (e[i+1] - e[i]) * max(0.0, float(a[i]))
    denom = (Eref - e0) * 1.0
    return float(np.clip(area / max(denom, 1e-12), 0.0, 1.0))

# =============== UCB + k-center 工具 ===============
def _winsorize(x, p=0.01):
    x = np.asarray(x, float)
    if not np.isfinite(x).any(): return x
    lo, hi = np.nanquantile(x[np.isfinite(x)], [p, 1-p])
    return np.clip(x, lo, hi)

def fit_linear_calibrator(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5: return (1.0, 0.0, np.array([0.1]))
    xw = _winsorize(x[m]); yw = _winsorize(y[m])
    a, b = np.polyfit(xw, yw, 1)
    yhat = a * xw + b
    res = np.abs(yw - yhat)
    return (a, b, res)

def calibrate_uncertainty(measured_df: pd.DataFrame, alpha=0.10):
    a1,b1,res1 = fit_linear_calibrator(measured_df["NWOT_pred"], measured_df["Acc_real"])
    d_acc = np.nanquantile(res1, 1 - alpha) if len(res1) else 0.02
    mm = measured_df
    if "Energy_src" in measured_df.columns:
        mm = measured_df[measured_df["Energy_src"]=="monsoon"]
        if len(mm) < 5:
            mm = measured_df
    # 只用有效能耗做标定
    mm = mm.copy()
    mm["Energy_mJ_real"] = pd.to_numeric(mm["Energy_mJ_real"], errors="coerce")
    mm = mm[np.isfinite(mm["Energy_mJ_real"]) & (mm["Energy_mJ_real"] > 0)]
    a2,b2,res2 = fit_linear_calibrator(mm["Energy_pred"], mm["Energy_mJ_real"])
    if len(res2):
        d_E = np.nanquantile(res2, 1 - alpha)
    else:
        base = float(measured_df["Energy_mJ_real"].median()) if len(measured_df) else 1.0
        d_E = 0.05 * max(1.0, base)
    return {"acc_ab": (a1,b1), "acc_band": float(d_acc),
            "E_ab":   (a2,b2), "E_band":  float(d_E)}

def predict_with_bands(nwot_pred, E_pred, cal):
    a1,b1 = cal["acc_ab"]; a2,b2 = cal["E_ab"]
    acc_hat = a1 * float(nwot_pred) + b1
    E_hat   = a2 * float(E_pred)     + b2
    return acc_hat, E_hat, cal["acc_band"], cal["E_band"]

def ucb_score_row(nwot_val, e_pred_val, cal, w_acc, w_en, beta=1.2, Emax=1.0):
    acc_hat, E_hat, da, dE = predict_with_bands(nwot_val, e_pred_val, cal)
    acc_ucb = acc_hat + beta * da
    E_lcb   = max(0.0, E_hat - beta * dE)
    return w_acc * acc_ucb - w_en * (E_lcb / max(1e-9, Emax))

def kcenter_diverse_pick(feat_mat: np.ndarray, names: List[str], k: int, seed=42) -> List[str]:
    if k <= 0 or len(names) == 0: return []
    rng = np.random.default_rng(seed)
    n = feat_mat.shape[0]
    idx = [int(rng.integers(0, n))]
    dist = np.full(n, np.inf)
    for _ in range(1, min(k, n)):
        last = feat_mat[idx[-1]][None, :]
        dist = np.minimum(dist, np.linalg.norm(feat_mat - last, axis=1))
        cand = int(np.argmax(dist))
        idx.append(cand)
    return [names[i] for i in idx[:k]]

# =============== 候选排序（UCB + k-center） ===============
def rank_candidates(pred_sub: pd.DataFrame,
                    measured: pd.DataFrame,
                    k: int,
                    explore_frac: float = 0.25,
                    iter_idx: int = 0) -> List[str]:
    already = set(measured["Model Name"].tolist()) if len(measured) else set()
    pool = pred_sub[~pred_sub[CFG["NAME_COL"]].isin(already)].copy()
    if len(pool) <= k:
        return pool[CFG["NAME_COL"]].tolist()

    w_acc, w_en = get_iter_weights(measured, iter_idx)

    use_ucb = (len(measured) >= 10) and all(
        c in measured.columns for c in ["NWOT_pred", "Acc_real", "Energy_pred", "Energy_mJ_real"]
    )

    top_exploit: List[str] = []
    if use_ucb:
        cal = calibrate_uncertainty(measured, alpha=CFG.get("UCB_ALPHA", 0.10))
        Emax = float(pred_sub[CFG["ENERGY_COL"]].astype(float).max())
        pool["_ucb"] = pool.apply(
            lambda r: ucb_score_row(
                r[CFG["SCORE_COL"]],
                r[CFG["ENERGY_COL"]],
                cal, w_acc, w_en,
                beta=CFG.get("UCB_BETA", 1.2), Emax=Emax
            ),
            axis=1
        )
        k_exp = max(1, int(k * (1.0 - explore_frac)))
        top_exploit = pool.nlargest(k_exp, "_ucb")[CFG["NAME_COL"]].tolist()
    else:
        pool["_logE"] = np.log1p(pool[CFG["ENERGY_COL"]])
        S = pd.to_numeric(pool[CFG["SCORE_COL"]], errors="coerce")
        E = pd.to_numeric(pool["_logE"], errors="coerce")
        zS = (S - S.mean()) / (S.std() + 1e-9)
        zE = (E - E.mean()) / (E.std() + 1e-9)
        pool["_exploit"] = w_acc * zS + w_en * (-zE)
        k_exp = max(1, int(k * (1.0 - explore_frac)))
        top_exploit = pool.nlargest(k_exp, "_exploit")[CFG["NAME_COL"]].tolist()

    rest = pool[~pool[CFG["NAME_COL"]].isin(set(top_exploit))].copy()
    k_explore = max(0, k - len(top_exploit))

    if k_explore > 0 and len(rest) > 0:
        feat = np.stack([
            _robust_z(pd.to_numeric(rest[CFG["SCORE_COL"]], errors="coerce").astype(float).values),
            _robust_z(np.log1p(pd.to_numeric(rest[CFG["ENERGY_COL"]], errors="coerce").astype(float).values)),
        ], axis=1)
        names_rest = rest[CFG["NAME_COL"]].tolist()
        top_explore = kcenter_diverse_pick(feat, names_rest, k_explore, seed=CFG["SEED"])
    else:
        top_explore = []

    names = top_exploit + top_explore
    if len(names) < k:
        more = pool[~pool[CFG["NAME_COL"]].isin(set(names))][CFG["NAME_COL"]].tolist()
        names += more[:k - len(names)]
    return names[:k]

# =============== 跳过记录辅助 ===============
def _log_skipped(name: str, it: Optional[int], reason: str, acc: Optional[float] = None):
    outdir = Path(CFG["OUT_DIR"]); _ensure(outdir)
    f = outdir / "skipped_models.csv"
    row = {"Model Name": name, "iter": (None if it is None else int(it)),
           "reason": str(reason), "acc": (None if acc is None else float(acc)),
           "ts": time.time()}
    if f.exists():
        pd.DataFrame([row]).to_csv(f, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(f, index=False)

def _model_has_nonfinite(model: nn.Module) -> bool:
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
        for b in model.buffers():
            if b.numel() and not torch.isfinite(b).all():
                return True
    return False

# =============== 构建 & 训练 & 能耗/延迟测量 ===============
def build_energynet_from_name(name: str) -> nn.Module:
    uid, cfg_id = parse_energynet_name(name)
    arch_str = nb201.arch_str(uid)
    genotype = str2structure(arch_str)
    C_in, N, _ = nb201.cfg_C_N_numcls(uid)
    cfg = energy_config_from_id(C_in, cfg_id)
    net = EnergyNet(N, genotype, CFG["NUM_CLASSES"], batch_size=CFG["BATCH"], config=cfg)
    return net

def train_and_eval(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> float:
    device_ids = CFG["GPU_IDS"]
    n_visible = torch.cuda.device_count()
    use_dp = torch.cuda.is_available() and len(device_ids) > 1 and n_visible >= len(device_ids)
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if use_dp:
        model = nn.DataParallel(model, device_ids=device_ids)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["EPOCHS"])
    scaler = GradScaler(enabled=CFG["AMP"])

    best_acc, bad = 0.0, 0
    t0 = time.time()
    for ep in range(CFG["EPOCHS"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=CFG["AMP"]):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        model.eval()
        tot, cor = 0, 0
        with torch.no_grad(), autocast(enabled=CFG["AMP"]):
            for xb, yb in test_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb); pred = logits.argmax(1)
                cor += (pred == yb).sum().item(); tot += yb.numel()
        acc = cor / max(1, tot)
        if acc > best_acc: best_acc, bad = acc, 0
        else: bad += 1
        print(f"[train] ep {ep+1}/{CFG['EPOCHS']} acc={acc:.4f} best={best_acc:.4f}")
        if bad >= CFG["PATIENCE"]:
            print("[early-stop] no improvement"); break

    wall = time.time() - t0
    used_gpus = (len(device_ids) if torch.cuda.is_available() and use_dp else
                 (1 if torch.cuda.is_available() else 0))
    RUN["gpu_seconds"] += wall * used_gpus
    RUN["wall_seconds"] += wall

    if isinstance(model, nn.DataParallel): model = model.module
    torch.cuda.empty_cache(); gc.collect()
    return float(best_acc)

# =============== TFLite 导出（AvgPool 兼容补丁） ===============
def clone_with_tf_friendly_avgpool(model: nn.Module) -> nn.Module:
    m2 = copy.deepcopy(model)
    def _patch(m: nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.AvgPool2d):
                new_pool = nn.AvgPool2d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    ceil_mode=False,
                    count_include_pad=True
                )
                setattr(m, name, new_pool)
            else:
                _patch(child)
    _patch(m2)
    return m2

def export_tflite(model: nn.Module, name: str, out_dir: str, input_size=(1,3,32,32)) -> str:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tfl_path = out_dir / f"{name}.tflite"

    export_model = clone_with_tf_friendly_avgpool(model).cpu().eval()
    export_model = export_model.float()
    dummy = torch.randn(*input_size)

    last_err = None
    for opset in (11, 12, 13):
        tmp_root = Path(tempfile.mkdtemp(prefix=f"onnx_exp_{name}_"))
        onnx_path = tmp_root / f"{name}_op{opset}.onnx"
        try:
            torch.onnx.export(
                export_model, dummy, str(onnx_path),
                opset_version=opset, do_constant_folding=True,
                input_names=["input"], output_names=["logits"],
                keep_initializers_as_inputs=False,
            )
            onnx_model = onnx.load(str(onnx_path))
            tf_dir = tmp_root / "saved_model"
            tf_rep = prepare(onnx_model, strict=False)
            tf_rep.export_graph(str(tf_dir))
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tfl_bytes = converter.convert()
            with open(tfl_path, "wb") as f:
                f.write(tfl_bytes)
            shutil.rmtree(tmp_root, ignore_errors=True)
            RUN["tflite_ok"] += 1
            return str(tfl_path)
        except Exception as e:
            last_err = e
            print(f"[tflite] opset {opset} 转换失败：{e}")
            shutil.rmtree(tmp_root, ignore_errors=True)
            continue
    raise RuntimeError(f"导出 TFLite 失败（已尝试 opset 11/12/13）。最后错误：{last_err}")

def ensure_tflite(model: nn.Module, name: str) -> Optional[str]:
    out = Path(CFG["TFLITE_DIR"]) / f"{name}.tflite"
    if out.exists():
        return str(out)
    try:
        return export_tflite(model, name, CFG["TFLITE_DIR"], input_size=(1,3,32,32))
    except Exception as e:
        print(f"[tflite] 导出失败：{e}")
        return None

# =============== Monsoon 真实测量（仅用 duration 平均） ===============
def _parse_latency_from_infer_csv(infer_csv: str) -> Optional[float]:
    try:
        df = pd.read_csv(infer_csv)
    except Exception:
        return None
    if df.empty: return None
    df.columns = [str(c).strip().lower() for c in df.columns]
    candidates = [
        ("duration_ms", "ms"), ("latency_ms", "ms"), ("elapsed_ms", "ms"),
        ("duration_us", "us"), ("latency_us", "us"), ("elapsed_us", "us"),
        ("duration_ns", "ns"), ("latency_ns", "ns"), ("elapsed_ns", "ns"),
        ("duration_s",  "s"),  ("latency_s",  "s"),  ("elapsed_s",  "s"),
    ]
    for col, unit in candidates:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            v = v[(v > 0) & np.isfinite(v)]
            if len(v) == 0: return None
            if unit == "ms": ms = v
            elif unit == "us": ms = v / 1e3
            elif unit == "ns": ms = v / 1e6
            elif unit == "s":  ms = v * 1e3
            else: ms = v
            return float(ms.mean())
    return None

def measure_energy_and_latency_with_monsoon(tflite_path: str) -> Tuple[Optional[float], Optional[float]]:
    if not HAVE_MONSOON:
        return None, None
    try:
        from Monsoon import pmapi
        HVMON, HVengine = power_monitor_setup(26589, pmapi.USB_protocol())
    except Exception as e:
        print(f"[monsoon] 初始化失败：{e}")
        return None, None

    raw_dir = Path(CFG["OUT_DIR"]) / "monsoon_raw"; _ensure(raw_dir)
    name = Path(tflite_path).stem
    power_csv = str(raw_dir / f"{name}_power.csv")

    ok = False
    try:
        ok = run_inference(
            HVMON, HVengine,
            model_path=tflite_path,
            power_csv=power_csv,
            benchmark_bin=(CFG["PHONE_BENCHMARK_BIN_GPU"] if CFG["BACKEND"].upper()=="GPU" else CFG["PHONE_BENCHMARK_BIN_CPU"]),
            num_runs=CFG["NUM_RUNS"],
            warmup_runs=CFG["WARMUP_RUNS"],
            backend=CFG["BACKEND"],
            kernel_path=CFG["KERNEL_CL_PATH"]
        )
    except Exception as e:
        print(f"[monsoon] run_inference 异常：{e}")
        ok = False

    if not ok:
        return None, None

    infer_csv = str(raw_dir / f"{name}_inference_time.csv")
    latency_ms = None
    if Path(infer_csv).exists():
        latency_ms = _parse_latency_from_infer_csv(infer_csv)
    else:
        print("[monsoon] 未解析到 inference timestamps；将只尝试能耗计算。")

    energy_mJ = None
    try:
        if Path(power_csv).exists() and Path(infer_csv).exists():
            energy_mJ = compute_energy(power_csv, infer_csv)
            RUN["monsoon_ok"] += 1
    except Exception as e:
        print(f"[monsoon] 计算能耗失败：{e}")
        energy_mJ = None

    return (None if energy_mJ is None else float(energy_mJ),
            None if latency_ms is None else float(latency_ms))

# =============== 旧结果懒加载索引（按需复用） ===============
def _build_old_results_cache(reuse_dir: Optional[str],
                             pred_sub: pd.DataFrame,
                             measured_cols: List[str]) -> Dict[str, dict]:
    """
    构建一个 name->row 的缓存，但**不写入**当前 measured。
    只有当某个候选被选中、且当前 measured 没有它时，才把缓存里的行写进去。
    - 仅保留与本次子集重叠的模型
    - NWOT_pred/Energy_pred 用本次 pred_sub 的值补齐
    """
    cache: Dict[str, dict] = {}
    if not reuse_dir:
        return cache
    old_csv = Path(reuse_dir) / "measured_energy_real.csv"
    if not old_csv.exists():
        print(f"[reuse] 未找到旧测量文件：{old_csv}")
        return cache
    try:
        df_old = pd.read_csv(old_csv)
    except Exception as e:
        print(f"[reuse] 读取旧测量失败：{e}")
        return cache
    if df_old.empty:
        return cache

    valid = set(pred_sub[CFG["NAME_COL"]].astype(str).tolist())
    df_old = df_old[df_old["Model Name"].astype(str).isin(valid)].copy()
    if df_old.empty:
        print("[reuse] 旧测量里没有与当前子集重叠的模型。")
        return cache

    # 用当前预测表补 NWOT_pred/Energy_pred
    aux = pred_sub[[CFG["NAME_COL"], CFG["SCORE_COL"], CFG["ENERGY_COL"]]].rename(columns={
        CFG["NAME_COL"]: "Model Name",
        CFG["SCORE_COL"]: "NWOT_pred_new",
        CFG["ENERGY_COL"]: "Energy_pred_new",
    })
    df_old = df_old.merge(aux, on="Model Name", how="left")
    if "NWOT_pred" not in df_old.columns: df_old["NWOT_pred"] = np.nan
    if "Energy_pred" not in df_old.columns: df_old["Energy_pred"] = np.nan
    df_old["NWOT_pred"]  = df_old["NWOT_pred"].fillna(df_old["NWOT_pred_new"])
    df_old["Energy_pred"] = df_old["Energy_pred"].fillna(df_old["Energy_pred_new"])
    df_old = df_old.drop(columns=["NWOT_pred_new", "Energy_pred_new"], errors="ignore")

    # 补齐缺失列
    for c in measured_cols:
        if c not in df_old.columns:
            df_old[c] = np.nan

    # 构建缓存：不改 iter（等写入时再设为当前 iter）
    df_old = df_old.drop_duplicates(subset=["Model Name"], keep="last").reset_index(drop=True)
    for _, r in df_old.iterrows():
        cache[str(r["Model Name"])] = {c: r.get(c, np.nan) for c in measured_cols}
    print(f"[reuse] 旧结果缓存就绪：{len(cache)} 条（按需复用）")
    return cache

# =============== 可视化 & 报告 ===============
def plot_scatter(measured: pd.DataFrame, it: int, outdir: Path):
    plt.figure(figsize=(7.2, 5.0))
    if len(measured):
        plt.scatter(measured["Energy_mJ_real"], measured["Acc_real"], s=10, alpha=0.7, label="measured")
        pf = pareto_front(measured, "Acc_real", "Energy_mJ_real")
        if len(pf):
            plt.plot(pf["Energy_mJ_real"], pf["Acc_real"], "r-o", ms=3, lw=1.2, label="Pareto")
        plt.legend()
    plt.xlabel("Energy (mJ) ↓"); plt.ylabel("Accuracy ↑")
    plt.title(f"EnergyNet measured (iter {it:02d})")
    _ensure(outdir); png = outdir / f"scatter_iter_{it:02d}.png"
    plt.tight_layout(); plt.savefig(png, dpi=160); plt.close()
    print(f"[plot] {png}")

def best_models_by_objectives(measured: pd.DataFrame) -> Dict[str, Dict]:
    measured = _clean_measured_energy(measured)
    out = {"best_acc": None, "best_energy": None, "best_latency": None, "best_tradeoff": None, "best_latency_tradeoff": None}
    if not len(measured): return out
    th = float(CFG.get("REPORT_MIN_ACC", 0.0))
    mm = measured.copy()
    m_ok = mm[pd.to_numeric(mm["Acc_real"], errors="coerce").astype(float) >= th].copy()
    if len(m_ok) == 0: return out

    bacc = m_ok.sort_values(["Acc_real", "Energy_mJ_real"], ascending=[False, True]).iloc[0]
    out["best_acc"] = {"name": bacc["Model Name"], "acc": float(bacc["Acc_real"]), "energy": float(bacc["Energy_mJ_real"]),
                       "latency": (None if pd.isna(bacc.get("Latency_ms_real", np.nan)) else float(bacc["Latency_ms_real"]))}

    beng = m_ok.sort_values(["Energy_mJ_real", "Acc_real"], ascending=[True, False]).iloc[0]
    out["best_energy"] = {"name": beng["Model Name"], "acc": float(beng["Acc_real"]), "energy": float(beng["Energy_mJ_real"]),
                          "latency": (None if pd.isna(beng.get("Latency_ms_real", np.nan)) else float(beng["Latency_ms_real"]))}

    has_lat = m_ok.dropna(subset=["Latency_ms_real"])
    if len(has_lat) > 0:
        blat = has_lat.sort_values(["Latency_ms_real", "Acc_real"], ascending=[True, False]).iloc[0]
        out["best_latency"] = {"name": blat["Model Name"], "acc": float(blat["Acc_real"]),
                               "energy": float(blat["Energy_mJ_real"]), "latency": float(blat["Latency_ms_real"])}

    mt = m_ok.copy()
    emax = max(1e-9, float(mt["Energy_mJ_real"].max()))
    mt["acc_n"] = mt["Acc_real"]; mt["eng_n"] = mt["Energy_mJ_real"] / emax
    w_acc_e, w_en = get_iter_weights(measured, len(mt))
    mt["trade"] = w_acc_e * mt["acc_n"] - w_en * mt["eng_n"]
    bt = mt.sort_values(["trade", "Acc_real"], ascending=[False, False]).iloc[0]
    out["best_tradeoff"] = {"name": bt["Model Name"], "acc": float(bt["Acc_real"]), "energy": float(bt["Energy_mJ_real"]),
                            "latency": (None if pd.isna(bt.get("Latency_ms_real", np.nan)) else float(bt["Latency_ms_real"])),
                            "score": float(bt["trade"]), "w_acc": w_acc_e, "w_en": w_en}

    if len(has_lat) > 0:
        mtl = has_lat.copy()
        lmax = max(1e-9, float(mtl["Latency_ms_real"].max()))
        mtl["acc_n"] = mtl["Acc_real"]; mtl["lat_n"] = mtl["Latency_ms_real"] / lmax
        w_acc_l, w_lat = get_iter_weights(measured, len(mtl))
        mtl["trade_lat"] = w_acc_l * mtl["acc_n"] - w_lat * mtl["lat_n"]
        btl = mtl.sort_values(["trade_lat", "Acc_real", "Latency_ms_real"], ascending=[False, False, True]).iloc[0]
        out["best_latency_tradeoff"] = {"name": btl["Model Name"], "acc": float(btl["Acc_real"]),
                                        "energy": float(btl["Energy_mJ_real"]), "latency": float(btl["Latency_ms_real"]),
                                        "score": float(btl["trade_lat"]), "w_acc": w_acc_l, "w_lat": w_lat}
    return out

def corr_and_error_metrics(measured: pd.DataFrame) -> Dict[str, float]:
    measured = _clean_measured_energy(measured)
    out = {}
    if not len(measured): return out
    pairs = measured[["NWOT_pred", "Acc_real"]].dropna()
    if len(pairs) >= 3:
        out["pearson_acc"] = float(np.corrcoef(pairs["NWOT_pred"].values, pairs["Acc_real"].values)[0,1])
    src = measured.get("Energy_src", pd.Series(index=measured.index, data="pred"))
    m_real = measured[src == "monsoon"].copy()
    m_real = m_real.dropna(subset=["Energy_mJ_real"])
    if len(m_real) >= 3:
        y = m_real["Energy_mJ_real"].astype(float).values
        yhat = m_real["Energy_pred"].astype(float).values
        mae = float(np.mean(np.abs(y - yhat)))
        wape = float(np.sum(np.abs(y - yhat)) / max(1e-9, np.sum(np.abs(y))))
        mape = float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), 1e-9, None))))
        out.update({"energy_mae": mae, "energy_wape": wape, "energy_mape": mape})
        out["real_energy_samples"] = int(len(m_real))
    out["measured_count"] = int(len(measured))
    out["monsoon_ratio"] = float((src == "monsoon").mean())
    out["latency_available"] = int(measured["Latency_ms_real"].notna().sum())
    return out

def save_iteration_report(it: int, measured: pd.DataFrame, pred_sub: pd.DataFrame, outdir: Path):
    _ensure(outdir)
    refE = float(np.quantile(pred_sub[CFG["ENERGY_COL"]].astype(float), CFG["REF_ENERGY_Q"]))
    au = aupec(measured, refE)
    bests = best_models_by_objectives(measured)
    mets = corr_and_error_metrics(measured)
    w_acc_iter, w_en_iter = get_iter_weights(measured, it)

    ba = bests.get("best_acc") or {}
    be = bests.get("best_energy") or {}
    bl = bests.get("best_latency") or {}
    bt = bests.get("best_tradeoff") or {}
    btl = bests.get("best_latency_tradeoff") or {}

    row = {
        "iter": it,
        "measured_cum": int(len(measured)),
        "measured_new": int((measured["iter"] == it).sum()),
        "aupec@q{:.2f}".format(CFG["REF_ENERGY_Q"]): au,
        "best_acc": ba.get("acc"),
        "best_acc_energy": ba.get("energy"),
        "best_acc_latency": ba.get("latency"),
        "best_energy": be.get("energy"),
        "best_energy_acc": be.get("acc"),
        "best_energy_latency": be.get("latency"),
        "best_latency": bl.get("latency"),
        "best_latency_acc": bl.get("acc"),
        "best_latency_energy": bl.get("energy"),
        "best_tradeoff_acc": bt.get("acc"),
        "best_tradeoff_energy": bt.get("energy"),
        "best_tradeoff_latency": bt.get("latency"),
        "best_tradeoff_score": bt.get("score"),
        "best_lat_tradeoff_acc": btl.get("acc"),
        "best_lat_tradeoff_latency": btl.get("latency"),
        "best_lat_tradeoff_energy": btl.get("energy"),
        "best_lat_tradeoff_score": btl.get("score"),
        "w_acc_used": w_acc_iter,
        "w_en_used": w_en_iter,
    }
    row.update(mets)
    RUN["iteration_stats"].append(row)

    log_csv = outdir / "search_log.csv"
    df_log = pd.DataFrame([row])
    if log_csv.exists():
        df_old = pd.read_csv(log_csv); df_all = pd.concat([df_old, df_log], ignore_index=True)
    else:
        df_all = df_log
    df_all.to_csv(log_csv, index=False)

    best_csv = outdir / "best_models.csv"
    recs = []
    for k in ("best_acc", "best_energy", "best_latency", "best_tradeoff", "best_latency_tradeoff"):
        if bests.get(k):
            rec = {"iter": it, "which": k, "name": bests[k]["name"],
                   "acc": bests[k]["acc"], "energy": bests[k]["energy"]}
            if "latency" in bests[k]: rec["latency"] = bests[k]["latency"]
            if "score" in bests[k]:   rec["score"]   = bests[k]["score"]
            recs.append(rec)
    if recs:
        df_b = pd.DataFrame(recs)
        if best_csv.exists():
            old = pd.read_csv(best_csv); df_b = pd.concat([old, df_b], ignore_index=True)
        df_b.to_csv(best_csv, index=False)

    with open(outdir / f"metrics_iter_{it:02d}.json", "w") as f:
        json.dump({"iter": it, "aupec": au, "bests": bests, "metrics": mets,
                   "run": {"gpu_hours_cum": RUN["gpu_seconds"]/3600.0,
                           "wall_hours_cum": RUN["wall_seconds"]/3600.0,
                           "models_started": RUN["models_started"],
                           "models_finished": RUN["models_finished"],
                           "tflite_ok": RUN["tflite_ok"],
                           "monsoon_ok": RUN["monsoon_ok"],
                           "energy_pred_fallback": RUN["energy_pred_fallback"]}}, f, indent=2)

# =============== 续跑辅助 ===============
def _load_subset_or_build(outdir: Path, pred_all: pd.DataFrame) -> pd.DataFrame:
    p = outdir / "subset_energy_only.csv"
    if p.exists():
        df = pd.read_csv(p)
        print(f"[resume] 载入已有子集：{p}（{len(df)} 条）")
        return df
    df = pick_subset(pred_all, CFG["SUBSET_SIZE"], seed=CFG["SEED"])
    df.to_csv(p, index=False)
    return df

def _load_to_measure(it: int, to_measure_dir: Path) -> Optional[List[str]]:
    f = to_measure_dir / f"iter_{it:02d}.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            if "Model Name" in df.columns:
                return df["Model Name"].astype(str).tolist()
        except Exception:
            pass
    return None

def _pending_in_iter(it: int, measured: pd.DataFrame, to_measure_dir: Path) -> List[str]:
    cand = _load_to_measure(it, to_measure_dir)
    if not cand: return []
    done = set(measured["Model Name"].tolist()) if len(measured) else set()
    return [n for n in cand if n not in done]

def _compute_resume_iter(measured: pd.DataFrame, to_measure_dir: Path) -> Tuple[int, List[str]]:
    if len(measured) == 0:
        pending0 = _pending_in_iter(0, measured, to_measure_dir)
        return (0, pending0) if pending0 else (0, [])
    last_it = int(measured["iter"].max())
    pend = _pending_in_iter(last_it, measured, to_measure_dir)
    if pend:
        return last_it, pend
    return last_it + 1, []

def _get_or_make_candidates(it: int, pred_sub: pd.DataFrame, measured: pd.DataFrame,
                            to_measure_dir: Path, k: int, explore_frac: float) -> List[str]:
    cand = _load_to_measure(it, to_measure_dir)
    if cand is not None:
        print(f"[resume] 使用已存在的候选清单：to_measure/iter_{it:02d}.csv（{len(cand)} 个）")
        return cand
    cand = rank_candidates(pred_sub, measured, k=k, explore_frac=explore_frac, iter_idx=it)
    pd.DataFrame({"Model Name": cand}).to_csv(to_measure_dir / f"iter_{it:02d}.csv", index=False)
    print(f"[select] 生成候选清单：to_measure/iter_{it:02d}.csv（{len(cand)} 个）")
    return cand

def build_train_measure(name: str,
                        pred_row: pd.Series,
                        train_loader: DataLoader,
                        test_loader: DataLoader,
                        iter_idx: Optional[int] = None) -> Tuple[Optional[float], Optional[float], str, Optional[float], str]:
    """
    构建→训练→能耗/延迟测量。
    返回 (acc, energy_mJ, energy_source, latency_ms, latency_source)。
    任一为 None 表示跳过该模型。
    依赖：
      - build_energynet_from_name
      - train_and_eval
      - ensure_tflite
      - measure_energy_and_latency_with_monsoon
      - CFG / RUN / _model_has_nonfinite / _log_skipped
    """
    # 1) 构建
    try:
        model = build_energynet_from_name(name)
    except Exception as e:
        print(f"[build] {name} 失败：{e}")
        _log_skipped(name, iter_idx, f"build_fail:{e}", None)
        return None, None, "build_fail", None, "none"

    # 2) 训练评估
    RUN["models_started"] += 1
    acc = train_and_eval(model, train_loader, test_loader)
    RUN["models_finished"] += 1

    # 3) 风险样本直接跳过并记录
    bad = (not np.isfinite(acc)) or (acc <= float(CFG.get("MIN_ACC_FOR_EXPORT", 0.0))) or _model_has_nonfinite(model)
    if bad:
        reason = []
        if not np.isfinite(acc): reason.append("nonfinite_acc")
        if acc <= float(CFG.get("MIN_ACC_FOR_EXPORT", 0.0)): reason.append("low_acc")
        if _model_has_nonfinite(model): reason.append("nonfinite_weights")
        r = ",".join(reason) if reason else "unstable"
        print(f"[skip] {name} 因 {r} 被跳过（acc={acc:.4f}）")
        _log_skipped(name, iter_idx, r, acc)
        return None, None, "skip", None, "none"

    # 4) 能耗/延迟：优先 Monsoon，失败则回退到预测
    energy, e_src = None, "pred"
    latency, l_src = None, "none"

    tfl = ensure_tflite(model, name)
    if tfl is not None:
        e_mJ, lat_ms = measure_energy_and_latency_with_monsoon(tfl)
        if e_mJ is not None:
            energy, e_src = e_mJ, "monsoon"
        if lat_ms is not None:
            latency, l_src = lat_ms, "monsoon"

    if energy is None:
        # 用预测能耗回退
        energy = float(pred_row[CFG["ENERGY_COL"]])
        RUN["energy_pred_fallback"] += 1
        print(f"[energy] 使用预测能耗回退：{energy:.3f} mJ")

    # 可选：延迟预测列回退
    lat_pred_col = CFG.get("LATENCY_PRED_COL")
    if latency is None and lat_pred_col and (lat_pred_col in pred_row.index):
        val = pd.to_numeric(pd.Series([pred_row[lat_pred_col]]), errors="coerce").iloc[0]
        if np.isfinite(val):
            latency, l_src = float(val), "pred"

    return float(acc), float(energy), e_src, (None if latency is None else float(latency)), l_src

# =============== 主循环 ===============
def main():
    outdir = Path(CFG["OUT_DIR"]); _ensure(outdir)
    to_measure_dir = outdir / "to_measure"; _ensure(to_measure_dir)
    models_dir = Path(CFG["TFLITE_DIR"]); _ensure(models_dir)

    RUN["gpu_count"] = torch.cuda.device_count()

    # Fresh start
    if CFG.get("FRESH_START", False):
        subset_csv = outdir / "subset_energy_only.csv"
        if subset_csv.exists():
            subset_csv.unlink(); print(f"[fresh] 删除旧子集：{subset_csv}")
        if to_measure_dir.exists():
            for f in to_measure_dir.glob("iter_*.csv"):
                f.unlink()
            print(f"[fresh] 清空候选清单：{to_measure_dir}")
        measured_csv_fs = outdir / "measured_energy_real.csv"
        if measured_csv_fs.exists():
            measured_csv_fs.unlink(); print(f"[fresh] 删除旧 measured：{measured_csv_fs}")
        # 注意：即使 FRESH_START=True，我们仍然可以构建旧结果缓存（按需复用），
        # 如果你想完全不复用，把 REUSE_FROM_DIR 设为 None
        print("[fresh] 完全从零开始；如仍想按需复用旧结果，请保留 REUSE_FROM_DIR；否则设为 None。")

    pred_all = load_pred_df()
    pred_sub = _load_subset_or_build(outdir, pred_all)
    print(f"[data] EnergyNet 子集规模: {len(pred_sub)}")

    measured_csv = outdir / "measured_energy_real.csv"
    measured_cols = ["Model Name","Acc_real","Energy_mJ_real","iter","NWOT_pred","Energy_pred","Energy_src",
                     "Latency_ms_real","Latency_pred","Latency_src"]
    measured = pd.read_csv(measured_csv) if measured_csv.exists() else pd.DataFrame(columns=measured_cols)
    if len(measured):
        measured = measured.drop_duplicates(subset=["Model Name"], keep="last").reset_index(drop=True)
        for c in measured_cols:
            if c not in measured.columns:
                measured[c] = np.nan

    # 构建“旧结果缓存”（不批量导入，按需复用）
    old_cache = _build_old_results_cache(CFG.get("REUSE_FROM_DIR"), pred_sub, measured_cols)

    start_it, pending = _compute_resume_iter(measured, to_measure_dir)
    print(f"[resume] 将从迭代 {start_it:02d} 开始继续；该轮未完成条目：{len(pending)} 个")

    train_loader, test_loader = get_loaders()

    it = start_it
    while True:
        remaining_set = set(pred_sub[CFG["NAME_COL"]].tolist()) - set(measured["Model Name"].tolist())
        remaining = len(remaining_set)
        if remaining <= 0:
            print("[complete] 子集所有模型均已测完。")
            break

        if it == 0:
            k = CFG["INIT_K"]; explore = 0.5
        else:
            k = CFG["NEXT_K"]; explore = CFG["EXPLORE_FRAC"]

        if it == start_it and len(pending) > 0:
            cand = pending
            print(f"[resume] 继续完成迭代 {it:02d} 的剩余 {len(cand)} 个候选。")
        else:
            cand = _get_or_make_candidates(it, pred_sub, measured, to_measure_dir, k, explore)

        cand = [n for n in cand if (n in remaining_set)]

        if len(cand) == 0 and remaining > 0:
            cand = rank_candidates(pred_sub, measured, k=k, explore_frac=explore, iter_idx=it)
            cand = [n for n in cand if (n in remaining_set)]
            if len(cand) > 0:
                pd.DataFrame({"Model Name": cand}).to_csv(to_measure_dir / f"iter_{it:02d}.csv", index=False)
                print(f"[select] 重新生成候选清单：to_measure/iter_{it:02d}.csv（{len(cand)} 个）")

        done_set = set(measured["Model Name"].tolist())
        for nm in cand:
            if nm in done_set:
                continue

            # —— 懒加载复用旧结果：若旧缓存有该模型，则直接写入 measured 并跳过实际训练/真测 —— #
            if nm in old_cache:
                prev = dict(old_cache[nm])  # 浅拷贝
                # 用当前预测表的值覆盖 NWOT_pred / Energy_pred（更一致）
                row_list = pred_sub.loc[pred_sub[CFG["NAME_COL"]] == nm]
                if len(row_list) > 0:
                    row = row_list.iloc[0]
                    prev["NWOT_pred"]  = float(row[CFG["SCORE_COL"]])
                    prev["Energy_pred"] = float(row[CFG["ENERGY_COL"]])
                # 本次行为属于当前轮次
                prev["iter"] = it
                # 确保必要字段是 float / nan 友好
                for kf in ["Acc_real","Energy_mJ_real","Latency_ms_real","Latency_pred"]:
                    if prev.get(kf, None) is not None and not pd.isna(prev[kf]):
                        prev[kf] = float(prev[kf])
                measured = pd.concat([measured, pd.DataFrame([prev])], ignore_index=True)
                measured.to_csv(measured_csv, index=False)
                print(f"[reuse] 本轮直接复用旧结果：{nm}")
                continue
            # —— /懒加载复用旧结果 —— #

            row_list = pred_sub.loc[pred_sub[CFG["NAME_COL"]] == nm]
            if len(row_list) == 0:
                print(f"[warn] 候选不在子集内：{nm}，跳过")
                continue
            row = row_list.iloc[0]

            # 正常流程：训练 & 测量
            acc, energy, esrc, latency, lsrc = build_train_measure(nm, row, train_loader, test_loader, iter_idx=it)
            if acc is None or energy is None:
                continue

            rec = {
                "Model Name": nm,
                "Acc_real": acc,
                "Energy_mJ_real": energy,
                "iter": it,
                "NWOT_pred": float(row[CFG["SCORE_COL"]]),
                "Energy_pred": float(row[CFG["ENERGY_COL"]]),
                "Energy_src": esrc,
                "Latency_ms_real": (np.nan if latency is None else float(latency)),
                "Latency_pred": (np.nan if CFG.get("LATENCY_PRED_COL") is None else
                                 (np.nan if CFG["LATENCY_PRED_COL"] not in row.index else float(pd.to_numeric(pd.Series([row[CFG['LATENCY_PRED_COL']]]), errors="coerce").iloc[0]))),
                "Latency_src": lsrc,
            }
            measured = pd.concat([measured, pd.DataFrame([rec])], ignore_index=True)
            measured.to_csv(measured_csv, index=False)

        plot_scatter(measured, it, outdir)
        save_iteration_report(it, measured, pred_sub, outdir)

        it += 1

    if len(measured):
        measured = _clean_measured_energy(measured)
        pf = pareto_front(measured, "Acc_real", "Energy_mJ_real")
        pf.to_csv(outdir / "pareto_final_energy.csv", index=False)
        meta = {
            "cfg": CFG,
            "run": {
                **RUN,
                "gpu_hours_total": RUN["gpu_seconds"] / 3600.0,
                "wall_hours_total": RUN["wall_seconds"] / 3600.0,
                "total_models": int(len(measured)),
                "end_ts": time.time()
            }
        }
        with open(outdir / "experiment_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] EnergyNet 总测量 {len(measured)}，Pareto {len(pf)}，结果在 {outdir}")

if __name__ == "__main__":
    main()
