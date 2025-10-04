# -*- coding: utf-8 -*-
"""
run_transfer.py  (live-line-only + bigger size + nested sampling + importance weighting)
- 只画折线（不画热力图），支持动态刷新；更大 FIGSIZE/DPI
- 嵌套式多样性抽样：同一 seed 下，各 src_k 为前缀关系，减少“抽样运气”
- 重要性加权（基于目标无标注特征的分布比）：缓解 covariate shift，避免少量源样本+零目标异常好
- 可选 overlap 诊断线（右轴），帮助理解“分布覆盖度”
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

import matplotlib  # 后端延迟选择
from joblib import dump, load

# ================================
# 配置
# ================================
CONFIG: Dict = {
    "ROOT": "/home/xiaolong/Downloads/GreenAuto",

    # 设备与后端
    "SOURCE_DEVICE": "Pixel7",
    "TARGET_DEVICE": "Pixel8pro",
    "BACKEND": "cpu",  # "cpu" 或 "gpu"

    # conv 过滤
    "KERNEL_NAME_FILTER": "conv",
    "DROP_DWCONV": True,

    # 目标用 log1p 训练，推理时还原
    "LOG_TARGET": True,

    # I/O
    "ENERGY_DATA_DIR": "prediction/energy_data",
    "CKPT_ROOT": "prediction/pred_ckpts",

    # 训练
    "N_ENSEMBLE": 5,
    "RF_PARAMS": {"n_estimators": 300, "max_depth": None, "random_state": 0, "n_jobs": -1},

    # 迁移（安全校准 + 主动学习）
    "TRANSFER": {
        "ENABLE": True,
        "BUDGET": 128,
        "SEED": 42,
        "DIVERSITY_KMEANS": True,
        "ACTIVE": True,
        "INIT_K": 32,
        "STEP_K": 32,
        "UNCERT_FRAC": 0.8,
        "CAL_KIND": "ridge",     # "ridge"|"huber"
        "CAL_ALPHA": 1.0,
        "GATE_MIN_R": 0.10,
        "RETREAT_TOL": 0.3,
    },

    # 分布自适应（关键改动）
    "DA_SHIFT": {
        "ENABLE": True,          # 开启重要性加权
        "CLIP_MIN": 0.2,         # 权重下/上截断，避免极端值
        "CLIP_MAX": 5.0,
        "LOGREG_C": 1.0          # 域分类器强度（越小正则越强）
    },

    # 抽样（关键改动）
    "SAMPLING": {
        "NESTED_DIVERSE": True,  # True: 同一 seed 下 src_k 为前缀关系
    },

    # Sweep 与作图
    "SWEEP": {
        "ENABLE": True,
        "SRC_SAMPLE_LIST": [64, 128, 256, 512, 1024, 2048, -1],  # -1=全部
        "TGT_BUDGET_LIST": [64, 256, 512, 1024],
        "REPEATS": 3,
        "PLOT": True,
        "SAVE_CSV": True,
        # 动态绘图
        "LIVE_PLOT": True,          # True 开启动态绘图；无 DISPLAY 自动降级为 PNG 快照
        "LIVE_INTERVAL": 0.15,      # plt.pause 秒数（仅窗口模式）
        # 尺寸与清晰度（放大）
        "FIGSIZE": (13, 8),
        "DPI": 180,
        # 诊断：分布重叠度（越小越好）
        "SHOW_OVERLAP": True
    },
}

EPS = 1e-9

# ================
# 工具与特征工程
# ================
def _ensure_dirs(*paths: Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def _summary_path(root: str, device: str, backend: str) -> Path:
    base = Path(root) / "kernel_energy" / device
    for name in [
        f"energy_summary_conv_{backend}_freq_fix.csv",
        f"energy_summary_conv_{backend}.csv",
        f"summary_{backend}.csv",
    ]:
        p = base / name
        if p.exists():
            return p
    return base / f"summary_{backend}.csv"

def _energy_col_anycase(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "energy" in str(c).lower():
            return c
    raise KeyError("未找到能耗列（列名需包含 'energy'）")

def _select_conv_rows(df: pd.DataFrame, kernel_name_filter: str = "conv", drop_dwconv: bool = True) -> pd.DataFrame:
    df = df.copy()
    op_col = None
    for c in ["KernelType", "OP", "Op", "op", "kernel", "type"]:
        if c in df.columns:
            op_col = c
            break
    if op_col is not None:
        op_source = df[op_col].astype(str); used_col = op_col
    elif "Model" in df.columns:
        op_source = df["Model"].astype(str); used_col = "Model"
    elif "id" in df.columns:
        op_source = df["id"].astype(str); used_col = "id"
    else:
        op_source = pd.Series(["unknown"] * len(df), index=df.index); used_col = "(none)"

    lower = op_source.str.lower()
    keep = pd.Series(True, index=df.index)
    if kernel_name_filter:
        keep &= lower.str.contains(kernel_name_filter.lower(), na=False)
    if drop_dwconv:
        keep &= ~lower.str.contains("dwconv", na=False)

    out = df[keep].copy()
    tag = np.where(lower.loc[out.index].str.contains("dwconv", na=False), "dwconv",
                   np.where(lower.loc[out.index].str.contains("conv", na=False), "conv", "unknown"))
    out["OP"] = tag.astype(str)
    print(f"[filter] 使用列: {used_col}  |  保留 {len(out)}/{len(df)} 行")
    return out

def _numericize_conv_params(df: pd.DataFrame) -> pd.DataFrame:
    need = ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    have_all = df[need].notna().all(axis=1).sum()
    print(f"[summary] Conv 行总数: {len(df)}, 形参已齐(数值判定): {have_all}")
    df = df.dropna(subset=need).copy()
    for c in need:
        df[c] = df[c].astype(int)
    return df

def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["spatial"] = (x["HW"] ** 2).astype(np.int64)
    x["k2"] = (x["KERNEL_SIZE"] ** 2).astype(np.int64)
    x["macs_approx"] = ((x["HW"] / np.maximum(1, x["STRIDES"])) ** 2) * x["CIN"] * x["COUT"] * (x["KERNEL_SIZE"] ** 2)
    for col in ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "spatial", "k2", "macs_approx"]:
        x[f"log_{col}"] = np.log1p(x[col].astype(float))
    return x

def _build_conv_csv_from_summary(summary_csv: Path, out_csv: Path, need_energy: bool = True) -> int:
    if not summary_csv.exists():
        print(f"[warn] summary 不存在: {summary_csv}")
        return 0
    df = pd.read_csv(summary_csv)
    energy_col = _energy_col_anycase(df)
    conv = _select_conv_rows(df, CONFIG["KERNEL_NAME_FILTER"], CONFIG["DROP_DWCONV"])
    conv = _numericize_conv_params(conv)

    if "Model" in conv.columns:
        conv["id"] = conv["Model"].astype(str)
    elif "id" not in conv.columns:
        conv["id"] = np.arange(len(conv)).astype(str)

    cols = ["id", "OP", "HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]
    if need_energy:
        cols = ["id", "OP", "energy_mJ", "HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]
        conv = conv.rename(columns={energy_col: "energy_mJ"})

    out = conv[cols].copy()
    out.to_csv(out_csv, index=False)
    print(f"[build] 写入 {out_csv}，行数: {len(out)}")
    return len(out)

# ================
# 指标
# ================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e8, neginf=0.0)

    resid = y_pred - y_true
    ae = np.abs(resid)
    se = resid**2
    ape = ae / np.maximum(np.abs(y_true), EPS)
    smape = 2 * ae / np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)

    try:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    except Exception:
        r = np.nan

    try:
        lr = LinearRegression().fit(y_pred.reshape(-1,1), y_true)
        slope = float(lr.coef_[0]); inter = float(lr.intercept_)
    except Exception:
        slope = np.nan; inter = np.nan

    return {
        "MAE_mJ": float(np.mean(ae)),
        "MedAE_mJ": float(np.median(ae)),
        "RMSE_mJ": float(np.sqrt(np.mean(se))),
        "MAPE_%": float(np.mean(ape) * 100.0),
        "WAPE_%": float(np.sum(ae) / np.maximum(np.sum(np.abs(y_true)), EPS) * 100.0),
        "sMAPE_%": float(np.mean(smape) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": r,
        "APE_P50_%": float(np.percentile(ape * 100.0, 50)),
        "APE_P90_%": float(np.percentile(ape * 100.0, 90)),
        "Calib_slope": slope,
        "Calib_intercept": inter,
    }

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> dict:
    m = compute_metrics(y_true, y_pred)
    print(f"[eval:{tag}] "
          f"MAPE={m['MAPE_%']:.2f}% | WAPE={m['WAPE_%']:.2f}% | "
          f"MAE={m['MAE_mJ']:.1f} mJ | RMSE={m['RMSE_mJ']:.1f} mJ | "
          f"R2={m['R2']:.3f} | r={m['Pearson_r']:.3f} | "
          f"P50={m['APE_P50_%']:.1f}% P90={m['APE_P90_%']:.1f}% | "
          f"slope={m['Calib_slope']:.3f} intercept={m['Calib_intercept']:.1f}")
    return m

# ==================
# 模型：RF 集成
# ==================
class SourceEnsemble:
    def __init__(self, n=5, rf_params=None):
        self.n = n
        self.rf_params = rf_params or {}
        self.models: List[RandomForestRegressor] = []
        self.scaler: Optional[StandardScaler] = None
        self.feat_cols: List[str] = []
        self.yfit_min: Optional[float] = None
        self.yfit_max: Optional[float] = None

    def _feat_cols(self):
        return ["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT",
                "spatial", "k2", "macs_approx",
                "log_HW", "log_KERNEL_SIZE", "log_STRIDES",
                "log_CIN", "log_COUT", "log_spatial", "log_k2", "log_macs_approx"]

    def _prep_x(self, df: pd.DataFrame) -> pd.DataFrame:
        self.feat_cols = self._feat_cols()
        return df[self.feat_cols].astype(float)

    def fit(self, df: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        x = _feature_engineering(df)
        X = self._prep_x(x)
        if len(X) == 0:
            raise ValueError("源训练样本为 0，无法训练。请检查源 CSV 是否建立成功。")
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        use_log = CONFIG.get("LOG_TARGET", False)
        y_fit = np.log1p(y) if use_log else y
        self.yfit_min = float(np.min(y_fit))
        self.yfit_max = float(np.max(y_fit))

        self.models = []
        for i in range(self.n):
            params = dict(self.rf_params)
            params["random_state"] = (params.get("random_state", 0) + i)
            m = RandomForestRegressor(**params)
            if sample_weight is not None:
                m.fit(Xs, y_fit, sample_weight=sample_weight)
            else:
                m.fit(Xs, y_fit)
            self.models.append(m)

        if use_log:
            print(f"[fit] y(train) lin: min={np.min(y):.1f}, max={np.max(y):.1f} | "
                  f"log1p: min={self.yfit_min:.3f}, max={self.yfit_max:.3f}")

    def _to_Xs(self, df: pd.DataFrame) -> np.ndarray:
        x = _feature_engineering(df)
        X = x[self._feat_cols()].astype(float)
        return self.scaler.transform(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None and len(self.models) > 0
        Xs = self._to_Xs(df)
        preds_fit = np.stack([m.predict(Xs) for m in self.models], axis=0).mean(axis=0)

        if CONFIG.get("LOG_TARGET", False):
            lo = (self.yfit_min if self.yfit_min is not None else -5.0) - 0.5
            hi = (self.yfit_max if self.yfit_max is not None else 15.0) + 0.5
            preds_fit = np.clip(preds_fit, lo, hi)
            preds = np.expm1(preds_fit)
        else:
            preds = preds_fit

        preds = np.nan_to_num(preds, nan=0.0, posinf=1e8, neginf=0.0)
        preds = np.clip(preds, 0.0, 1e8)
        return preds

    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        assert self.scaler is not None and len(self.models) > 0
        Xs = self._to_Xs(df)
        per_tree = []
        for m in self.models:
            tree_preds = np.stack([t.predict(Xs) for t in m.estimators_], axis=0)
            per_tree.append(tree_preds)
        per_tree = np.concatenate(per_tree, axis=0)
        mu_fit = per_tree.mean(axis=0)
        std_fit = per_tree.std(axis=0)

        if CONFIG.get("LOG_TARGET", False):
            lo = (self.yfit_min if self.yfit_min is not None else -5.0) - 0.5
            hi = (self.yfit_max if self.yfit_max is not None else 15.0) + 0.5
            mu_fit = np.clip(mu_fit, lo, hi)
            std_fit = np.clip(std_fit, 0.0, 10.0)
            mu = np.expm1(mu_fit)
            std = (np.expm1(mu_fit + std_fit) - mu).clip(min=0.0)
        else:
            mu = mu_fit
            std = std_fit

        mu = np.nan_to_num(mu, nan=0.0, posinf=1e8, neginf=0.0).clip(0.0, 1e8)
        std = np.nan_to_num(std, nan=0.0, posinf=1e8, neginf=0.0)
        return mu, std

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump({
            "n": self.n,
            "rf_params": self.rf_params,
            "models": self.models,
            "scaler": self.scaler,
            "feat_cols": self.feat_cols,
            "yfit_min": self.yfit_min,
            "yfit_max": self.yfit_max,
        }, path)
        print(f"[save] 源模型已保存: {path}")

    @staticmethod
    def load(path: Path) -> "SourceEnsemble":
        obj = load(path)
        ens = SourceEnsemble(n=obj["n"], rf_params=obj["rf_params"])
        ens.models = obj["models"]
        ens.scaler = obj["scaler"]
        ens.feat_cols = obj["feat_cols"]
        ens.yfit_min = obj.get("yfit_min", None)
        ens.yfit_max = obj.get("yfit_max", None)
        return ens

# ==================
# 安全校准
# ==================
class SafeCalibrator:
    def __init__(self, kind="ridge", alpha=1.0, use_log=True, retreat_tol=0.3):
        self.kind = kind
        self.alpha = alpha
        self.use_log = use_log
        self.retreat_tol = float(retreat_tol)
        self.bias_only = False
        self.model = None
        self.yfit_min = None
        self.yfit_max = None

    def _tx(self, y):
        return np.log1p(np.maximum(y, EPS)) if self.use_log else y

    def _itx(self, y):
        return np.expm1(y) if self.use_log else y

    def _clipfit(self, y):
        lo = (self.yfit_min if self.yfit_min is not None else -5.0) - 0.5
        hi = (self.yfit_max if self.yfit_max is not None else 15.0) + 0.5
        return np.clip(y, lo, hi)

    def fit(self, y_true, y_base):
        X = self._tx(y_base).reshape(-1,1)
        t = self._tx(y_true)
        self.yfit_min, self.yfit_max = float(t.min()), float(t.max())

        b = HuberRegressor(alpha=0.0, fit_intercept=True).fit(np.zeros_like(X), t)
        if self.kind == "ridge":
            m = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, t)
        else:
            m = HuberRegressor().fit(X, t)

        def mape_of(mdl, use_bias=False):
            x = np.zeros_like(X) if use_bias else X
            yhat = mdl.predict(x)
            yhat = self._clipfit(yhat)
            y = self._itx(yhat)
            return float(np.mean(np.abs(y - y_true) / np.maximum(y_true, EPS)) * 100.0)

        mape_bias = mape_of(b, use_bias=True)
        mape_full = mape_of(m, use_bias=False)
        if mape_full <= mape_bias - self.retreat_tol:
            self.model = m; self.bias_only = False
        else:
            self.model = b; self.bias_only = True

    def predict(self, y_base):
        x = self._tx(y_base).reshape(-1,1)
        if self.bias_only:
            x = np.zeros_like(x)
        yhat = self.model.predict(x)
        yhat = self._clipfit(yhat)
        return self._itx(yhat)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump({"kind": self.kind, "alpha": self.alpha, "use_log": self.use_log,
              "retreat_tol": self.retreat_tol, "bias_only": self.bias_only,
              "yfit_min": self.yfit_min, "yfit_max": self.yfit_max,
              "model": self.model}, path)
        print(f"[save] 线性校准器已保存: {path}")

    @staticmethod
    def load(path: Path) -> "SafeCalibrator":
        obj = load(path)
        cal = SafeCalibrator(obj["kind"], obj["alpha"], obj["use_log"], obj["retreat_tol"])
        cal.bias_only = obj["bias_only"]
        cal.yfit_min = obj["yfit_min"]; cal.yfit_max = obj["yfit_max"]
        cal.model = obj["model"]
        return cal

# ================
# 采样 & 诊断
# ================
def _sample_diverse(df_feat: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """保留：独立取 k 个原型（非嵌套）。"""
    feat_cols = ["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT",
                 "spatial", "k2", "macs_approx",
                 "log_HW", "log_KERNEL_SIZE", "log_STRIDES",
                 "log_CIN", "log_COUT", "log_spatial", "log_k2", "log_macs_approx"]
    scaler = StandardScaler()
    Z = scaler.fit_transform(df_feat[feat_cols].astype(float).values)
    if k >= len(Z):
        return np.arange(len(Z))
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(Z)
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, Z)
    return np.unique(idx)

def _diverse_order(df_feat: pd.DataFrame, max_k: int, seed: int) -> np.ndarray:
    """嵌套式多样性顺序（前缀最具多样性，贪心最远点采样）。返回长度 max_k 的索引序列。"""
    feat_cols = ["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT",
                 "spatial", "k2", "macs_approx",
                 "log_HW", "log_KERNEL_SIZE", "log_STRIDES",
                 "log_CIN", "log_COUT", "log_spatial", "log_k2", "log_macs_approx"]
    X = df_feat[feat_cols].astype(float).values
    Z = StandardScaler().fit_transform(X)
    n = len(Z)
    max_k = int(min(max_k, n))
    rng = np.random.RandomState(seed)
    start = int(rng.randint(n))
    order = [start]
    # 当前每个点到已选集合的最小距离
    dmin = np.linalg.norm(Z - Z[start], axis=1)
    for _ in range(1, max_k):
        # 选取与已选集合最远的点
        j = int(np.argmax(dmin))
        order.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(order, dtype=int)

def _read_or_build(src_summary: Path, tgt_summary: Path, energy_data_dir: Path, backend: str,
                   source_device: str, target_device: str) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    SRC_CSV = energy_data_dir / f"conv_source_{source_device}_{backend}.csv"
    TGT_FULL_CSV = energy_data_dir / f"{target_device}_full_{backend}.csv"
    PRED_CSV = energy_data_dir / f"preds_{target_device}_{backend}.csv"

    if SRC_CSV.exists():
        print(f"[info] 源 CSV 已存在，跳过重建：{SRC_CSV}")
    else:
        _build_conv_csv_from_summary(src_summary, SRC_CSV, need_energy=True)

    if TGT_FULL_CSV.exists():
        print(f"[info] 目标 FULL CSV 已存在，跳过重建：{TGT_FULL_CSV}")
    else:
        _build_conv_csv_from_summary(tgt_summary, TGT_FULL_CSV, need_energy=True)

    src = pd.read_csv(SRC_CSV) if SRC_CSV.exists() else pd.DataFrame()
    tgt = pd.read_csv(TGT_FULL_CSV) if TGT_FULL_CSV.exists() else pd.DataFrame()

    if len(src) == 0:
        print(f"[warn] 源 CSV 当前为 0 行，尝试根据 summary 重新构建。")
        _build_conv_csv_from_summary(src_summary, SRC_CSV, need_energy=True)
        src = pd.read_csv(SRC_CSV) if SRC_CSV.exists() else pd.DataFrame()

    if len(tgt) == 0:
        print(f"[warn] 目标 CSV 当前为 0 行，尝试根据 summary 重新构建。")
        _build_conv_csv_from_summary(tgt_summary, TGT_FULL_CSV, need_energy=True)
        tgt = pd.read_csv(TGT_FULL_CSV) if TGT_FULL_CSV.exists() else pd.DataFrame()

    return src, tgt, SRC_CSV, TGT_FULL_CSV, PRED_CSV

def _overlap_score(src_df: pd.DataFrame, tgt_df: pd.DataFrame, k_src: int, seed: int = 0, order: Optional[np.ndarray]=None) -> float:
    """目标到源子集的平均最近邻距离（标准化后的特征空间），越小代表分布越贴近。"""
    feat = ["HW","KERNEL_SIZE","STRIDES","CIN","COUT","spatial","k2","macs_approx",
            "log_HW","log_KERNEL_SIZE","log_STRIDES","log_CIN","log_COUT","log_spatial","log_k2","log_macs_approx"]

    Xs_full = _feature_engineering(src_df[["HW","KERNEL_SIZE","STRIDES","CIN","COUT"]])[feat].astype(float)
    Xt = _feature_engineering(tgt_df[["HW","KERNEL_SIZE","STRIDES","CIN","COUT"]])[feat].astype(float)

    if k_src <= 0 or k_src >= len(src_df):
        idx = np.arange(len(src_df))
    else:
        if order is None:
            idx = _sample_diverse(Xs_full, k=k_src, seed=seed)
        else:
            idx = order[:k_src]

    Xs = Xs_full.iloc[idx]
    scaler = StandardScaler().fit(pd.concat([Xs, Xt], axis=0))
    Zs, Zt = scaler.transform(Xs), scaler.transform(Xt)

    nn = NearestNeighbors(n_neighbors=1).fit(Zs)
    dist, _ = nn.kneighbors(Zt, n_neighbors=1)
    return float(dist.mean())

def _compute_importance_weights(X_src: pd.DataFrame, X_tgt: pd.DataFrame) -> np.ndarray:
    """基于无标注目标特征的域分类器估计权重（目标密度/源密度比）。"""
    if not CONFIG.get("DA_SHIFT", {}).get("ENABLE", False):
        return np.ones(len(X_src), dtype=float)

    feat = ["HW","KERNEL_SIZE","STRIDES","CIN","COUT","spatial","k2","macs_approx",
            "log_HW","log_KERNEL_SIZE","log_STRIDES","log_CIN","log_COUT","log_spatial","log_k2","log_macs_approx"]

    Xs = _feature_engineering(X_src)[feat].astype(float)
    Xt = _feature_engineering(X_tgt)[feat].astype(float)

    X_all = pd.concat([Xs, Xt], axis=0).values
    y_all = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    scaler = StandardScaler().fit(X_all)
    Z = scaler.transform(X_all)

    clf = LogisticRegression(max_iter=1000, C=float(CONFIG["DA_SHIFT"]["LOGREG_C"]), class_weight="balanced")
    clf.fit(Z, y_all)
    ps = clf.predict_proba(scaler.transform(Xs))[:, 1]  # p(target | x_s)
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    w = ps / (1.0 - ps)

    w = np.clip(w, float(CONFIG["DA_SHIFT"]["CLIP_MIN"]), float(CONFIG["DA_SHIFT"]["CLIP_MAX"]))
    w = w / float(np.mean(w))  # 归一到均值 1
    return w

# ================
# Matplotlib 后端初始化 & Live 折线
# ================
def _init_matplotlib_backend():
    live_flag = CONFIG.get("SWEEP", {}).get("LIVE_PLOT", False)
    try:
        if live_flag and os.environ.get("DISPLAY"):
            matplotlib.use("TkAgg", force=True)
            print("[plot] 使用 TkAgg（动态图窗口）")
        else:
            if live_flag and not os.environ.get("DISPLAY"):
                print("[plot] 未检测到 DISPLAY，关闭动态图；退回 Agg，仅持续输出 PNG 快照")
                CONFIG["SWEEP"]["LIVE_PLOT"] = False
            matplotlib.use("Agg", force=True)
            print("[plot] 使用 Agg（无窗口）")
    except Exception as e:
        print("[plot] 后端初始化失败，退回 Agg：", e)
        matplotlib.use("Agg", force=True)
    global plt
    import matplotlib.pyplot as plt  # noqa: E402
    return plt

class LiveLinePlotter:
    """
    仅负责动态折线图（MAPE vs Source samples），不做视频也不画 overlap/heatmap。
    """
    def __init__(self, plt, src_ticks, tgt_ticks, tag, outdir: Path,
                 live: bool, pause_sec: float, figsize=(18, 11), dpi=220):
        self.plt = plt
        self.src_ticks = list(src_ticks)      # x 轴
        self.tgt_ticks = list(tgt_ticks)      # 每条线一条 target 预算
        self.tag = tag
        self.outdir = Path(outdir)
        self.live = bool(live)
        self.pause = float(pause_sec)
        self.figsize = tuple(figsize)
        self.dpi = int(dpi)

        # 存放每条 target 线在各个 source 样本下的 MAPE（NaN 表示尚未填充）
        self.grid = np.full((len(self.tgt_ticks), len(self.src_ticks)), np.nan, dtype=float)

        if self.live:
            self.plt.ion()  # 交互模式

        self.fig, self.ax = self.plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.ax.set_xlabel("Source samples")
        self.ax.set_ylabel("MAPE (%)")
        self.ax.set_title(f"Accuracy vs Source/Target samples ({self.tag})")

        # 为每个 target 预算建一条线
        self.lines = {}
        for t in self.tgt_ticks:
            (line,) = self.ax.plot([], [], marker="o", label=f"target={t}")
            self.lines[t] = line

        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self._draw(initial=True)

    def _save_snapshot(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.outdir / f"lines_{self.tag}_live.png", dpi=self.dpi)

    def _draw(self, initial=False):
        # 强制刷新一帧
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.live:
            self.plt.pause(0.001 if initial else self.pause)
        else:
            self._save_snapshot()

    def update_mape_point(self, src_k: int, tgt_k: int, mape: float):
        # 将 (src_k, tgt_k) 的点更新到网格，并刷新曲线
        j = self.src_ticks.index(src_k)
        i = self.tgt_ticks.index(tgt_k)
        self.grid[i, j] = float(mape)

        line = self.lines.get(tgt_k)
        if line is not None:
            xs, ys = [], []
            for jj, sk in enumerate(self.src_ticks):
                v = self.grid[i, jj]
                if np.isfinite(v):
                    xs.append(sk); ys.append(float(v))
            line.set_data(xs, ys)
            self.ax.relim()
            self.ax.autoscale_view()

        self._draw()

    def finalize(self):
        if self.live:
            self.plt.ioff()
        self._save_snapshot()

# ================
# 主逻辑片段
# ================
def _gate_by_correlation(y_true: np.ndarray, y_pred: np.ndarray, min_r: float) -> bool:
    try:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        return np.isfinite(r) and (abs(r) >= min_r)
    except Exception:
        return False

def _active_calibrate(src_ens: SourceEnsemble, X_tgt: pd.DataFrame, y_tgt_true: np.ndarray,
                      budget: int, seed: int, init_k: int, step_k: int,
                      cal_kind: str, cal_alpha: float, gate_min_r: float, retreat_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = len(X_tgt)
    all_idx = np.arange(N)

    y_base, uncert = src_ens.predict_with_uncertainty(X_tgt)
    _ = _gate_by_correlation(y_tgt_true, y_base, min_r=gate_min_r)

    if budget <= 0:
        return y_base, np.array([], dtype=int)

    k0 = min(init_k, budget, N)
    feat_tgt = _feature_engineering(X_tgt)
    chosen = set(_sample_diverse(feat_tgt, k=k0, seed=seed).tolist())

    remain_budget = budget - len(chosen)
    step = max(1, min(step_k, N))
    while remain_budget > 0 and len(chosen) < N:
        k = min(step, remain_budget)
        pool = np.array(sorted(list(set(all_idx) - chosen)))
        if len(pool) == 0:
            break
        pool_unc = uncert[pool]
        order = np.argsort(-pool_unc)
        top_m = max(k, int(np.ceil(k * 3)))
        cand = pool[order[:min(top_m, len(pool))]]
        sub_feat = feat_tgt.iloc[cand]
        pick = _sample_diverse(sub_feat, k=k, seed=int(rng.randint(1e9)))
        chosen.update(cand[pick].tolist())
        remain_budget = budget - len(chosen)

    idx = np.array(sorted(list(chosen)), dtype=int)

    cal = SafeCalibrator(kind=cal_kind, alpha=cal_alpha, use_log=CONFIG["LOG_TARGET"], retreat_tol=retreat_tol)
    cal.fit(y_tgt_true[idx], y_base[idx])
    y_cal = cal.predict(y_base)

    return y_cal, idx

def _train_predict_once(src_df: pd.DataFrame, tgt_df: pd.DataFrame,
                        src_k: int, tgt_budget: int, seed: int,
                        src_order: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """训练一次并在目标上评估，返回 (MAPE_base, MAPE_final)"""
    X_src_all = src_df[["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]].copy()
    y_src_all = src_df["energy_mJ"].astype(float).values

    if src_k <= 0 or src_k >= len(src_df):
        src_idx = np.arange(len(src_df))
    else:
        if CONFIG["SAMPLING"].get("NESTED_DIVERSE", True) and src_order is not None:
            src_idx = src_order[:src_k]
        else:
            src_idx = _sample_diverse(_feature_engineering(X_src_all), k=src_k, seed=seed)

    X_src = X_src_all.iloc[src_idx]
    y_src = y_src_all[src_idx]

    # 重要性加权（基于目标无标注特征）
    X_tgt_all = tgt_df[["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]].copy()
    w = _compute_importance_weights(X_src, X_tgt_all)

    src_ens = SourceEnsemble(n=CONFIG["N_ENSEMBLE"], rf_params=CONFIG["RF_PARAMS"])
    src_ens.fit(X_src, y_src, sample_weight=w)

    X_tgt = X_tgt_all
    y_tgt_true = tgt_df["energy_mJ"].astype(float).values
    y_tgt_base = src_ens.predict(X_tgt)

    m_base = _evaluate(y_tgt_true, y_tgt_base, tag=f"zero-shot(src={len(X_src)})")["MAPE_%"]

    if tgt_budget > 0:
        tr = CONFIG["TRANSFER"]
        y_tgt_final, _ = _active_calibrate(
            src_ens, X_tgt, y_tgt_true, budget=tgt_budget, seed=seed,
            init_k=tr["INIT_K"], step_k=tr["STEP_K"],
            cal_kind=tr["CAL_KIND"], cal_alpha=tr["CAL_ALPHA"],
            gate_min_r=tr["GATE_MIN_R"], retreat_tol=tr["RETREAT_TOL"]
        )
    else:
        y_tgt_final = y_tgt_base

    m_final = _evaluate(y_tgt_true, y_tgt_final, tag=f"transfer@{tgt_budget}")["MAPE_%"]
    return float(m_base), float(m_final)

# ================
# 边算边画：Sweep & 动态折线（含可选 overlap 诊断）
# ================
def _run_sweep_and_plot(src: pd.DataFrame, tgt: pd.DataFrame, plots_dir: Path,
                        src_dev: str, tgt_dev: str, backend: str):
    sw = CONFIG["SWEEP"]
    if not sw.get("ENABLE", False):
        return

    _ensure_dirs(plots_dir)

    # 后端选择：有 DISPLAY 时 TkAgg，可弹大窗口；否则退回 Agg 只保存图片
    try:
        if sw.get("LIVE_PLOT", True) and os.environ.get("DISPLAY"):
            import matplotlib
            matplotlib.use("TkAgg", force=True)
            print("[plot] 使用 TkAgg（动态大窗口）")
        else:
            import matplotlib
            matplotlib.use("Agg", force=True)
            print("[plot] 使用 Agg（无窗口，仅保存PNG）")
    except Exception:
        import matplotlib
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt  # 放在后端设定之后再导入

    src_list = sw.get("SRC_SAMPLE_LIST", [64, 128, 256, 512, 1024, 2048, -1])
    tgt_list = sw.get("TGT_BUDGET_LIST", [16, 128, 256, 512])   # 没有 0
    repeats = int(sw.get("REPEATS", 3))
    figsize = tuple(sw.get("FIGSIZE", (18, 11)))                 # 大窗口
    dpi = int(sw.get("DPI", 220))
    tag = f"{src_dev}_to_{tgt_dev}_{backend}"

    # 将 -1 归一化为“全部样本数”
    src_ticks = [(len(src) if s <= 0 else min(s, len(src))) for s in src_list]

    # —— 动态折线图（无 overlap/无 heatmap）——
    plotter = LiveLinePlotter(
        plt=plt,
        src_ticks=src_ticks,
        tgt_ticks=tgt_list,
        tag=tag,
        outdir=plots_dir,
        live=bool(sw.get("LIVE_PLOT", True)),
        pause_sec=float(sw.get("LIVE_INTERVAL", 0.10)),
        figsize=figsize,
        dpi=dpi
    )

    rows = []
    for s_cfg, s_norm in zip(src_list, src_ticks):
        for t in tgt_list:
            base_arr, fin_arr = [], []
            for r in range(repeats):
                seed = CONFIG["TRANSFER"]["SEED"] + r
                m_base, m_final = _train_predict_once(src, tgt, src_k=s_cfg, tgt_budget=t, seed=seed)
                base_arr.append(m_base); fin_arr.append(m_final)
                # 即时更新曲线（用当前均值让曲线更稳）
                plotter.update_mape_point(s_norm, t, float(np.mean(fin_arr)))

            rows.append({
                "src_samples": s_norm,
                "tgt_budget": t,
                "mape_base_mean": float(np.mean(base_arr)),
                "mape_final_mean": float(np.mean(fin_arr)),
                "mape_final_std": float(np.std(fin_arr)),
            })

    df_sweep = pd.DataFrame(rows).sort_values(["tgt_budget", "src_samples"])

    if sw.get("SAVE_CSV", True):
        out_csv = plots_dir / f"sweep_{tag}.csv"
        df_sweep.to_csv(out_csv, index=False)
        print(f"[sweep] 结果已保存: {out_csv}")

    plotter.finalize()

    if not sw.get("PLOT", True):
        return

    # —— 汇总静态大图（仍然不画 overlap/heatmap）——
    plt.figure(figsize=figsize, dpi=dpi)
    for t in tgt_list:
        sub = df_sweep[df_sweep["tgt_budget"] == t]
        x = sub["src_samples"].values
        y = sub["mape_final_mean"].values
        yerr = sub["mape_final_std"].values
        plt.plot(x, y, marker="o", label=f"target={t}")
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.12)
    plt.xlabel("Source samples")
    plt.ylabel("MAPE (%)")
    plt.title(f"Accuracy vs Source/Target samples ({tag})")
    plt.legend()
    line_png = plots_dir / f"lines_{tag}.png"
    plt.tight_layout()
    plt.savefig(line_png, dpi=dpi)
    plt.close()
    print(f"[plot] 折线图已保存: {line_png}")

# ================
# main
# ================
def main():
    _init_matplotlib_backend()

    root = Path(CONFIG["ROOT"])
    energy_data_dir = root / CONFIG["ENERGY_DATA_DIR"]
    ckpt_root = root / CONFIG["CKPT_ROOT"]
    _ensure_dirs(energy_data_dir, ckpt_root)

    src_summary = _summary_path(CONFIG["ROOT"], CONFIG["SOURCE_DEVICE"], CONFIG["BACKEND"])
    tgt_summary = _summary_path(CONFIG["ROOT"], CONFIG["TARGET_DEVICE"], CONFIG["BACKEND"])

    print("[info] 配置：")
    print(f"  - 源设备: {CONFIG['SOURCE_DEVICE']}  目标设备: {CONFIG['TARGET_DEVICE']}  后端: {CONFIG['BACKEND']}")
    print(f"  - summary 源: {src_summary}")
    print(f"  - summary 目标: {tgt_summary}")
    print(f"  - 数据目录: {energy_data_dir}")
    print(f"  - 模型目录: {ckpt_root}")

    src, tgt, SRC_CSV, TGT_FULL_CSV, PRED_CSV = _read_or_build(
        src_summary, tgt_summary, energy_data_dir, CONFIG["BACKEND"],
        CONFIG["SOURCE_DEVICE"], CONFIG["TARGET_DEVICE"]
    )

    # 单次基准（不涉及 sweep）
    y_src = src["energy_mJ"].astype(float).values
    X_src = src[["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]].copy()

    # 这里也使用重要性加权
    X_tgt_all = tgt[["HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]].copy()
    w_single = _compute_importance_weights(X_src, X_tgt_all)

    src_ens = SourceEnsemble(n=CONFIG["N_ENSEMBLE"], rf_params=CONFIG["RF_PARAMS"])
    src_ens.fit(X_src, y_src, sample_weight=w_single)

    src_ckpt = ckpt_root / CONFIG["SOURCE_DEVICE"] / f"conv_rf_ens_{CONFIG['BACKEND']}.joblib"
    _ensure_dirs(src_ckpt.parent); src_ens.save(src_ckpt)

    X_tgt = X_tgt_all
    y_tgt_true = tgt["energy_mJ"].astype(float).values
    y_tgt_base = src_ens.predict(X_tgt)
    _evaluate(y_tgt_true, y_tgt_base, tag="zero-shot(full)")

    y_tgt_final = y_tgt_base
    if CONFIG["TRANSFER"]["ENABLE"]:
        tr = CONFIG["TRANSFER"]
        y_tgt_final, idx_used = _active_calibrate(
            src_ens, X_tgt, y_tgt_true,
            budget=int(tr["BUDGET"]), seed=int(tr["SEED"]),
            init_k=int(tr["INIT_K"]), step_k=int(tr["STEP_K"]),
            cal_kind=tr["CAL_KIND"], cal_alpha=float(tr["CAL_ALPHA"]),
            gate_min_r=float(tr["GATE_MIN_R"]), retreat_tol=float(tr["RETREAT_TOL"])
        )
        print(f"[transfer] 主动学习使用目标样本: {len(idx_used)} | 示例索引: {idx_used[:10]}")
        _evaluate(y_tgt_true, y_tgt_final, tag=f"active@{len(idx_used)}")

    out_df = tgt[["id", "OP", "HW", "KERNEL_SIZE", "STRIDES", "CIN", "COUT"]].copy()
    out_df["energy_true_mJ"] = y_tgt_true
    out_df["energy_pred_base_mJ"] = y_tgt_base
    out_df["energy_pred_final_mJ"] = y_tgt_final
    out_df.to_csv(PRED_CSV, index=False)
    print(f"[done] 已导出目标预测：{PRED_CSV}")

    plots_dir = energy_data_dir / "plots"
    _run_sweep_and_plot(src, tgt, plots_dir, CONFIG["SOURCE_DEVICE"], CONFIG["TARGET_DEVICE"], CONFIG["BACKEND"])

if __name__ == "__main__":
    main()
