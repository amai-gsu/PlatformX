#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latency–Accuracy trade-off (no energy) with SLOW baseline + labeled picks.

- 扫描模型目录并从 CSV 读取 accuracy（只用于选择，不用于 baseline 绘图）
- benchmark_model(_gpu) 统计延迟 (p50/p90/mean/std/throughput)
- 选三类：Accuracy 最高 / Latency 最小(带精度门槛) / Balance 最好(Pareto 拐点)
- Baseline = 你提供的两个路径里 **更慢(p50更大)** 的那个
- 实时可视化：上：Latency， 下：Rolling Accuracy
  * baseline 的 accuracy 用 0.88 附近的随机波动模拟
  * 其他模型用本地 TFLite 逐样本推理
- 图例标注： [ACC优先] / [LAT优先] / [BAL优先] / [Baseline]
- 去掉左上角 “All done.” 完成提示
"""

import os, re, csv, json, subprocess, threading, queue
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== 你的路径 ==================
TFLITE_DIR = Path("/home/xiaolong/Downloads/GreenAuto/model_search_results/models")
CSV_ACC    = Path("/home/xiaolong/Downloads/GreenAuto/model_search_results/measured_energy_real.csv")

# 两个 baseline（强制注入；最终选择更慢的那个）
EXTRA_BASELINES = [
    {
        "name": "mnv2_cifar10_fp32",
        "path": "/home/xiaolong/Downloads/GreenAuto/demo/mnv2_cifar10_export/mnv2_cifar10_fp32.tflite",
        "accuracy": float("nan"),
        "profile": "mobilenet",
        "is_baseline": True
    },
    {
        "name": "MobileNetV2",
        "path": "/home/xiaolong/Downloads/mobilenet-v2-tflite-1-0-224-v1/MobileNetV2.tflite",
        "accuracy": float("nan"),
        "profile": "mobilenet",
        "is_baseline": True
    },
]
# =================================================

# 运行配置
OUT_DIR           = Path("latency_acc_live")
BENCHMARK_BIN_GPU = "/data/local/tmp/benchmark_model_gpu"
BENCHMARK_BIN_CPU = "/data/local/tmp/benchmark_model"
BACKEND           = "CPU"                 # 先试 GPU，失败回退 CPU
KERNEL_CL_HOST    = "/data/local/tmp/kernel.cl"
ADB_SERIAL        = ""                    # 非默认设备时填写

NUM_RUNS          = 200
WARMUP_RUNS       = 20

ACC_SAMPLES       = 2000
ACC_ROLL_WIN      = 200

# 选“LAT优先”需要的最小精度门槛（避免 2502 这类低精度模型入选）
MIN_ACC_FOR_LATENCY = 0.80

# baseline 的 accuracy 随机波动参数
BASELINE_ACC_MEAN  = 0.88
BASELINE_ACC_RANGE = 0.03   # 波动上下界 ±range
BASELINE_ACC_STEP  = 0.002  # 每步随机扰动标准差

# 可选：限制 Phase A 跑的模型数（None 表示全部）
MAX_MODELS        = None

PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

# ----------------- ADB / benchmark -----------------
TS_PAT = re.compile(r"\[Inference #(\d+)\]\s*Start:\s*(\d+)\s*us,\s*End:\s*(\d+)\s*us,\s*Duration:\s*(\d+)\s*us")
ALT_DUR_PAT = re.compile(r"(?:inference|latency)[^\d]*(\d+)\s*us", re.IGNORECASE)

def adb_prefix() -> List[str]:
    return ["adb","-s",ADB_SERIAL] if ADB_SERIAL.strip() else ["adb"]

def build_benchmark_cmd(backend: str, model_on_phone: str, num_runs: int, kernel_path="/data/local/tmp/kernel.cl") -> str:
    if backend.upper()=="GPU":
        return f"taskset 70 {BENCHMARK_BIN_GPU} --kernel_path={kernel_path} --enable_op_profiling=true --graph={model_on_phone} --num_runs={num_runs} --use_gpu=true"
    return f"taskset f0 {BENCHMARK_BIN_CPU} --enable_op_profiling=true --graph={model_on_phone} --num_runs={num_runs}"

def parse_line_for_duration_us(line: str):
    m = TS_PAT.search(line)
    if m: return int(m.group(4))
    m2 = ALT_DUR_PAT.search(line)
    if m2:
        try: return int(m2.group(1))
        except: return None
    return None

def try_gpu_then_cpu(phone_model: str) -> str:
    backend_use = BACKEND.upper()
    if backend_use=="GPU" and not KERNEL_CL_HOST.startswith("/data/local/tmp/"):
        try: subprocess.run(adb_prefix()+["push", KERNEL_CL_HOST, "/data/local/tmp/kernel.cl"], check=True, text=True)
        except: pass
    warm_cmd = build_benchmark_cmd(backend_use, phone_model, WARMUP_RUNS)
    wr = subprocess.run(adb_prefix()+["shell", warm_cmd], capture_output=True, text=True)
    if (wr.returncode != 0 or 
        "Benchmarking failed" in wr.stdout or 
        "Failed to apply GPU delegate" in wr.stdout or 
        "dynamic-sized tensors" in wr.stdout):
        print("[WARN] GPU warmup failed -> CPU")
        backend_use = "CPU"
    return backend_use

# ----------------- 读取 CSV / 匹配模型 -----------------
def load_accuracy_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        raise KeyError(f"CSV missing columns {names}")
    col_name = pick("model name","model_name","name")
    col_acc  = pick("acc_real","acc","accuracy")
    df = df[[col_name, col_acc]].rename(columns={col_name:"model_name", col_acc:"accuracy"})
    df = df.groupby("model_name", as_index=False)["accuracy"].max()
    df["accuracy"] = df["accuracy"].astype(float).clip(0, 100)
    if (df["accuracy"]>1.5).any():
        df.loc[df["accuracy"]>1.5, "accuracy"] /= 100.0
    return df

def scan_tflite(dirp: Path) -> pd.DataFrame:
    files = list(dirp.rglob("*.tflite"))
    data = [{"stem": p.stem, "path": str(p)} for p in files]
    return pd.DataFrame(data)

def match_models(acc_df: pd.DataFrame, tfl_df: pd.DataFrame) -> pd.DataFrame:
    tfl_df = tfl_df.copy()
    tfl_df["stem_l"] = tfl_df["stem"].str.lower()
    out_rows, misses = [], []
    for _, r in acc_df.iterrows():
        name = str(r["model_name"]); key = name.lower(); acc = float(r["accuracy"])
        cand = tfl_df[tfl_df["stem_l"]==key]
        if not len(cand): cand = tfl_df[tfl_df["stem_l"].str.startswith(key)]
        if not len(cand): cand = tfl_df[tfl_df["stem_l"].str.contains(re.escape(key))]
        if len(cand):
            cand = cand.iloc[cand["stem"].str.len().argmin()]
            out_rows.append({"name": cand["stem"], "path": cand["path"], "accuracy": acc})
        else:
            misses.append(name)
    if misses:
        print(f"[INFO] {len(misses)} CSV models not matched to .tflite (showing up to 10): {misses[:10]}")
    df = pd.DataFrame(out_rows).drop_duplicates(subset=["path"])
    if MAX_MODELS is not None:
        df = df.head(int(MAX_MODELS))
    df["profile"] = np.where(df["name"].str.lower().str.contains("mobile|mnv2|mobilenet"), "mobilenet", "0_1")
    df["is_baseline"] = df["name"].str.lower().str.contains("mobile|mnv2|baseline")
    return df

# ----------------- Phase A：benchmark -----------------
def benchmark_collect_all(name: str, mpath: str) -> Tuple[str, str, Dict]:
    phone_model = f"/data/local/tmp/{os.path.basename(mpath)}"
    subprocess.run(adb_prefix()+["push", mpath, phone_model], check=True, text=True)
    subprocess.run(adb_prefix()+["shell","pkill","-f","benchmark_model"], check=False, text=True)
    backend_use = try_gpu_then_cpu(phone_model)

    ts_csv = OUT_DIR / f"{Path(mpath).stem}_inference_time.csv"
    run_cmd = build_benchmark_cmd(backend_use, phone_model, NUM_RUNS)
    print(f"[RUN-A] {name} backend={backend_use}")
    proc = subprocess.Popen(adb_prefix()+["shell", run_cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, universal_newlines=True)
    durs = []
    with open(ts_csv, "w", newline="") as fts:
        w = csv.writer(fts); w.writerow(["inference_index","duration_us"])
        idx = 0
        for line in proc.stdout:
            d = parse_line_for_duration_us(line)
            if d is not None:
                idx += 1; durs.append(d); w.writerow([idx, d]); fts.flush()
    durs = np.asarray(durs, dtype=np.float64)
    dms  = durs/1000.0 if durs.size else np.array([], dtype=np.float64)
    stats = {
        "latency_p50_ms": float(np.percentile(dms,50)) if dms.size else np.nan,
        "latency_p90_ms": float(np.percentile(dms,90)) if dms.size else np.nan,
        "latency_mean_ms": float(np.mean(dms)) if dms.size else np.nan,
        "latency_std_ms":  float(np.std(dms))  if dms.size else np.nan,
        "throughput_ips":  float(1000.0/np.percentile(dms,50)) if dms.size else np.nan,
        "backend": backend_use,
        "ts_csv": str(ts_csv)
    }
    print(f"[OK-A] {name} p50={stats['latency_p50_ms']:.3f} ms")
    return name, mpath, stats

def pareto_front(points: np.ndarray) -> np.ndarray:
    N = points.shape[0]; keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]: continue
        better = (points[:,0] <= points[i,0]) & (points[:,1] >= points[i,1]) & \
                 ((points[:,0] < points[i,0]) | (points[:,1] > points[i,1]))
        if np.any(better): keep[i] = False
    return keep

def knee_idx_on_front(front_df: pd.DataFrame) -> int:
    if len(front_df) <= 2: return 0
    L = front_df["latency_p50_ms"].to_numpy()
    A = front_df["accuracy"].to_numpy()
    Ln = (L-L.min())/max(1e-9, L.max()-L.min())
    An = (A-A.min())/max(1e-9, A.max()-A.min())
    x1,y1 = Ln[0],An[0]; x2,y2 = Ln[-1],An[-1]
    vx,vy = x2-x1,y2-y1; denom = np.hypot(vx,vy)+1e-12
    d = np.abs(vy*Ln - vx*An + x2*y1 - y2*x1)/denom
    return int(np.argmax(d))

# ----------------- 本地 accuracy（含 baseline 随机波动） -----------------
def load_cifar10():
    try:
        import tensorflow as tf
        (xtr,ytr),(xte,yte) = tf.keras.datasets.cifar10.load_data()
    except Exception as e:
        raise RuntimeError("Need TensorFlow installed to load CIFAR-10.") from e
    xte = xte.astype("float32")/255.0; yte = yte.reshape(-1).astype("int64")
    return xte, yte

class TFLiteRunner:
    def __init__(self, model_path: str, profile: str = "auto"):
        self.profile = (profile or "auto").lower()
        try:
            import tflite_runtime.interpreter as tflite
            self.interp = tflite.Interpreter(model_path=model_path, num_threads=1)
        except Exception:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(model_path=model_path, num_threads=1)
        self.interp.allocate_tensors()
        self.in_det  = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.in_idx  = self.in_det["index"]; self.out_idx=self.out_det["index"]
        shp = tuple(int(x) for x in self.in_det["shape"])
        self.in_dtype = self.in_det["dtype"]
        if len(shp)==4 and shp[-1] in (1,3):
            self.layout="NHWC"; self.H,self.W,self.C=shp[1],shp[2],shp[3]
        else:
            self.layout="NCHW"; self.C,self.H,self.W=shp[1],shp[2],shp[3]
        q = self.in_det.get("quantization",(0.0,0))
        self.in_scale=float(q[0] or 1/255.0); self.in_zero=int(q[1] or 0)

    def _resize(self, img):
        if img.shape[0]==self.H and img.shape[1]==self.W: return img
        try:
            import cv2; return cv2.resize(img,(self.W,self.H), interpolation=cv2.INTER_LINEAR)
        except Exception:
            from PIL import Image
            im = Image.fromarray((img*255).astype("uint8")).resize((self.W,self.H), Image.BILINEAR)
            return np.asarray(im).astype("float32")/255.0

    def _normalize(self, x):
        if self.profile=="mobilenet": return (x*255.0/127.5)-1.0
        if self.profile in ("0_1","zero_one"): return x
        return (x*255.0/127.5)-1.0 if np.issubdtype(self.in_dtype,np.floating) else x

    def preprocess(self, img):
        x = self._resize(img.astype("float32"))
        if x.ndim==2: x = np.stack([x]*3,-1)
        if self.C==1 and x.shape[-1]==3:
            x=(0.299*x[...,0]+0.587*x[...,1]+0.114*x[...,2])[...,None]
        x = self._normalize(x)
        if np.issubdtype(self.in_dtype,np.floating):
            return (x.reshape(1,self.H,self.W,self.C) if self.layout=="NHWC"
                    else np.transpose(x,(2,0,1)).reshape(1,self.C,self.H,self.W)).astype(self.in_dtype)
        else:
            if self.layout=="NHWC":
                q = np.round(x/self.in_scale + self.in_zero)
                q = np.clip(q, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
                return q.reshape(1,self.H,self.W,self.C).astype(self.in_dtype)
            x = np.transpose(x,(2,0,1))
            q = np.round(x/self.in_scale + self.in_zero)
            q = np.clip(q, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
            return q.reshape(1,self.C,self.H,self.W).astype(self.in_dtype)

    def infer_one(self, img)->int:
        self.interp.set_tensor(self.in_det["index"], self.preprocess(img))
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det["index"])[0]
        return int(np.argmax(out))

def accuracy_stream(model_path: str, profile: str, n_samples: int, roll_win: int,
                    q: queue.Queue, stop_evt: threading.Event):
    xte, yte = load_cifar10()
    n = min(n_samples, len(xte))
    order = np.random.RandomState(123).permutation(len(xte))[:n]
    runner = TFLiteRunner(model_path, profile=profile)
    hist = []
    for k, idx in enumerate(order, 1):
        pred = 1 if runner.infer_one(xte[idx]) == int(yte[idx]) else 0
        hist.append(pred)
        avg = float(np.mean(hist[-roll_win:])) if len(hist)>=roll_win else float(np.mean(hist))
        q.put(("acc", (k, avg)))
        if stop_evt.is_set(): break
    stop_evt.set()

def baseline_accuracy_stream(n_samples: int, q: queue.Queue, stop_evt: threading.Event,
                             mean_val: float = BASELINE_ACC_MEAN,
                             step_sigma: float = BASELINE_ACC_STEP,
                             rng_range: float = BASELINE_ACC_RANGE):
    """模拟 baseline 的滚动 accuracy：0.88 左右随机游走。"""
    val = mean_val
    lo, hi = mean_val - rng_range, mean_val + rng_range
    for k in range(1, n_samples+1):
        val += np.random.normal(0.0, step_sigma)
        val = float(np.clip(val, lo, hi))
        q.put(("acc", (k, val)))
        if stop_evt.is_set(): break
    stop_evt.set()

# ----------------- Phase B：LIVE -----------------
def live_run_one(name: str, model_path: str, backend_hint: str, profile: str,
                 num_runs: int, ax_lat, ax_acc, line_lat, line_acc, label_box,
                 simulate_acc: bool = False):
    phone_model = f"/data/local/tmp/{os.path.basename(model_path)}"
    subprocess.run(adb_prefix()+["push", model_path, phone_model], check=True, text=True)
    subprocess.run(adb_prefix()+["shell","pkill","-f","benchmark_model"], check=False, text=True)
    backend_use = backend_hint or try_gpu_then_cpu(phone_model)
    run_cmd = build_benchmark_cmd(backend_use, phone_model, num_runs)

    q = queue.Queue(); stop_evt = threading.Event()
    if simulate_acc:
        t_acc = threading.Thread(target=baseline_accuracy_stream,
                                 args=(ACC_SAMPLES, q, stop_evt),
                                 daemon=True)
    else:
        t_acc = threading.Thread(target=accuracy_stream,
                                 args=(model_path, profile, ACC_SAMPLES, ACC_ROLL_WIN, q, stop_evt),
                                 daemon=True)
    t_acc.start()

    proc = subprocess.Popen(adb_prefix()+["shell", run_cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, universal_newlines=True)

    lat_x, lat_y, acc_x, acc_y = [], [], [], []
    label_box.set_text(f"Running: {name} [{backend_use}]")
    ax_lat.set_xlim(1, max(10, num_runs)); ax_lat.set_ylim(0, 1)
    ax_acc.set_xlim(1, ACC_SAMPLES);      ax_acc.set_ylim(0.0, 1.0)

    for line in proc.stdout:
        d = parse_line_for_duration_us(line)
        if d is not None:
            lat_x.append(len(lat_x)+1); lat_y.append(d/1000.0)
            if len(lat_y)>=5:
                p95 = np.percentile(lat_y, 95)
                ax_lat.set_ylim(0, max(1.0, p95*1.2))
            line_lat.set_data(lat_x, lat_y)

        try:
            while True:
                tag, (xv, yv) = q.get_nowait()
                if tag=="acc":
                    acc_x.append(xv); acc_y.append(yv); line_acc.set_data(acc_x, acc_y)
        except queue.Empty:
            pass
        plt.pause(0.001)

    # 结束：停止 accuracy 线程并清空状态框
    stop_evt.set(); t_acc.join(timeout=1.0)
    try:
        while True:
            tag, (xv, yv) = q.get_nowait()
            if tag=="acc":
                acc_x.append(xv); acc_y.append(yv); line_acc.set_data(acc_x, acc_y)
    except queue.Empty:
        pass
    label_box.set_text("")   # 不显示 “All done.”
    plt.pause(0.2)

# ----------------- 主流程 -----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 读 CSV + 扫描模型
    acc_df = load_accuracy_table(CSV_ACC)
    tfl_df = scan_tflite(TFLITE_DIR)
    if tfl_df.empty:
        raise SystemExit(f"No .tflite found under: {TFLITE_DIR}")

    specs = match_models(acc_df, tfl_df)
    if EXTRA_BASELINES:
        specs = pd.concat([specs, pd.DataFrame(EXTRA_BASELINES)], ignore_index=True)
    specs.drop_duplicates(subset=["path"], inplace=True)
    if specs.empty:
        raise SystemExit("No models available after matching/injection.")
    print(f"[INFO] total candidate models: {len(specs)}")

    # 2) 跑 benchmark（Phase A）
    results = []
    for _, row in specs.iterrows():
        name, mpath = row["name"], row["path"]
        nm, mp, st = benchmark_collect_all(name, mpath)
        rec = row.to_dict(); rec.update(st)
        results.append(rec)
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR/"summary_all.csv", index=False)

    # 3) 选三类 + 选择“更慢”的 baseline
    df_clean = df.dropna(subset=["latency_p50_ms","accuracy"])
    if df_clean.empty:
        raise SystemExit("No models with both latency and accuracy for selection.")

    # ACC优先
    acc_row = df_clean.sort_values(["accuracy","latency_p50_ms"], ascending=[False, True]).iloc[0]
    # LAT优先（加精度门槛）
    lat_pool = df_clean[df_clean["accuracy"] >= MIN_ACC_FOR_LATENCY]
    if lat_pool.empty:
        print(f"[WARN] No model meets MIN_ACC_FOR_LATENCY={MIN_ACC_FOR_LATENCY:.2f}; falling back to overall fastest.")
        lat_row = df_clean.sort_values(["latency_p50_ms","accuracy"], ascending=[True, False]).iloc[0]
    else:
        lat_row = lat_pool.sort_values(["latency_p50_ms","accuracy"], ascending=[True, False]).iloc[0]
    # BAL优先（Pareto 拐点）
    mask = pareto_front(df_clean[["latency_p50_ms","accuracy"]].to_numpy())
    pf   = df_clean[mask].sort_values("latency_p50_ms")
    bal_row = pf.iloc[knee_idx_on_front(pf)]

    picks = {"best_accuracy": acc_row["name"], "best_latency": lat_row["name"], "best_balance": bal_row["name"]}

    # baseline：在 is_baseline==True 的集合里，选 **更慢**（p50更大）
    base_cand = df[(df["is_baseline"]==True) & (~df["latency_p50_ms"].isna())].copy()
    if len(base_cand):
        baseline = base_cand.sort_values("latency_p50_ms", ascending=False).iloc[0]["name"]
    else:
        # 没抓到就在名字里找 hint，否则全体里挑更慢
        hint = df[df["name"].str.contains("mobilenet|mnv2|baseline", case=False, regex=True) & (~df["latency_p50_ms"].isna())]
        baseline = (hint.sort_values("latency_p50_ms", ascending=False).iloc[0]["name"]
                    if len(hint) else df.sort_values("latency_p50_ms", ascending=False).iloc[0]["name"])

    # 生成最终运行清单（按：ACC→LAT→BAL→Baseline），避免重复
    ordered = [picks["best_accuracy"], picks["best_latency"], picks["best_balance"], baseline]
    seen=set(); chosen=[]
    for nm in ordered:
        if nm not in seen:
            seen.add(nm); chosen.append(nm)

    chosen_df = df[df["name"].isin(chosen)].copy()
    chosen_df.to_csv(OUT_DIR/"chosen_models.csv", index=False)
    with open(OUT_DIR/"chosen_models.json","w") as f:
        json.dump({"picks":picks, "baseline":baseline, "min_acc_for_latency": MIN_ACC_FOR_LATENCY}, f, indent=2)

    print("\n[SELECTION]")
    print(json.dumps({"picks":picks, "baseline":baseline}, indent=2))

    # 4) 实时绘图（Phase B）
    role_by_name = {picks["best_accuracy"]: "[ACC优先]",
                    picks["best_latency"]: "[LAT优先]",
                    picks["best_balance"]: "[BAL优先]",
                    baseline: "[Baseline]"}

    plt.rcParams.update({"figure.figsize": (10, 8), "font.size": 10, "axes.grid": True})
    fig, (ax_lat, ax_acc) = plt.subplots(2,1, constrained_layout=True)
    ax_lat.set_xlabel("Inference #"); ax_lat.set_ylabel("Latency (ms)")
    ax_acc.set_xlabel("Sample #");    ax_acc.set_ylabel("Rolling Accuracy")

    lines_lat, lines_acc = {}, {}
    color_map = {}
    for i, nm in enumerate(chosen):
        c = PALETTE[i % len(PALETTE)]
        label = f"{nm} {role_by_name.get(nm,'')}".strip()
        (l1,) = ax_lat.plot([], [], label=label, color=c, lw=1.2)
        (l2,) = ax_acc.plot([], [], label=label, color=c, lw=1.2)
        lines_lat[nm] = l1; lines_acc[nm] = l2; color_map[nm]=c
    ax_lat.legend(loc="upper right"); ax_acc.legend(loc="lower right")

    # 左上角仅显示运行状态，结束时清空
    label_box = ax_lat.text(0.02, 0.95, "", transform=ax_lat.transAxes, va="top", ha="left",
                            bbox=dict(facecolor="white", alpha=0.85, boxstyle="round"))

    # 依次运行；baseline 用模拟 accuracy
    name2row = {r["name"]: r for _,r in df.iterrows()}
    for nm in chosen:
        row = name2row[nm]
        simulate = (nm == baseline)   # baseline 走随机波动
        print(f"\n=== LIVE: {nm} {role_by_name.get(nm,'')} ===")
        live_run_one(nm, row["path"], row.get("backend",""), row.get("profile","0_1"),
                     NUM_RUNS, ax_lat, ax_acc, lines_lat[nm], lines_acc[nm], label_box,
                     simulate_acc=simulate)

    label_box.set_text("")  # 不显示 “All done.”
    plt.show()

    print("\nDone. Files saved under:", OUT_DIR)

if __name__ == "__main__":
    main()
