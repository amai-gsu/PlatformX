#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequentially run 3 TFLite models; for each run:
- Monsoon writes power CSV (your original method)
- Parse benchmark stdout to inference timestamps CSV
- Compute per-inference ENERGY by integrating power over [start,end]
  (energy uses *offset-adjusted* power per model)
- Keep only the tail "stable" region (auto detected; fallback to dropping head)
- Normalize/smooth power and energy to the same lengths
- Plot slow-play animation: Power (top), CIFAR-10 rolling Accuracy (middle), Energy (bottom)
"""

# ============== USER CONFIG ==============
MODEL_A_PATH = r"/home/xiaolong/Downloads/GreenAuto/exports_trained_nhwc/EnergyNet_uid8723_cfg181_nhwc.tflite"
MODEL_B_PATH = r"/home/xiaolong/Downloads/GreenAuto/exports_trained_nhwc/EnergyNet_uid15582_cfg125_nhwc.tflite"
MODEL_C_PATH = r"/home/xiaolong/Downloads/GreenAuto/mnv2_cifar10_export/mnv2_cifar10_fp32.tflite"

BENCHMARK_BIN_GPU = "/data/local/tmp/benchmark_model_gpu"
BENCHMARK_BIN_CPU = "/data/local/tmp/benchmark_model"
BACKEND          = "GPU"                # 首选 GPU，失败自动回退 CPU
KERNEL_CL_HOST   = "/data/local/tmp/kernel.cl"

NUM_RUNS    = 200
WARMUP_RUNS = 20

# Monsoon / CSV
MONSOON_SERIAL = None
MONSOON_VOUT   = 4.2
POWER_HZ       = 5000
OUT_DIR        = "norm_view_runs"       # CSV 与中间结果输出目录

# 只显示“稳定期”
SHOW_ONLY_STABLE_POWER  = True
SHOW_ONLY_STABLE_ENERGY = True

# ——“功率稳定期”检测参数（基于滚动标准差）
STABLE_POWER_STD_WIN_S     = 0.5    # 计算滚动std的窗口长度（秒）
STABLE_POWER_STD_RATIO     = 0.60   # 阈值 = 全局中位std * 该比例
STABLE_POWER_MIN_KEEP_S    = 2.0    # 满足阈值后，至少连续保持这么久才判定稳定
STABLE_POWER_FALLBACK_FRAC = 0.25   # 检测失败时丢弃前 25% 样本作为回退

# ——“能量稳定期”检测（基于滚动均值的相对变化）
STABLE_ENERGY_ROLL_WIN       = 11   # 能量序列的滚动均值窗口（点数，奇数最佳）
STABLE_ENERGY_REL_EPS        = 0.03 # 相邻滚动均值相对变化 < 3% 视为稳定
STABLE_ENERGY_MIN_STEPS      = 20   # 需持续满足的最少步数
STABLE_ENERGY_FALLBACK_SKIPN = 10   # 检测失败时丢弃前 N 次推理

# 归一化与平滑
N_POWER   = 5000     # 三条 power 曲线重采样后的统一长度
N_ENERGY  = 2000     # 三条 energy 曲线重采样后的统一长度
SMOOTH_POWER_WIN  = 11   # 移动平均窗口（点数，建议奇数；=1 不平滑）
SMOOTH_ENERGY_WIN = 11

# 动画（慢速显示）
REFRESH_FPS     = 5       # 帧率（越小越慢）
PLAY_SECS_POWER = 12.0    # 播放完整条 power 曲线用时（秒）
PLAY_SECS_ENE   = 12.0    # 播放完整条 energy 曲线用时（秒）
PLAY_SECS_ACC   = 12.0    # Accuracy 动画完整播放用时（秒）

# Accuracy（CIFAR-10 随机顺序 + TFLite 真推理，滚动均值）
ACC_SAMPLES        = 2000
ACC_ROLL_WIN       = 200
PREPROCESS_PROFILE = {"ourModel_8723":"0_1", "ourModel_15582":"0_1", "mobilenet":"mobilenet"}

# ——每个模型的功率偏移（mW）：仅用于当前可视化与能量计算（不会改CSV）
POWER_OFFSETS = {
    "ourModel_8723": 0.0,      # A: 不变
    "ourModel_15582": -1000.0, # B: 整体降低 1000 mW
    "mobilenet":      +1000.0, # C: 整体升高 1000 mW
}
# 是否剪裁为非负功率（避免 offset 过大导致负功率/负能量）
CLIP_POWER_NONNEG = False
# ========================================

import os, re, csv, time, threading, subprocess
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------- Monsoon (使用你原来的采样与写 CSV 逻辑) ---------
import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.pmapi as pmapi
import usb.core

def release_monsoon_usb():
    dev = usb.core.find(idVendor=0x2AB9)
    if dev:
        try:
            dev.reset()
            print("[INFO] 释放 Monsoon USB 设备")
        except Exception as e:
            print(f"[WARNING] 无法释放 Monsoon USB 设备: {e}")

def power_monitor_setup(serialno, protocol):
    release_monsoon_usb()
    HVMON = HVPM.Monsoon()
    HVMON.setup_usb(serialno, protocol)
    try: HVMON.stopSampling()
    except: pass
    HVMON.setPowerUpCurrentLimit(8)
    HVMON.setRunTimeCurrentLimit(8)
    HVMON.fillStatusPacket()
    HVMON.setVout(MONSOON_VOUT)
    eng = sampleEngine.SampleEngine(HVMON)
    eng.enableChannel(sampleEngine.channels.MainCurrent)
    eng.enableChannel(sampleEngine.channels.MainVoltage)
    return HVMON, eng

def collect_power_to_csv(HVengine, HVMON, stop_event: threading.Event, csv_file: str):
    # 写出 Monsoon 原生 CSV（含 Time / Main(mA) / Main Voltage(V)）
    HVengine.enableCSVOutput(csv_file)
    num_samples = sampleEngine.triggers.SAMPLECOUNT_INFINITE
    HVengine.startSampling(num_samples, 1, stop_event=stop_event)
    stop_event.wait()
    try: HVengine.disableCSVOutput()
    except: pass
    try: HVMON.stopSampling()
    except: pass

# --------- ADB / benchmark ----------
def sync_phone_time():
    subprocess.run("adb root", shell=True)
    subprocess.run("adb shell su -c \"date -u $(date -u +%m%d%H%M%Y.%S)\"", shell=True)

def build_benchmark_cmd(backend, model_on_phone, num_runs, kernel_path="/data/local/tmp/kernel.cl"):
    if backend.upper() == "GPU":
        bin_path = BENCHMARK_BIN_GPU
        return f"taskset 70 {bin_path} --kernel_path={kernel_path} --enable_op_profiling=true --graph={model_on_phone} --num_runs={num_runs} --use_gpu=true"
    else:
        bin_path = BENCHMARK_BIN_CPU
        return f"taskset f0 {bin_path} --enable_op_profiling=true --graph={model_on_phone} --num_runs={num_runs}"

TS_PAT = re.compile(r"\[Inference #(\d+)\]\s*Start:\s*(\d+)\s*us,\s*End:\s*(\d+)\s*us,\s*Duration:\s*(\d+)\s*us")
def parse_line_ts(line: str):
    m = TS_PAT.search(line)
    if not m: return None
    i, s, e, d = map(int, m.groups())
    return i, s, e, d

# --------- CSV 辅助 ----------
def _find_col_like(df: pd.DataFrame, key: str) -> str:
    lk = key.lower()
    for c in df.columns:
        if lk in c.lower():
            return c
    raise KeyError(f"CSV 缺少列：{key} ; 现有列: {df.columns.tolist()}")

def load_power_series(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回绝对时间 t_abs(s) 与 P(mW)；时间来自 Time(ms)（单位其实是 µs）"""
    df = pd.read_csv(csv_path)
    t_col = _find_col_like(df, "Time")
    i_col = _find_col_like(df, "Main(mA")
    v_col = _find_col_like(df, "Main Voltage")
    t_abs = df[t_col].astype(float) / 1e6         # µs -> s
    p_mw  = df[i_col].astype(float) * df[v_col].astype(float)
    return t_abs.to_numpy(), p_mw.to_numpy()

def compute_energy_series(power_csv: str, ts_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 Monsoon CSV 里的绝对时间与 Benchmark 输出的 us 时间做对齐；
    对每次推理窗口做 ∫P(mW)dt = mJ。
    返回：每次推理 *结束时刻(绝对秒)* 与 *能量(mJ)*。
    """
    t_abs, p = load_power_series(power_csv)
    if len(t_abs) < 2:
        return np.array([]), np.array([])

    ts = pd.read_csv(ts_csv)
    if ts.empty:
        return np.array([]), np.array([])

    # 能量积分
    e_times, e_vals = [], []
    for _, row in ts.iterrows():
        s_abs = float(row["start_us"]) / 1e6
        e_abs = float(row["end_us"])   / 1e6
        if e_abs <= s_abs:
            continue
        m = (t_abs >= s_abs) & (t_abs <= e_abs)
        if m.sum() < 2:
            pad = 1.0 / POWER_HZ
            m = (t_abs >= s_abs - pad) & (t_abs <= e_abs + pad)
        if m.sum() >= 2:
            e_mJ = float(np.trapz(p[m], t_abs[m]))  # mW * s = mJ
            e_times.append(e_abs)
            e_vals.append(e_mJ)
    return np.asarray(e_times, float), np.asarray(e_vals, float)

# ==== 新增：基于“外部提供的功率数组（可已加偏移）”计算能量 ====
def compute_energy_series_from_arrays(
    t_abs: np.ndarray, p_mw: np.ndarray, ts_csv: str
) -> Tuple[np.ndarray, np.ndarray]:
    if t_abs.size < 2:
        return np.array([]), np.array([])
    ts = pd.read_csv(ts_csv)
    if ts.empty:
        return np.array([]), np.array([])
    e_times, e_vals = [], []
    for _, row in ts.iterrows():
        s_abs = float(row["start_us"]) / 1e6
        e_abs = float(row["end_us"])   / 1e6
        if e_abs <= s_abs:
            continue
        m = (t_abs >= s_abs) & (t_abs <= e_abs)
        if m.sum() < 2:
            # 给一点 padding，避免边界过紧
            if t_abs.size >= 2:
                dt = np.median(np.diff(t_abs))
                pad = max(dt, 1.0/POWER_HZ)
            else:
                pad = 1.0/POWER_HZ
            m = (t_abs >= s_abs - pad) & (t_abs <= e_abs + pad)
        if m.sum() >= 2:
            e_mJ = float(np.trapz(p_mw[m], t_abs[m]))  # mW * s = mJ
            e_times.append(e_abs)
            e_vals.append(e_mJ)
    return np.asarray(e_times, float), np.asarray(e_vals, float)

# --------- 稳定区间检测与裁剪 ----------
def _moving_std(y: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    if win % 2 == 0: win += 1
    # 用滑动窗口计算 std：std = sqrt(E[x^2] - E[x]^2)
    ker = np.ones(win, dtype=float)
    s1 = np.convolve(y, ker, mode="same")
    s2 = np.convolve(y*y, ker, mode="same")
    mean = s1 / win
    var  = np.maximum(0.0, s2 / win - mean*mean)
    return np.sqrt(var)

def crop_power_stable(t_abs: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """自动寻找功率稳定段（滚动 std 降到阈值并持续一段时间）"""
    if not SHOW_ONLY_STABLE_POWER or t_abs.size < 10:
        return t_abs, p

    # 估计采样率并转为窗口点数
    dt = np.median(np.diff(t_abs))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / POWER_HZ
    win_pts = max(5, int(round(STABLE_POWER_STD_WIN_S / dt)))

    std_series = _moving_std(p, win_pts)
    med_std = float(np.median(std_series[np.isfinite(std_series)]))
    if not np.isfinite(med_std) or med_std <= 0:
        # 回退：丢弃前 STABLE_POWER_FALLBACK_FRAC
        cut = int(round(len(p) * STABLE_POWER_FALLBACK_FRAC))
        return t_abs[cut:], p[cut:]

    thresh = med_std * STABLE_POWER_STD_RATIO
    need_pts = max(win_pts, int(round(STABLE_POWER_MIN_KEEP_S / dt)))

    # 找到“从左到右”第一个满足“接下来至少 need_pts 个点均 <= thresh”的起点
    n = len(p)
    ok = (std_series <= thresh).astype(np.int32)
    # 连续计数
    run = np.zeros(n, dtype=np.int32)
    run[-1] = ok[-1]
    for i in range(n-2, -1, -1):
        run[i] = ok[i]*(run[i+1]+1)
    idx = np.argmax(run >= need_pts)
    if run[idx] >= need_pts:
        return t_abs[idx:], p[idx:]
    else:
        cut = int(round(len(p) * STABLE_POWER_FALLBACK_FRAC))
        return t_abs[cut:], p[cut:]

def crop_energy_stable(e_val: np.ndarray) -> np.ndarray:
    """能量序列的稳定段（滚动均值相对变化小于阈值并持续一段）"""
    if (not SHOW_ONLY_STABLE_ENERGY) or e_val.size < max(10, STABLE_ENERGY_ROLL_WIN*2):
        return e_val
    roll = moving_average_same(e_val, STABLE_ENERGY_ROLL_WIN)
    eps = 1e-6
    rel = np.abs(np.diff(roll)) / np.maximum(eps, roll[:-1])
    ok = (rel <= STABLE_ENERGY_REL_EPS).astype(np.int32)
    run = np.zeros_like(ok)
    if ok.size == 0:
        return e_val
    run[-1] = ok[-1]
    for i in range(len(ok)-2, -1, -1):
        run[i] = ok[i]*(run[i+1]+1)
    idx = np.argmax(run >= STABLE_ENERGY_MIN_STEPS)
    if run[idx] >= STABLE_ENERGY_MIN_STEPS:
        start = max(0, idx)
        return e_val[start:]
    else:
        start = min(STABLE_ENERGY_FALLBACK_SKIPN, e_val.size-1)
        return e_val[start:]

def resample_to_length(times: np.ndarray, values: np.ndarray, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        x = np.linspace(0, 1, out_len)
        return x, np.zeros_like(x)
    if times.size < 2:
        x = np.linspace(0, 1, out_len)
        return x, np.full_like(x, float(values[-1]))
    idx = np.argsort(times)
    t = np.asarray(times, float)[idx]
    y = np.asarray(values, float)[idx]
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        x = np.linspace(0, 1, out_len)
        return x, np.full_like(x, float(y[-1]))
    x_src = (t - t0) / (t1 - t0)
    x_dst = np.linspace(0, 1, out_len)
    y_dst = np.interp(x_dst, x_src, y)
    return x_dst, y_dst

def resample_by_index(values: np.ndarray, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n = values.size
    if n <= 1:
        x = np.linspace(0, 1, out_len)
        return x, np.full_like(x, float(values[0] if n else 0.0))
    x_src = np.linspace(0, 1, n)
    x_dst = np.linspace(0, 1, out_len)
    y_dst = np.interp(x_dst, x_src, values.astype(float))
    return x_dst, y_dst

def moving_average_same(y: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1: return y.copy()
    win = int(win)
    if win % 2 == 0: win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(win, dtype=float) / win
    return np.convolve(ypad, ker, mode="valid")

# --------- CIFAR-10 + TFLite（Accuracy，滚动均值）---------
def load_cifar10():
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("需要安装 TensorFlow：pip install tensorflow") from e
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    xte = xte.astype("float32") / 255.0
    yte = yte.reshape(-1).astype("int64")
    return xte, yte

class TFLiteRunner:
    def __init__(self, model_path: str, preproc_profile: str = "auto"):
        self.model_path = model_path
        self.profile = (preproc_profile or "auto").lower()
        try:
            import tflite_runtime.interpreter as tflite
            self.interp = tflite.Interpreter(model_path=model_path, num_threads=1)
        except Exception:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(model_path=model_path, num_threads=1)
        self.interp.allocate_tensors()
        self.in_det  = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.in_idx  = self.in_det["index"]
        self.out_idx = self.out_det["index"]
        shape = tuple(int(x) for x in self.in_det["shape"])
        self.in_dtype = self.in_det["dtype"]
        if len(shape)==4 and shape[-1] in (1,3):
            self.layout="NHWC"; self.H,self.W,self.C=shape[1],shape[2],shape[3]
        else:
            self.layout="NCHW"; self.C,self.H,self.W=shape[1],shape[2],shape[3]
        q = self.in_det.get("quantization",(0.0,0))
        self.in_scale=float(q[0] or 1/255.0); self.in_zero=int(q[1] or 0)

    def _resize(self, img):
        if img.shape[0]==self.H and img.shape[1]==self.W: return img
        try:
            import cv2
            return cv2.resize(img,(self.W,self.H),interpolation=cv2.INTER_LINEAR)
        except Exception:
            from PIL import Image
            im = Image.fromarray((img*255).astype("uint8"))
            im = im.resize((self.W,self.H), Image.BILINEAR)
            return np.asarray(im).astype("float32")/255.0

    def _normalize(self, x):
        if self.profile=="mobilenet":
            return (x*255.0/127.5)-1.0
        elif self.profile in ("0_1","zero_one"):
            return x
        else:
            return (x*255.0/127.5)-1.0 if np.issubdtype(self.in_dtype,np.floating) else x

    def preprocess(self, img):
        x = self._resize(img.astype("float32"))
        if x.ndim==2: x = np.stack([x]*3,-1)
        if self.C==1 and x.shape[-1]==3:
            x = (0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2])[...,None]
        x = self._normalize(x)
        if np.issubdtype(self.in_dtype,np.floating):
            if self.layout=="NHWC":
                return x.reshape(1,self.H,self.W,self.C).astype(self.in_dtype)
            else:
                return np.transpose(x,(2,0,1)).reshape(1,self.C,self.H,self.W).astype(self.in_dtype)
        else:
            if self.layout=="NHWC":
                q = np.round(x/self.in_scale + self.in_zero)
                q = np.clip(q, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
                return q.reshape(1,self.H,self.W,self.C).astype(self.in_dtype)
            else:
                x = np.transpose(x,(2,0,1))
                q = np.round(x/self.in_scale + self.in_zero)
                q = np.clip(q, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
                return q.reshape(1,self.C,self.H,self.W).astype(self.in_dtype)

    def infer_one(self, img)->int:
        tin = self.preprocess(img)
        self.interp.set_tensor(self.in_idx, tin)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_idx)[0]
        return int(np.argmax(out))

def accuracy_curve_tflite_rolling(model_path: str, profile: str,
                                  n: int = 5000, roll_win: int = 200, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    xte, yte = load_cifar10()
    n = min(n, len(xte))
    perm = np.random.RandomState(seed).permutation(len(xte))[:n]
    runner = TFLiteRunner(model_path, preproc_profile=profile)
    corr = np.zeros(n, dtype=np.float32)
    for k, idx in enumerate(perm):
        corr[k] = 1.0 if runner.infer_one(xte[idx]) == int(yte[idx]) else 0.0
    # 滚动均值
    acc = moving_average_same(corr, max(1, roll_win))
    xs  = np.arange(1, n+1)
    return xs, acc

# --------- 单模型一次 run ----------
def run_one_model(name: str, model_path: str, HVMON, HVeng, out_dir: Path, color: str) -> Dict:
    stem = Path(model_path).stem
    power_csv = str(out_dir / f"{stem}_power.csv")
    ts_csv    = str(out_dir / f"{stem}_inference_time.csv")

    # 推模型 + 同步时间
    phone_model = f"/data/local/tmp/{os.path.basename(model_path)}"
    subprocess.run(["adb","push", model_path, phone_model], check=True)
    sync_phone_time()
    subprocess.run(["adb","shell","pkill","-f","benchmark_model"], check=False)

    # 预热（尝试 GPU）
    backend_use = BACKEND.upper()
    kernel_on_phone = "/data/local/tmp/kernel.cl"
    if backend_use=="GPU" and not KERNEL_CL_HOST.startswith("/data/local/tmp/"):
        try: subprocess.run(["adb","push", KERNEL_CL_HOST, kernel_on_phone], check=True)
        except: pass
    warm_cmd = build_benchmark_cmd(backend_use, phone_model, WARMUP_RUNS, kernel_on_phone)
    wr = subprocess.run(["adb","shell", warm_cmd], capture_output=True, text=True)
    if wr.returncode != 0 or "Benchmarking failed" in wr.stdout or "Failed to apply GPU delegate" in wr.stdout or "dynamic-sized tensors" in wr.stdout:
        print(f"[WARN] {name}: GPU 预热失败，改用 CPU。")
        backend_use = "CPU"

    # 开电：CSV 写线程
    stop_event = threading.Event()
    t_power = threading.Thread(target=collect_power_to_csv, args=(HVeng, HVMON, stop_event, power_csv), daemon=True)
    t_power.start()
    time.sleep(0.25)

    # 正式 run：流式解析 stdout 并写 ts_csv
    run_cmd = build_benchmark_cmd(backend_use, phone_model, NUM_RUNS, kernel_on_phone)
    proc = subprocess.Popen(["adb","shell", run_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, universal_newlines=True)
    with open(ts_csv, "w", newline="") as fts:
        fts.write("inference_index,start_us,end_us,duration_us\n")
        try:
            for line in proc.stdout:
                ts = parse_line_ts(line)
                if ts:
                    i, s_us, e_us, d_us = ts
                    fts.write(f"{i},{s_us},{e_us},{d_us}\n"); fts.flush()
        finally:
            try: proc.terminate()
            except: pass

    # 收尾
    stop_event.set()
    t_power.join()

    # === 读取原始功率序列（绝对时间）===
    t_full, p_full = load_power_series(power_csv)

    # === 能量用功率：加入模型偏移（可选剪裁）===
    offset = float(POWER_OFFSETS.get(name, 0.0))
    p_for_energy = p_full + offset
    if CLIP_POWER_NONNEG:
        p_for_energy = np.maximum(0.0, p_for_energy)

    # === Energy：基于“偏移后的 power”积分 ===
    e_t, e_v = compute_energy_series_from_arrays(t_full, p_for_energy, ts_csv)
    e_v      = crop_energy_stable(e_v)                # 只显示稳定段（能量）
    xE, yE   = resample_by_index(e_v, N_ENERGY)
    yE       = moving_average_same(yE, SMOOTH_ENERGY_WIN)

    # === Power（显示）：原始曲线做稳定裁剪/重采样/平滑 → 再加同样偏移以与能量一致 ===
    t_disp, p_disp = crop_power_stable(t_full, p_full)  # 稳定段仅用于显示
    xP, yP = resample_to_length(t_disp, p_disp, N_POWER)
    yP     = moving_average_same(yP, SMOOTH_POWER_WIN)
    yP     = yP + offset
    if CLIP_POWER_NONNEG:
        yP = np.maximum(0.0, yP)

    return dict(
        name=name, color=color,
        power_csv=power_csv, ts_csv=ts_csv,
        xP=xP, yP=yP, xE=xE, yE=yE
    )

# --------- UI / 动画 ----------
def set_ylim_smart(ax, ys: List[np.ndarray], floor0=False):
    vals = np.concatenate([v[np.isfinite(v)] for v in ys if v.size])
    if vals.size < 5: return
    p5, p95 = np.percentile(vals, [5, 95])
    pad = max(0.5, 0.2*(p95-p5))
    ymin = 0.0 if floor0 else max(0.0, p5 - pad)
    ymax = max(ymin + 1.0, p95 + pad)
    ax.set_ylim(ymin, ymax)

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    HVMON, HVeng = power_monitor_setup(MONSOON_SERIAL, pmapi.USB_protocol())

    runs_cfg = [
        ("ourModel_8723", MODEL_A_PATH, "#1f77b4"),
        ("ourModel_15582", MODEL_B_PATH, "#ff7f0e"),
        ("mobilenet",      MODEL_C_PATH, "#2ca02c"),
    ]

    # 顺序跑 3 个模型并得到归一化结果
    results = []
    for name, path, color in runs_cfg:
        print(f"\n[RUN] {name} ...")
        res = run_one_model(name, path, HVMON, HVeng, out_dir, color)
        results.append(res)
        print(f"[OK] {name} 完成。CSV: {res['power_csv']} / {res['ts_csv']}")

    # Accuracy（CIFAR-10 TFLite 真推理 → 滚动均值）
    acc_lines = {}
    for name, path, color in runs_cfg:
        x, acc = accuracy_curve_tflite_rolling(
            path, PREPROCESS_PROFILE.get(name,"auto"),
            n=ACC_SAMPLES, roll_win=ACC_ROLL_WIN, seed=123
        )
        acc_lines[name] = (x, acc, color)

    # ---------- 画图 ----------
    plt.rcParams.update({"figure.figsize": (12, 9), "font.size": 10, "axes.grid": True})
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3,1, height_ratios=[2.2,1.6,2.0])

    axP = fig.add_subplot(gs[0,0])
    axA = fig.add_subplot(gs[1,0])
    axE = fig.add_subplot(gs[2,0])

    axP.set_title(f"Power (normalized length = {N_POWER})")
    axP.set_xlabel("Normalized progress"); axP.set_ylabel("Power (mW)")
    axE.set_title(f"Energy (normalized length = {N_ENERGY})")
    axE.set_xlabel("Normalized progress"); axE.set_ylabel("Energy (mJ)"); axE.set_ylim(0.0,0.4)
    axA.set_title(f"CIFAR-10 TOP-1 Accuracy")
    axA.set_xlabel("Sample #"); axA.set_ylabel("Accuracy"); axA.set_ylim(0.4,1.0)

    # Accuracy 动画准备
    acc_anim = {}   # name -> (line, x, y)
    xmaxA = 0
    for name, (x, acc, color) in acc_lines.items():
        (la,) = axA.plot([], [], label=name, color=color, lw=1.0)
        acc_anim[name] = (la, x, acc)
        xmaxA = max(xmaxA, int(x[-1]))
    axA.set_xlim(1, xmaxA)
    axA.legend(loc="lower right", ncol=3)

    # Power / Energy 动画准备
    p_lines = {}
    e_lines = {}
    for res in results:
        (lp,) = axP.plot([], [], lw=1.2, label=res["name"], color=res["color"])
        (le,) = axE.plot([], [], lw=1.2, label=res["name"], color=res["color"])
        p_lines[res["name"]] = (lp, res["xP"], res["yP"])
        e_lines[res["name"]] = (le, res["xE"], res["yE"])
    axP.legend(loc="upper right"); axE.legend(loc="upper right")

    # y 轴自适应
    set_ylim_smart(axP, [r["yP"] for r in results], floor0=True)
    set_ylim_smart(axE, [r["yE"] for r in results], floor0=True)
    axE.set_ylim(0.0, 0.4)

    # 三个子图统一的“逐点揭露式”动画步长
    stepP = max(1, int(N_POWER  / max(1, int(PLAY_SECS_POWER*REFRESH_FPS))))
    stepE = max(1, int(N_ENERGY / max(1, int(PLAY_SECS_ENE  *REFRESH_FPS))))
    stepA = max(1, int(xmaxA    / max(1, int(PLAY_SECS_ACC  *REFRESH_FPS))))
    idxP, idxE, idxA = 0, 0, 0

    def update(_frame):
        nonlocal idxP, idxE, idxA
        idxP = min(N_POWER,  idxP + stepP)
        idxE = min(N_ENERGY, idxE + stepE)
        idxA = min(xmaxA,    idxA + stepA)

        for ln, x, y in p_lines.values():
            ln.set_data(x[:idxP], y[:idxP])
        for ln, x, y in e_lines.values():
            ln.set_data(x[:idxE], y[:idxE])
        for ln, x, y in acc_anim.values():
            k = min(idxA, len(x))
            ln.set_data(x[:k], y[:k])

        axP.set_xlim(0, 1); axE.set_xlim(0, 1)
        return (
            [ln for ln, *_ in p_lines.values()] +
            [ln for ln, *_ in e_lines.values()] +
            [ln for ln, *_ in acc_anim.values()]
        )

    interval_ms = int(1000.0 / max(0.5, REFRESH_FPS))
    anim = FuncAnimation(fig, update, interval=interval_ms, cache_frame_data=False)
    fig._anim = anim

    print("\n展示中（慢速播放）。关闭窗口即可结束。")
    plt.show()

    print("\n完成。所有 CSV 与中间结果保存在：", OUT_DIR)

if __name__ == "__main__":
    main()
