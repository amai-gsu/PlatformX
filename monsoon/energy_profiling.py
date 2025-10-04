import time
import csv
import os
import threading
from datetime import datetime
import subprocess

import Monsoon.HVPM as HVPM
import Monsoon.sampleEngine as sampleEngine
import Monsoon.pmapi as pmapi
import usb.core

import re
import pandas as pd
from pathlib import Path
import json

# ---------------- Monsoon Setup ---------------- #
def release_monsoon_usb():
    """ 释放 Monsoon 设备占用的 USB 资源 """
    dev = usb.core.find(idVendor=0x2AB9)  # Monsoon 的 USB Vendor ID
    if dev:
        try:
            dev.reset()  # 释放 USB 设备
            print("[INFO] 释放 Monsoon USB 设备")
        except Exception as e:
            print(f"[WARNING] 无法释放 Monsoon USB 设备: {e}")

# **Monsoon Power Monitor 初始化**
def power_monitor_setup(serialno, protocol):
    release_monsoon_usb()  # 确保不会有 USB 设备占用问题
    HVMON = HVPM.Monsoon()
    HVMON.setup_usb(serialno, protocol)
    try:
        HVMON.stopSampling()
    except Exception as e:
        print(f"Error stopping previous sampling: {e}")
    HVMON.setPowerUpCurrentLimit(8)
    HVMON.setRunTimeCurrentLimit(8)
    HVMON.fillStatusPacket()
    HVMON.setVout(4.2)
    HVengine = sampleEngine.SampleEngine(HVMON)
    HVengine.enableChannel(sampleEngine.channels.MainCurrent)
    HVengine.enableChannel(sampleEngine.channels.MainVoltage)
    return HVMON, HVengine

# ---------------- Power Collection ---------------- #
def collect_power(HVengine, HVMON, stop_event, csv_file):
    start_server_time = time.time()
    print(f"[INFO] Power collection started at server time: {start_server_time:.6f}")

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        HVengine.enableCSVOutput(csv_file)
        num_samples = sampleEngine.triggers.SAMPLECOUNT_INFINITE
        HVengine.startSampling(num_samples, 1, stop_event=stop_event)

        while not stop_event.is_set():
            try:
                power_samples = HVengine.getSamples()
                timestamp = time.time()  # 当前服务器时间戳
                for sample in power_samples:
                    voltage = sample[1]  # V
                    current = sample[2]  # mA
                    power_mw = voltage * current
                    writer.writerow([timestamp, round(power_mw, 3)])
            except Exception as e:
                print(f"Error getting power samples: {e}")
            time.sleep(0.1)

        HVengine.disableCSVOutput()
        HVMON.stopSampling()

# ---------------- CPU Frequency Lock ---------------- #
def run_adb_shell(cmd):
    return subprocess.run(["adb", "shell", cmd], capture_output=True, text=True).stdout.strip()

def run_adb_su(cmd):
    return subprocess.run(["adb", "shell", f"su -c \"{cmd}\""], capture_output=True, text=True).stdout.strip()

def get_all_cpu_ids():
    output = run_adb_shell("cat /sys/devices/system/cpu/possible").replace("\r", "")
    if "-" in output:
        start, end = map(int, output.split("-"))
        return list(range(start, end + 1))
    return [0]

def sync_phone_time():
    subprocess.run("adb root", shell=True)
    subprocess.run("adb shell su -c \"date -u $(date -u +%m%d%H%M%Y.%S)\"", shell=True)

def parse_inference_timestamps(stdout_text: str, model_name: str, out_dir: str, backend: str):
    """
    统一解析 CPU/GPU 的推理时间：
    - 格式A: [Inference #i] Start: <us>, End: <us>, Duration: <us>
    - 格式B: 行内仅包含 16+ 位数字，按 (start,end) 成对出现
    输出: {out_dir}/{model_name}_inference_time.csv
    """
    path = os.path.join(out_dir, f"{model_name}_inference_time.csv")
    pat_A = re.compile(r"\[Inference #(\d+)\]\s*Start:\s*(\d+)\s*us,\s*End:\s*(\d+)\s*us,\s*Duration:\s*(\d+)\s*us")
    matches_A = [m.groups() for m in pat_A.finditer(stdout_text)]

    rows = []
    if matches_A:
        for idx, s, e, d in matches_A:
            rows.append((int(idx), int(s), int(e), int(d)))
    else:
        # 尝试格式B：只含超长数字的行
        nums = []
        for line in stdout_text.splitlines():
            line = line.strip()
            if re.fullmatch(r"\d{16,}", line):
                nums.append(int(line))
        for i in range(0, len(nums) - 1, 2):
            s, e = nums[i], nums[i+1]
            rows.append((i//2, s, e, e - s))

    with open(path, "w") as f:
        f.write("inference_index,start_us,end_us,duration_us\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    print(f"Inference timestamps saved to {path} (backend={backend}, count={len(rows)})")
    return len(rows) > 0

def build_benchmark_cmd(backend, benchmark_bin, model_on_phone, num_runs,
                        kernel_path="/data/local/tmp/kernel.cl"):
    """
    backend       : "CPU" 或 "GPU"
    benchmark_bin : 手机端可执行文件路径（如 /data/local/tmp/benchmark_model_gpu）
    model_on_phone: /data/local/tmp/xxx.tflite
    num_runs      : 运行次数
    kernel_path   : 仅 GPU 需要
    """
    if backend.upper() == "GPU":
        # GPU：taskset 0x70，带 kernel_path 与 --use_gpu=true
        return (f"taskset 70 {benchmark_bin} "
                f"--kernel_path={kernel_path} "
                f"--enable_op_profiling=true "
                f"--graph={model_on_phone} "
                f"--num_runs={num_runs} "
                f"--use_gpu=true")
    else:
        # CPU：taskset 0xf0，无 GPU 参数
        return (f"taskset f0 {benchmark_bin} "
                f"--enable_op_profiling=true "
                f"--graph={model_on_phone} "
                f"--num_runs={num_runs}")

# ---------------- Inference Function ---------------- #
DEVICE_KERNEL_PATH = "/data/local/tmp/kernel.cl"

def build_benchmark_cmd(
    backend: str,
    benchmark_bin: str,
    model_on_phone: str,
    num_runs: int,
    kernel_path: str = None,
) -> str:
    """
    严格按需构造命令：
      - GPU: taskset 70 <bin> --kernel_path=/data/local/tmp/kernel.cl --enable_op_profiling=true --graph=<model> --num_runs=N --use_gpu=true
      - CPU: taskset f0 <bin> --enable_op_profiling=true --graph=<model> --num_runs=N
    """
    backend = (backend or "CPU").upper()
    if backend == "GPU":
        return (
            f"taskset 70 {benchmark_bin} "
            f"--kernel_path={DEVICE_KERNEL_PATH} "
            f"--enable_op_profiling=true "
            f"--graph={model_on_phone} "
            f"--num_runs={int(num_runs)} "
            f"--use_gpu=true"
        )
    else:
        return (
            f"taskset f0 {benchmark_bin} "
            f"--enable_op_profiling=true "
            f"--graph={model_on_phone} "
            f"--num_runs={int(num_runs)}"
        )


def run_inference(HVMON, HVengine,
                  model_path,
                  power_csv,
                  benchmark_bin,
                  num_runs=10,
                  warmup_runs=10,
                  backend="GPU",
                  kernel_path="/data/local/tmp/kernel.cl"):
    """
    push TFLite → 同步时间 → pkill → warmup（不开电）→ 开电采样 → 正式 run
    * GPU: 强制使用 taskset 70，并传 --kernel_path=/data/local/tmp/kernel.cl、--use_gpu=true
    * CPU: 强制使用 taskset f0，不传 kernel_path / use_gpu
    """
    backend = (backend or "CPU").upper()
    model_basename = os.path.basename(model_path)
    model_on_phone = f"/data/local/tmp/{model_basename}"
    out_dir = os.path.dirname(power_csv)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 推模型到手机
    subprocess.run(["adb", "push", model_path, "/data/local/tmp/"], check=True)

    # 2) 同步时间 & 清理旧进程
    try:
        sync_phone_time()
    except Exception as e:
        print(f"[WARN] sync_phone_time failed: {e}")
    subprocess.run(["adb", "shell", "pkill", "-f", "benchmark_model"], check=False)
    subprocess.run(["adb", "shell", f"chmod +x {benchmark_bin}"], check=False)

    # 3) GPU 需要确保 kernel.cl 在设备端
    if backend == "GPU":
        # 若传入的是主机路径，push 到固定设备路径；若已是设备端路径，则确保存在
        if not kernel_path.startswith("/data/local/tmp/"):
            subprocess.run(["adb", "push", kernel_path, DEVICE_KERNEL_PATH], check=True)
        else:
            # 设备端自带路径，仍统一放到固定路径，便于命令使用
            if kernel_path != DEVICE_KERNEL_PATH:
                subprocess.run(["adb", "shell", f"cp {kernel_path} {DEVICE_KERNEL_PATH}"], check=False)

    # 4) Warmup（不开电）
    warmup_cmd = build_benchmark_cmd(
        backend=backend,
        benchmark_bin=benchmark_bin,
        model_on_phone=model_on_phone,
        num_runs=warmup_runs,
        kernel_path=DEVICE_KERNEL_PATH if backend == "GPU" else None,
    )
    warmup_res = subprocess.run(["adb", "shell", warmup_cmd], capture_output=True, text=True)
    print(f"[warmup] rc={warmup_res.returncode}")
    if warmup_res.stdout:
        print("[warmup][stdout]\n" + warmup_res.stdout)
    if warmup_res.stderr:
        print("[warmup][stderr]\n" + warmup_res.stderr)
    if warmup_res.returncode != 0:
        print(f"[WARN] Warmup failed (backend={backend}, rc={warmup_res.returncode}). Skip.")
        return None

    # 5) 开始采电
    stop_event = threading.Event()
    power_thread = threading.Thread(target=collect_power, args=(HVengine, HVMON, stop_event, power_csv))
    power_thread.start()
    time.sleep(0.05)

    # 6) 正式 run（开电）
    run_cmd = build_benchmark_cmd(
        backend=backend,
        benchmark_bin=benchmark_bin,
        model_on_phone=model_on_phone,
        num_runs=num_runs,
        kernel_path=DEVICE_KERNEL_PATH if backend == "GPU" else None,
    )
    result = subprocess.run(["adb", "shell", run_cmd], capture_output=True, text=True)

    # 7) 收尾
    stop_event.set()
    power_thread.join()

    if result.returncode != 0:
        print(f"[WARN] Inference failed (backend={backend}, rc={result.returncode}).")
        if result.stdout:
            print("[run][stdout]\n" + result.stdout)
        if result.stderr:
            print("[run][stderr]\n" + result.stderr)
        return None

    # 8) 解析输出（沿用你的解析器）
    try:
        model_name = model_basename.replace(".tflite", "")
        ok = parse_inference_timestamps(result.stdout, model_name, out_dir, backend=backend)
    except Exception as e:
        print(f"[WARN] parse_gpu_benchmark_output exception: {e}")
        ok = False

    if not ok:
        print(f"[WARN] No inference timestamps parsed (backend={backend}).")
        return None

    print("[OK] Inference done & timestamps parsed.")
    return True
# ---------------- Energy Calculation ---------------- #

def compute_energy(power_csv, inference_csv):
    power_df = pd.read_csv(power_csv)
    power_df['Time'] = power_df['Time(ms)'] / 1e6  # from µs to seconds
    power_start = power_df['Time'].min()
    power_end = power_df['Time'].max()

    inference_df = pd.read_csv(inference_csv)
    inference_df['start_s'] = inference_df['start_us'] / 1e6
    inference_df['end_s'] = inference_df['end_us'] / 1e6

    total_energy = 0.0
    valid_inference_count = 0

    for _, row in inference_df.iterrows():
        start_time = row['start_s']
        end_time = row['end_s']

        if start_time < power_start or end_time > power_end:
            continue

        mask = (power_df['Time'] >= start_time) & (power_df['Time'] <= end_time)
        segment = power_df[mask]

        dt = 1 / 5000
        energy_mJ = (segment['Main(mA)'] * segment['Main Voltage(V)'] * dt).sum()

        total_energy += energy_mJ
        valid_inference_count += 1

    print(f"Total Energy: {total_energy:.4f} mJ")
    print(f"Valid Inferences Count: {valid_inference_count}")

    return total_energy


# ---------------- Config readers (fixed 5 keys) ---------------- #
CANONICAL_KEYS = ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"]

def extract_kernel_id(base_name: str):
    m = re.match(r"^.+_test_([A-Za-z0-9]+)$", base_name)
    return m.group(1) if m else None

def kernel_type_base_from_name(base: str) -> str:
    # 例如 fc_test_XXXX -> fc ; conv-block_test_YYYY -> conv-block
    return base.split("_test_")[0].lower()

def _load_kernel_json(super_type: str, kernel_type_base: str, results_root: Path):
    # 期望路径：results/<SuperType>/<kernel_type_base>_test.json
    candidates = [
        results_root / super_type          / f"{kernel_type_base}_test.json",
        results_root / super_type.lower()  / f"{kernel_type_base}_test.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f), str(p)
    return None, None

def _find_config_in_json(js: dict, kernel_type_base: str, kid: str):
    # A) { "<kernel_type_base>": { "<ID>": { "config": {...} } } }
    if isinstance(js, dict) and kernel_type_base in js and isinstance(js[kernel_type_base], dict):
        node = js[kernel_type_base].get(kid)
        if isinstance(node, dict):
            return node.get("config", node)
    # B) { "<ID>": { "config": {...} } }
    if isinstance(js, dict) and kid in js and isinstance(js[kid], dict):
        node = js[kid]
        return node.get("config", node)
    return None

def get_params_from_results(super_type: str, kernel_type_base: str, kid: str,
                            cache: dict, results_root: Path):
    """
    只读取 config 里的 5 个固定键（缺失则置空）。
    返回: (HW, CIN, COUT, KERNEL_SIZE, STRIDES, ok, debug_json_path)
    """
    key = (super_type, kernel_type_base)
    if key not in cache:
        js, jpath = _load_kernel_json(super_type, kernel_type_base, results_root)
        cache[key] = {"js": js or {}, "path": jpath}

    js    = cache[key]["js"]
    jpath = cache[key]["path"]

    if not js:
        return "", "", "", "", "", False, jpath

    cfg = _find_config_in_json(js, kernel_type_base, kid)
    if not cfg or not isinstance(cfg, dict):
        return "", "", "", "", "", False, jpath

    HW          = str(cfg.get("HW", ""))
    CIN         = str(cfg.get("CIN", ""))
    COUT        = str(cfg.get("COUT", ""))
    KERNEL_SIZE = str(cfg.get("KERNEL_SIZE", ""))
    STRIDES     = str(cfg.get("STRIDES", ""))

    ok = any([HW, CIN, COUT, KERNEL_SIZE, STRIDES])
    return HW, CIN, COUT, KERNEL_SIZE, STRIDES, ok, jpath

# ---------------- Main Loop ---------------- #
if __name__ == "__main__":
    HVMON, HVengine = power_monitor_setup(26589, pmapi.USB_protocol())

    device_name = "Pixel8pro"
    model_root  = Path("kernel_dataset/kernels")        # 递归扫描
    results_root = Path("kernel_dataset/results")       # ### CHANGED: 提到循环外
    out_root    = Path("energy_results") / device_name
    out_root.mkdir(parents=True, exist_ok=True)

    backend_bins = {
        "GPU": "/data/local/tmp/benchmark_model_gpu",
        "CPU": "/data/local/tmp/benchmark_model",
    }

    summary_files = {}
    summary_writers = {}
    for be in backend_bins:
        spath = out_root / f"summary_{be.lower()}.csv"
        f = open(spath, "w", newline="")
        w = csv.writer(f)
        w.writerow(["Model", "KernelType", "HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "Energy_mJ"])
        summary_files[be] = f
        summary_writers[be] = w

    # ### CHANGED: cfg_cache 在循环外，避免重复读盘
    cfg_cache = {}  # (SuperType, kernel_type_base) -> json cache

    # 递归扫描
    for root, _, files in os.walk(model_root):
        rel = Path(root).resolve().relative_to(Path(model_root).resolve())
        parts = rel.parts  # 相对 model_root 的路径分段

        # 规则：第一段就是 SuperType（例如 FC/Conv/Bn/...）
        super_type = parts[0] if len(parts) >= 1 else os.path.basename(root)

        # kernel_type（用于输出目录展示）：如果有第二段则用第二段，否则沿用 super_type
        kernel_type = parts[1] if len(parts) >= 2 else super_type

        tflites = [f for f in files if f.endswith(".tflite")]
        if not tflites:
            continue

        for fname in tflites:
            model_path = Path(root) / fname
            base = model_path.stem

            # 1) 提取 ID 与 JSON 前缀
            kernel_id = extract_kernel_id(base)
            if not kernel_id:
                print(f"[WARN] 无法从文件名提取 ID：{fname}，跳过")
                continue
            kernel_type_base = kernel_type_base_from_name(base)

            # 2) 读取 5 键参数
            HW, CIN, COUT, KS, STRIDE, ok, jpath = get_params_from_results(
                super_type, kernel_type_base, kernel_id, cfg_cache, results_root
            )
            if not ok:
                print(f"[WARN] 未在 {jpath or f'results/{super_type}/{kernel_type_base}_test.json'} 找到 ID={kernel_id} 的配置（或5键全空），参数将留空。")

            # 3) 两个 backend 分别执行
            for be, bin_on_phone in backend_bins.items():
                be_dir = out_root / be / kernel_type
                be_dir.mkdir(parents=True, exist_ok=True)

                power_csv     = be_dir / f"{base}_power.csv"
                inference_csv = be_dir / f"{base}_inference_time.csv"

                ok_run = run_inference(
                    HVMON, HVengine,
                    model_path=str(model_path),
                    power_csv=str(power_csv),
                    benchmark_bin=bin_on_phone,
                    num_runs=10,
                    warmup_runs=10,
                    backend=be,
                    kernel_path="/data/local/tmp/kernel.cl"
                )
                if not ok_run:
                    continue

                if not (power_csv.exists() and inference_csv.exists()):
                    print(f"[WARN] Missing files for energy: {power_csv} or {inference_csv}. Skip.")
                    continue

                total_energy = compute_energy(str(power_csv), str(inference_csv))

                summary_writers[be].writerow(
                    [fname, kernel_type, HW, CIN, COUT, KS, STRIDE, total_energy]
                )
                summary_files[be].flush()

                print(f"[✓] {be} energy saved for {fname}")

    for f in summary_files.values():
        f.close()

    print("\n[Done] Profiling finished. Summaries are under:", out_root)