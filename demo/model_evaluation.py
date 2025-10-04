# -*- coding: utf-8 -*-
"""
CIFAR-10 accuracy comparison for two TFLite models (e.g., 32x32 vs 224x224).
- Auto-detect input layout (NHWC or NCHW) per model.
- Auto-resize per model input size.
- Supports float and quantized (uint8/int8) inputs/outputs.
- Reports Top-1 / Top-K accuracy and sec/img.
Dependencies: tensorflow, pillow, numpy
"""

# ======== User config ========
MODEL_A_PATH = "/home/xiaolong/Downloads/GreenAuto/exports_trained_nhwc/EnergyNet_uid8723_cfg181_nhwc.tflite"   # 你的模型（NCHW: 1x3x32x32）
MODEL_B_PATH = "/home/xiaolong/Downloads/GreenAuto/mnv2_cifar10_export/mnv2_cifar10_fp32.tflite"  
PROFILE_A = "0_1"        # 你的模型常用: '0_1'；可选: auto|mobilenet|0_1|imagenet|raw
PROFILE_B = "mobilenet"  # MobileNet 常用: 'mobilenet'
TOPK = 5
LIMIT = 0                # 0=用完整 10k 测试集；否则仅用前 LIMIT 张
NUM_THREADS = 4
# ============================

import os, sys, time
from typing import Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def build_interpreter(model_path: str, num_threads: int) -> tf.lite.Interpreter:
    itp = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    itp.allocate_tensors()
    return itp

def get_io_details(interpreter: tf.lite.Interpreter):
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    in_shape = list(in_d["shape"])  # [N, d1, d2, d3]
    in_dtype = in_d["dtype"]
    in_q = in_d.get("quantization", (0.0, 0))
    out_dtype = out_d["dtype"]
    out_q = out_d.get("quantization", (0.0, 0))

    if len(in_shape) != 4:
        raise RuntimeError(f"Expected 4D input, got {in_shape}")

    N, d1, d2, d3 = in_shape
    # 判断布局：优先把最后一维=通道视为 NHWC，否则第二维=通道视为 NCHW
    if d3 in (1, 3):
        layout = "NHWC"
        h, w, c = d1, d2, d3
    elif d1 in (1, 3):
        layout = "NCHW"
        c, h, w = d1, d2, d3
    else:
        # 兜底按 NHWC
        layout = "NHWC"
        h, w, c = d1, d2, d3

    num_classes = int(out_d["shape"][-1])
    return {
        "h": h, "w": w, "c": c, "layout": layout,
        "in_dtype": in_dtype, "in_q": in_q,
        "out_dtype": out_dtype, "out_q": out_q,
        "in_d": in_d, "out_d": out_d, "num_classes": num_classes,
    }

def _normalize(real: np.ndarray, c: int, in_dtype, in_q: Tuple[float, int], profile: str):
    """
    real: float32 NHWC in pixel space (0..255).
    return: normalized & (if needed) quantized NHWC array.
    """
    if profile == "mobilenet":
        real = (real / 127.5) - 1.0
    elif profile == "0_1":
        real = real / 255.0
    elif profile == "imagenet":
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        if c == 1:
            mean = np.array([123.675], dtype=np.float32)
            std  = np.array([58.395], dtype=np.float32)
        real = (real - mean) / std
    elif profile == "raw":
        # 不做归一化（保持 0..255）
        pass
    elif profile == "auto":
        if in_dtype == np.float32:
            real = (real / 127.5) - 1.0
        else:
            real = real / 255.0
    else:
        raise ValueError(f"Unknown profile: {profile}")

    # 量化（若模型输入是量化类型）
    if in_dtype in (np.uint8, np.int8):
        scale, zero = in_q if in_q is not None else (0.0, 0)
        if scale == 0.0:
            if in_dtype == np.uint8:
                q = np.clip(real, 0.0, 255.0).astype(np.uint8)
            else:
                q = np.clip(real, -128.0, 127.0).astype(np.int8)
        else:
            q = np.round(real / float(scale) + float(zero))
            if in_dtype == np.uint8:
                q = np.clip(q, 0, 255).astype(np.uint8)
            else:
                q = np.clip(q, -128, 127).astype(np.int8)
        return q.astype(in_dtype)
    else:
        return real.astype(np.float32)

def preprocess_batch_cifar10(x: np.ndarray, io, profile: str) -> np.ndarray:
    """
    x: CIFAR-10 测试集，可能是 NHWC [N,32,32,3] 或 NCHW [N,3,32,32]
    io: get_io_details 返回的字典（含 h,w,c,layout,in_dtype,in_q）
    返回：匹配模型输入布局/形状/类型的数组：NHWC 或 NCHW
    """
    h, w, c   = io["h"], io["w"], io["c"]
    layout    = io["layout"]
    in_dtype  = io["in_dtype"]
    in_q      = io["in_q"]

    if x.ndim != 4:
        raise ValueError(f"Expected 4D array for CIFAR-10, got {x.shape}")

    # 统一到 NHWC
    if (x.shape[-1] not in (1, 3)) and (x.shape[1] in (1, 3)):
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

    N = x.shape[0]
    nhwc = np.empty((N, h, w, c), dtype=np.float32)

    for i in range(N):
        arr = x[i]  # 期望 NHWC
        # 转为 PIL 期望的 uint8 HWC
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).round()
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        img = img.convert('RGB' if c == 3 else 'L')
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if c == 1 and arr.ndim == 2:
            arr = arr[:, :, None]  # HW -> HW1
        nhwc[i] = arr

    # 归一化（在 NHWC 空间里做）
    nhwc = _normalize(nhwc, c, in_dtype, in_q, profile)

    # 若模型输入布局为 NCHW，则转换
    if layout == "NCHW":
        nchw = np.transpose(nhwc, (0, 3, 1, 2))  # NHWC -> NCHW
        return nchw
    else:
        return nhwc

def dequantize(y: np.ndarray, out_dtype, out_q: Tuple[float, int]) -> np.ndarray:
    if out_dtype in (np.uint8, np.int8):
        scale, zero = out_q if out_q is not None else (0.0, 0)
        if scale == 0.0:
            return y.astype(np.float32)
        return scale * (y.astype(np.float32) - float(zero))
    return y.astype(np.float32)

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[-1]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]

def evaluate_on_cifar10(interpreter: tf.lite.Interpreter,
                        profile: str,
                        images: np.ndarray,
                        labels: np.ndarray,
                        topk: int):
    io = get_io_details(interpreter)
    h, w, c = io["h"], io["w"], io["c"]
    num_classes = io["num_classes"]
    out_dtype, out_q = io["out_dtype"], io["out_q"]
    in_d, out_d = io["in_d"], io["out_d"]

    if num_classes != 10:
        print(f"[WARN] model outputs {num_classes} classes, but CIFAR-10 has 10. "
              f"Unless it’s a CIFAR-10 model, accuracy is not meaningful.", file=sys.stderr)

    # 预处理到与模型输入布局一致的张量
    x_all = preprocess_batch_cifar10(images, io, profile)

    N = x_all.shape[0]
    t1 = 0
    tk = 0
    t0 = time.time()
    for i in range(N):
        x = x_all[i:i+1]  # [1,H,W,C] 或 [1,C,H,W]
        interpreter.set_tensor(in_d['index'], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_d['index'])[0]
        y = dequantize(y, out_dtype, out_q)

        pred1 = int(np.argmax(y))
        tk_idx = topk_indices(y, topk)
        if pred1 == int(labels[i]):
            t1 += 1
        if int(labels[i]) in tk_idx:
            tk += 1

    elapsed = time.time() - t0
    return {
        "images": N,
        "input_hw": (h, w),
        "channels": c,
        "num_classes": num_classes,
        "top1_acc": t1 / N,
        f"top{topk}_acc": tk / N,
        "sec_per_img": elapsed / max(1, N),
        "profile": profile,
        "layout": io["layout"],
    }

def sanity_compare_two_tflites(nchw_tflite, nhwc_tflite, profile="0_1", num=32):
    import numpy as np, tensorflow as tf
    from tensorflow.keras.datasets import cifar10
    from PIL import Image

    def load_io(p):
        itp = tf.lite.Interpreter(model_path=p); itp.allocate_tensors()
        i = itp.get_input_details()[0]; o = itp.get_output_details()[0]
        return itp, i, o

    def prep_batch_hwc(x, H, W, C, profile):
        N = x.shape[0]
        buf = np.empty((N, H, W, C), dtype=np.float32)
        for i in range(N):
            img = Image.fromarray(x[i]).convert('RGB' if C==3 else 'L').resize((W,H), Image.BILINEAR)
            a = np.array(img, dtype=np.float32)
            if C==1 and a.ndim==2: a = a[:, :, None]
            buf[i] = a
        if profile == "mobilenet":
            real = (buf / 127.5) - 1.0
        elif profile == "0_1":
            real = buf / 255.0
        else:
            real = buf / 255.0
        return real.astype(np.float32)

    # 载入 CIFAR-10 测试集的小批
    (_, _), (xt, yt) = cifar10.load_data()
    xt, yt = xt[:num], yt[:num].flatten()

    itp_old, i_old, o_old = load_io(nchw_tflite)
    itp_new, i_new, o_new = load_io(nhwc_tflite)

    # 解析两模型的尺寸 & 布局
    def io_shape_to_layout(i):
        N,d1,d2,d3 = list(i["shape"])
        if d3 in (1,3): return ("NHWC", d1,d2,d3)
        elif d1 in (1,3): return ("NCHW", d2,d3,d1)
        else: return ("NHWC", d1,d2,d3)
    lay_old, H_old, W_old, C_old = io_shape_to_layout(i_old)
    lay_new, H_new, W_new, C_new = io_shape_to_layout(i_new)

    # 准备同一批输入：先做成 HWC，再分别喂给两模型
    X_hwc_old = prep_batch_hwc(xt, H_old, W_old, C_old, profile)
    X_hwc_new = prep_batch_hwc(xt, H_new, W_new, C_new, profile)

    # 旧模型若是 NCHW，转一下
    if lay_old == "NCHW":
        X_old = np.transpose(X_hwc_old, (0,3,1,2)).astype(i_old["dtype"])
    else:
        X_old = X_hwc_old.astype(i_old["dtype"])
    X_new = X_hwc_new.astype(i_new["dtype"])

    def run(itp, i, o, X):
        preds = []
        for k in range(X.shape[0]):
            itp.set_tensor(i['index'], X[k:k+1])
            itp.invoke()
            y = itp.get_tensor(o['index'])[0].astype(np.float32)
            preds.append(y)
        return np.stack(preds, 0)

    Y_old = run(itp_old, i_old, o_old, X_old)
    Y_new = run(itp_new, i_new, o_new, X_new)

    top1_old = (Y_old.argmax(-1) == yt[:len(Y_old)]).mean()
    top1_new = (Y_new.argmax(-1) == yt[:len(Y_new)]).mean()
    agree = (Y_old.argmax(-1) == Y_new.argmax(-1)).mean()
    cos_sim = np.mean([
        np.dot(Y_old[k], Y_new[k]) / (np.linalg.norm(Y_old[k])+1e-6) / (np.linalg.norm(Y_new[k])+1e-6)
        for k in range(len(Y_old))
    ])
    print(f"[sanity] old_top1={top1_old:.3f}, new_top1={top1_new:.3f}, argmax_agree={agree:.3f}, avg_cos={cos_sim:.3f}")


def main():

#     sanity_compare_two_tflites(
#     nchw_tflite="/home/xiaolong/Downloads/nasbench-NWOT-score/generated_model_deploy_top5_acc_3/NAS201Net_deploy_13714_2.tflite",
#     nhwc_tflite="/home/xiaolong/Downloads/GreenAuto/NAS201Net_deploy_13714_2_nhwc_out/onnx2tf_out/model_simplified_float32.tflite",
#     profile="0_1",  # 你原来评估 A 用的是 0_1
#     num=64
# )

    # Load CIFAR-10 test set
    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = y_test.flatten().astype(np.int32)
    if LIMIT and LIMIT > 0:
        x_test = x_test[:LIMIT]
        y_test = y_test[:LIMIT]

    # Build interpreters
    inter_a = build_interpreter(MODEL_A_PATH, NUM_THREADS)
    inter_b = build_interpreter(MODEL_B_PATH, NUM_THREADS)

    print("\n=== Evaluating Model A ===")
    print(f"Model A: {MODEL_A_PATH}")
    res_a = evaluate_on_cifar10(inter_a, PROFILE_A, x_test, y_test, TOPK)
    print(f" Input: {res_a['input_hw'][0]}x{res_a['input_hw'][1]}x{res_a['channels']} | layout={res_a['layout']} | classes: {res_a['num_classes']}")
    print(f" Preprocess: {res_a['profile']} | images: {res_a['images']}")
    print(f" Top-1 Acc: {res_a['top1_acc']:.3%}")
    print(f" Top-{TOPK} Acc: {res_a[f'top{TOPK}_acc']:.3%}")
    print(f" Sec/img: {res_a['sec_per_img']:.4f}\n")

    print("=== Evaluating Model B ===")
    print(f"Model B: {MODEL_B_PATH}")
    res_b = evaluate_on_cifar10(inter_b, PROFILE_B, x_test, y_test, TOPK)
    print(f" Input: {res_b['input_hw'][0]}x{res_b['input_hw'][1]}x{res_b['channels']} | layout={res_b['layout']} | classes: {res_b['num_classes']}")
    print(f" Preprocess: {res_b['profile']} | images: {res_b['images']}")
    print(f" Top-1 Acc: {res_b['top1_acc']:.3%}")
    print(f" Top-{TOPK} Acc: {res_b[f'top{TOPK}_acc']:.3%}")
    print(f" Sec/img: {res_b['sec_per_img']:.4f}\n")

    print("=== Summary ===")
    print(f"A: Top-1={res_a['top1_acc']:.3%} | Top-{TOPK}={res_a[f'top{TOPK}_acc']:.3%} | {res_a['input_hw'][0]}x{res_a['input_hw'][1]} ({res_a['profile']}, {res_a['layout']})")
    print(f"B: Top-1={res_b['top1_acc']:.3%} | Top-{TOPK}={res_b[f'top{TOPK}_acc']:.3%} | {res_b['input_hw'][0]}x{res_b['input_hw'][1]} ({res_b['profile']}, {res_b['layout']})")

if __name__ == "__main__":
    main()
