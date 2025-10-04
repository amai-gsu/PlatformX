# -*- coding: utf-8 -*-
"""
Train two EnergyNet models on CIFAR-10 with the SAME recipe as your MobileNetV2 Keras script,
then export each to NHWC TFLite via ONNX -> onnxsim -> onnx2tf.

Targets:
 - (uid=8723, config_idx=181)
 - (uid=15582, config_idx=125)
"""

import os, sys, math, time, subprocess, shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets

import onnx
import numpy as np

from nas_201_api import NASBench201API as API
from model_generation.model_generater import EnergyNet
from model_generation.genotypes import Structure

# ========= 可调参数（对齐 Keras 脚本） =========
API_PATH      = "NAS-Bench-201-v1_0-e61699.pth"   # 改成你的实际路径
OUTPUT_DIR    = Path("exports_trained_nhwc"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE      = (32, 32)
BATCH_SIZE    = 128
EPOCHS        = 50
BASE_LR       = 1e-3
LABEL_SMOOTH  = 0.1
PATIENCE      = 8            # 早停容忍
WEIGHT_DECAY  = 0.0          # 与你Keras例保持简洁；需要时可设 1e-4
NUM_WORKERS   = 4
SEED          = 1

# 只导出如下 (uid, 1-based config_idx)
DESIRED = {
    8723: {181},
    15582: {125},
}

# EnergyNet 搜索空间顺序（与原脚本一致）
COUT_RANGE        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256]
KERNEL_SIZE_RANGE = [1, 3, 5, 7]
STRIDE_RANGE      = [1, 2, 3, 4]

# ========== 工具函数 ==========
def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2structure(xstr: str) -> Structure:
    nodestrs = xstr.split('+'); genotypes = []
    for node_str in nodestrs:
        inputs = [x for x in node_str.split('|') if x != '']
        inputs = (xi.split('~') for xi in inputs)
        input_infos = tuple((op, int(idx)) for (op, idx) in inputs)
        genotypes.append(input_infos)
    return Structure(genotypes)

def generate_energy_configs(C):
    cfgs = []
    for cout in COUT_RANGE:
        for k in KERNEL_SIZE_RANGE:
            for s in STRIDE_RANGE:
                cfgs.append({'CIN': C, 'COUT': cout, 'KERNEL_SIZE': k, 'STRIDE': s, 'PADDING': 1})
    return cfgs

def _to_minus1_1(t):
    # torchvision.ToTensor() 产出 [0,1]，这里转成 [-1,1]
    return t.mul(2.0).sub(1.0)

def make_dataloaders(data_root: str):
    tf_train = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE[0], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(_to_minus1_1),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_to_minus1_1),
    ])
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH).to(device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def train_one(model, train_loader, val_loader, device, save_path: Path):
    model.to(device)
    model.train()

    # 训练配置：Adam + Cosine（按迭代）
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    total_iters = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH).to(device)

    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    iters_done = 0

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            iters_done += 1
            running += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1:03d}] train_loss={running/len(train_loader):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc; epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f"  * best so far, saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("  * early stop")
                break

    # 加载最佳
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc

def run(cmd, check=True):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}\n----- OUTPUT -----\n{p.stdout}")
    return p

def export_to_tflite_nhwc(model, tflite_out: Path):
    """
    PyTorch -> ONNX -> onnxsim -> onnx2tf -> TFLite (NHWC, fp32)
    """
    model_cpu = model.cpu().eval()
    onnx_path = tflite_out.with_suffix(".onnx")
    dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1], dtype=torch.float32)
    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True, verbose=False,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}
    )
    print(f"[ONNX] saved: {onnx_path}")

    # onnxsim
    onnx_simpl = onnx_path.with_name(onnx_path.stem + "_simplified.onnx")
    run([sys.executable, "-m", "onnxsim", str(onnx_path), str(onnx_simpl)])
    use_onnx = onnx_simpl if onnx_simpl.exists() else onnx_path

    # onnx2tf（默认生成 NHWC）
    out_dir = tflite_out.with_suffix("").with_name(tflite_out.stem + "_onnx2tf_out")
    if out_dir.exists(): shutil.rmtree(out_dir)
    cmd = ["onnx2tf", "-i", str(use_onnx), "-o", str(out_dir),
        "--non_verbose", "-kat", "input"]
    run(cmd)

    # 取最大的 .tflite 作为 fp32 主体
    tflites = list(out_dir.glob("*.tflite"))
    if not tflites:
        raise RuntimeError("onnx2tf 未生成 .tflite，请检查日志")
    src = max(tflites, key=lambda p: p.stat().st_size)
    shutil.copyfile(src, tflite_out)
    print(f"[TFLite NHWC] saved: {tflite_out}")

    # 简检
    import tensorflow as tf
    inter = tf.lite.Interpreter(model_path=str(tflite_out)); inter.allocate_tensors()
    i = inter.get_input_details()[0]; o = inter.get_output_details()[0]
    print(f"[Check] input={i['shape']} dtype={i['dtype']} | output={o['shape']} dtype={o['dtype']}")

# ========== 主流程 ==========
def main(data_loc="../cifardata"):
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not Path(API_PATH).exists():
        raise FileNotFoundError(f"NAS-Bench-201 not found: {API_PATH}")
    api = API(API_PATH, verbose=False)

    train_loader, val_loader = make_dataloaders(data_loc)

    for uid, wanted_idxs in DESIRED.items():
        cfg = api.get_net_config(uid, 'cifar10-valid')
        C, N, num_classes = cfg['C'], cfg['N'], cfg['num_classes']
        genotype = str2structure(cfg['arch_str'])
        all_cfgs = generate_energy_configs(C)
        total = len(all_cfgs)
        print(f"\n[UID {uid}] total energy configs: {total}")

        for idx_1based, energy_cfg in enumerate(all_cfgs, start=1):
            if idx_1based not in wanted_idxs:
                continue

            print(f"[Build] EnergyNet uid={uid} cfg_idx={idx_1based} -> {energy_cfg}")
            model = EnergyNet(
                N, genotype, num_classes,
                batch_size=BATCH_SIZE, config=energy_cfg
            )

            # 训练 & 保存最佳权重
            ckpt_path = OUTPUT_DIR / f"EnergyNet_uid{uid}_cfg{idx_1based}_best.pth"
            best_acc = train_one(model, train_loader, val_loader, device, ckpt_path)
            print(f"[Best] uid={uid} cfg={idx_1based}  val_acc={best_acc:.4f}")

            # 载入最佳再导出
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            tflite_out = OUTPUT_DIR / f"EnergyNet_uid{uid}_cfg{idx_1based}_nhwc.tflite"
            export_to_tflite_nhwc(model, tflite_out)

            # 小清理
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n[Done] All requested models trained and exported to:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    # 可改数据目录
    main(data_loc="../cifardata")
