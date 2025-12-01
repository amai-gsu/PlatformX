# PlatformX: End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search

PlatformX is an end-to-end, fully automated platform for **hardware-aware Neural Architecture Search (HW-NAS)** targeting **energy-efficient deep neural networks** on mobile and edge devices.

This repository accompanies the paper:

**PlatformX: An End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search**  
(To appear at ACM/IEEE SEC 2025)

---

## Key Features

- **Model generation & search space design**
- **Transferable energy prediction** (LitePred-style, RF/MLP-based, etc.)
- **Multi-objective NAS algorithms**  
  - GD-Pareto  
  - Bayesian Optimization  
  - Evolutionary (Greedy+Random)  
  - UCB  
  - Random  
- **On-device inference & Monsoon Power Monitor measurement**
- **Latency + enercy live profiling**

---

## Repository Structure

```
PlatformX/
│
├── benchmark_tools/                 # Android/TFLite benchmark tools & wrappers
│
├── cifar10/
│   └── cifar-10-batches-py/         # CIFAR-10 dataset
│
├── demo/                            # Quickstart demos
│
├── latency_acc_live/                # Live latency and accuracy profiling
│   ├── scripts/
│   └── utils/
│
├── model_generation/                # Architecture and search space generation
│
├── model_search/                    # NAS algorithms (GD/BO/EVO/UCB/Random)
│   ├── Model_search_compare_mn.py   # Main NAS script
│   ├── search_utils.py
│   ├── pareto_utils.py
│   └── configs/
│
├── monsoon/                         # Monsoon HVPM measurement pipeline
│   ├── server_orchestrator_phone.py
│   ├── monsoon_measurement.py
│   └── HV Main Example.csv
│
├── pred_ckpts/                      # Pretrained predictor checkpoints
│   └── pixel7/
│
├── prediction/                      # Energy predictor & accuracy proxy training
│
├── prediction_results/              # Predictor outputs
│
├── results/                         # Final NAS results, plots, logs
│
├── energy_genotype.txt
└── README.md
```

---

## Requirements

- PyTorch  
- TensorFlow + TFLite  
- Monsoon Power API  
- adb (Android Platform Tools)

---

## Running NAS Search

Example for GD-Pareto:

```
python model_search/Model_search_compare_mn.py \
    --energy_json prediction_results/energy.json \
    --config_json prediction_results/config.json \
    --backend cpu \
    --data_root cifar10/cifar-10-batches-py \
    --base_out results/ \
    --pool_size 100000 \
    --max_evals none \
    --acc_proxy zero_cost \
    --probe_epochs 1 \
    --probe_batches 80 \
    --method gd_pareto \
    --use_energy_predictor true \
    --acc_threshold 0.80
```

### Supported Methods

```
--method gd_pareto
--method bo
--method evo
--method ucb
--method random
```

### Enable/Disable Energy Predictor

```
--use_energy_predictor true
--use_energy_predictor false
```

---

## Real-Device Power Measurement (Android + Monsoon)

```
python monsoon/energy_prifiling.py
```

This script:
- Pushes TFLite models to phone  
- Runs TFLite benchmark  
- Collects high-frequency power traces with Monsoon HVPM  
- Aligns timestamps  
- Outputs latency, power, energy CSVs  

---
## 🎬 Demo Video

Watch a demo of PlatformX in action:

[PlatformX Demo Video](https://github.com/amai-gsu/PlatformX/releases/download/v1.0-demo/your_video_filename.mp4)


## Citation

```
@inproceedings{tu2025platformx,
  title     = {PlatformX: An End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search},
  author    = {Xiaolong Tu and others},
  booktitle = {Proceedings of the ACM/IEEE Symposium on Edge Computing (SEC)},
  year      = {2025}
}
```

---
