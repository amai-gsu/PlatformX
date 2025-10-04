# train_mnv2_cifar10_big_to_tflite_gpu_fast.py
# -*- coding: utf-8 -*-
import os, traceback, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import mobilenet_v2

# ================== 预设（改这里）==================
# "fast_head"     : 只训练分类头（最快），几分钟内出结果
# "quick_finetune": 冻结主干后逐步解冻最后 ~30 层做短暂微调（速度/精度折中）
# "full"          : 全量训练（最慢，效果最好）
PRESET = "fast_head"   # "fast_head" | "quick_finetune" | "full"
# ==================================================

# ========= 训练/硬件开关 =========
USE_MIXED_PRECISION = True   # AMP: FP16 训练（最后一层保持 FP32）
USE_XLA            = True    # XLA JIT
SEED               = 42

# ========= 基本模型参数（会被预设覆盖）=========
IMG_SIZE = (128, 128)
ALPHA    = 1.40
USE_WIDE_HEAD = True
HEAD_CHANNELS = int(1280 * ALPHA)
HEAD_DROPOUT  = 0.35

# ========= 训练配置（会被预设覆盖）=========
BATCH_SIZE = 128
EPOCHS     = 5
BASE_LR    = 1e-3
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY    = 1e-5
IMAGENET_PRETRAINED = None   # None 或 "imagenet"
TRAIN_SAMPLES = 50000        # 训练子集（<=50000）
VAL_SAMPLES   = 10000        # 验证子集（<=10000）
FREEZE_BASE   = False        # 是否冻结 backbone
UNFREEZE_TAIL_LAYERS = 0     # 解冻尾部可训练的层数（仅 quick/full 用）

# ========= 导出 =========
OUTPUT_DIR   = "mnv2_cifar10_big_export"
EXPORT_FP32  = True
EXPORT_INT8  = True
REP_SAMPLES  = 200
NUM_CLASSES  = 10
# ======================================

# ---------- 根据预设自动调整 ----------
if PRESET == "fast_head":
    # 极快：用 ImageNet 预训练，只训分类头；分辨率 128，alpha=1.0 更快
    IMG_SIZE = (128, 128)
    ALPHA = 1.0
    USE_WIDE_HEAD = True
    HEAD_CHANNELS = int(1280 * ALPHA)
    BATCH_SIZE = 192
    EPOCHS = 3
    BASE_LR = 2e-3
    IMAGENET_PRETRAINED = "imagenet"
    TRAIN_SAMPLES = 20000   # 2 万子集就够得到不错的结果
    FREEZE_BASE = True
    UNFREEZE_TAIL_LAYERS = 0
elif PRESET == "quick_finetune":
    # 折中：先冻住，再解冻最后 30 层微调；分辨率 128，alpha=1.0
    IMG_SIZE = (128, 128)
    ALPHA = 1.0
    USE_WIDE_HEAD = True
    HEAD_CHANNELS = int(1280 * ALPHA)
    BATCH_SIZE = 160
    EPOCHS = 8
    BASE_LR = 1e-3
    IMAGENET_PRETRAINED = "imagenet"
    TRAIN_SAMPLES = 40000
    FREEZE_BASE = True
    UNFREEZE_TAIL_LAYERS = 30
elif PRESET == "full":
    # 全量：你原始的放大设定，时间最长
    IMG_SIZE = (128, 128)
    ALPHA = 1.40
    USE_WIDE_HEAD = True
    HEAD_CHANNELS = int(1280 * ALPHA)
    BATCH_SIZE = 128
    EPOCHS = 20
    BASE_LR = 1e-3
    IMAGENET_PRETRAINED = None
    TRAIN_SAMPLES = 50000
    FREEZE_BASE = False
    UNFREEZE_TAIL_LAYERS = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

# ----------------- 运行环境 -----------------
def setup_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] {len(gpus)} GPU(s) available")
        except Exception as e:
            print("[GPU] memory growth failed:", e)

    if USE_MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[AMP] mixed_float16 ON")

    if USE_XLA:
        tf.config.optimizer.set_jit(True)
        print("[XLA] JIT ON")

    if gpus and len(gpus) >= 2:
        return tf.distribute.MirroredStrategy()
    elif gpus:
        return tf.distribute.OneDeviceStrategy("/GPU:0")
    else:
        return tf.distribute.OneDeviceStrategy("/CPU:0")

strategy = setup_runtime()

# 1) 数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 子集以提速
if TRAIN_SAMPLES < len(x_train):
    x_train = x_train[:TRAIN_SAMPLES]
    y_train = y_train[:TRAIN_SAMPLES]
if VAL_SAMPLES < len(x_test):
    x_test  = x_test[:VAL_SAMPLES]
    y_test  = y_test[:VAL_SAMPLES]

y_train = y_train.squeeze().astype(np.int32)
y_test  = y_test.squeeze().astype(np.int32)

# 2) 预处理/增强
def preprocess(x, y):
    x = tf.image.resize(x, IMG_SIZE)
    x = tf.cast(x, tf.float32)
    x = mobilenet_v2.preprocess_input(x)   # [-1,1]
    return x, tf.one_hot(y, NUM_CLASSES)

def augment(x, y):
    x = tf.image.random_flip_left_right(x)  # 简单增强，轻量
    # 如需更强增强可加随机裁剪/平移
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
opts = tf.data.Options()
opts.deterministic = False  # 允许非确定执行以提速

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .with_options(opts)
            .shuffle(min(50000, TRAIN_SAMPLES), seed=SEED, reshuffle_each_iteration=True)
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .map(augment,   num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

# 3) 模型
with strategy.scope():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base = mobilenet_v2.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=IMAGENET_PRETRAINED,  # "imagenet" for fast convergence in fast_head/quick_finetune
        alpha=ALPHA
    )
    if FREEZE_BASE:
        for l in base.layers:
            l.trainable = False

    x = base.output
    if USE_WIDE_HEAD:
        x = layers.Conv2D(
            HEAD_CHANNELS, 1, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(HEAD_DROPOUT)(x)
    # AMP 下输出层保持 float32 更稳
    out_dtype = "float32" if USE_MIXED_PRECISION else None
    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                           dtype=out_dtype)(x)
    model = models.Model(inputs, outputs)
    model.summary()

    steps_per_epoch = int(np.ceil(len(x_train) / BATCH_SIZE))
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=BASE_LR, decay_steps=max(1, steps_per_epoch * EPOCHS)
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
        jit_compile=USE_XLA
    )

# 4) 回调
ckpt_path = os.path.join(OUTPUT_DIR, f"mnv2_cifar10_{PRESET}_best.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                       save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                     restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                         patience=3, min_lr=1e-5, verbose=1),
]

# 5) 训练（quick_finetune 可选阶段二：解冻尾部层微调几轮）
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

if PRESET == "quick_finetune" and UNFREEZE_TAIL_LAYERS > 0:
    print(f"[fine-tune] unfreezing last {UNFREEZE_TAIL_LAYERS} layers ...")
    # 解冻尾部部分层（通常不解冻 BN 更稳）
    trainable = 0
    for l in reversed(base.layers):
        if trainable >= UNFREEZE_TAIL_LAYERS:
            break
        if not isinstance(l, layers.BatchNormalization):
            l.trainable = True
            trainable += 1
    # 细调学习率再训几轮
    ft_epochs = max(2, EPOCHS // 2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR * 0.25),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
        jit_compile=USE_XLA
    )
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs, callbacks=callbacks)

# 6) 评估
metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
print(f"[Keras] {PRESET}  Top-1: {metrics['accuracy']:.4f} | Top-5: {metrics['top5']:.4f}")

# 7) 保存 Keras
keras_path = os.path.join(OUTPUT_DIR, f"mnv2_cifar10_{PRESET}_keras.h5")
model.save(keras_path)
print("Saved Keras model to:", keras_path)

# ================== 稳健导出器（新增）==================
def _log_converter_env():
    try:
        tf.get_logger().setLevel("INFO")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        print("[TFLite] TF:", tf.__version__)
    except Exception:
        pass

def _build_concrete_function(keras_model, input_shape_hw):
    """用 ConcreteFunction 固定 training=False 与输入 dtype/shape。"""
    H, W = input_shape_hw
    @tf.function(input_signature=[tf.TensorSpec([1, H, W, 3], tf.float32)])
    def serving(x):
        x = tf.cast(x, tf.float32)
        y = keras_model(x, training=False)
        return {"logits": tf.cast(y, tf.float32)}
    return serving.get_concrete_function()

def export_tflite_fp32_robust(keras_model, out_path, input_shape_hw=None, allow_select_tf_ops=True):
    """
    尝试顺序：
    A) 直接 from_keras_model
    B) ConcreteFunction（固定 training=False）
    C) SELECT_TF_OPS
    D) 兜底导出 FP16（返回 *_fp16.tflite）
    """
    _log_converter_env()
    # 确保推理态
    keras_model.trainable = False

    # --- A) 直接 Keras 模型 ---
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        conv._experimental_lower_to_saved_model = True
        conv.experimental_enable_resource_variables = True
        tfl = conv.convert()
        with open(out_path, "wb") as f: f.write(tfl)
        print("[TFLite] FP32 export OK (from_keras_model):", out_path)
        return out_path
    except Exception as e:
        print("[TFLite][A] FP32 convert failed:", e)
        traceback.print_exc(limit=2)

    # --- B) ConcreteFunction（冻结 training=False）---
    if input_shape_hw is not None:
        try:
            conc = _build_concrete_function(keras_model, input_shape_hw)
            conv = tf.lite.TFLiteConverter.from_concrete_functions([conc], keras_model)
            conv._experimental_lower_to_saved_model = True
            conv.experimental_enable_resource_variables = True
            tfl = conv.convert()
            with open(out_path, "wb") as f: f.write(tfl)
            print("[TFLite] FP32 export OK (ConcreteFunction):", out_path)
            return out_path
        except Exception as e:
            print("[TFLite][B] FP32 convert failed (ConcreteFunction):", e)
            traceback.print_exc(limit=2)

    # --- C) 放宽算子集（SELECT_TF_OPS）---
    if allow_select_tf_ops:
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            conv._experimental_lower_to_saved_model = True
            conv.experimental_enable_resource_variables = True
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tfl = conv.convert()
            with open(out_path, "wb") as f: f.write(tfl)
            print("[TFLite] FP32 export OK (SELECT_TF_OPS):", out_path)
            return out_path
        except Exception as e:
            print("[TFLite][C] FP32 convert failed (SELECT_TF_OPS):", e)
            traceback.print_exc(limit=2)

    # --- D) 兜底：导出 FP16 ---
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        conv._experimental_lower_to_saved_model = True
        tfl16 = conv.convert()
        fp16_path = os.path.splitext(out_path)[0] + "_fp16.tflite"
        with open(fp16_path, "wb") as f: f.write(tfl16)
        print("[TFLite] FP16 export OK:", fp16_path)
        return fp16_path
    except Exception as e:
        print("[TFLite][D] FP16 convert failed too:", e)
        traceback.print_exc(limit=2)
        raise RuntimeError("TFLite export failed at all attempts.") from e

def export_tflite_int8_robust(keras_model, out_path, rep_data_gen=None, input_shape_hw=None):
    """
    整模型 int8 量化。失败时也尝试 ConcreteFunction / SELECT_TF_OPS。
    """
    _log_converter_env()
    keras_model.trainable = False

    def _try(conv):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if rep_data_gen is not None:
            conv.representative_dataset = rep_data_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        return conv.convert()

    # 直接 from_keras_model
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        conv._experimental_lower_to_saved_model = True
        tfl = _try(conv)
        with open(out_path, "wb") as f: f.write(tfl)
        print("[TFLite] INT8 export OK:", out_path)
        return out_path
    except Exception as e:
        print("[TFLite][INT8-A] failed:", e)
        traceback.print_exc(limit=2)

    # ConcreteFunction
    if input_shape_hw is not None:
        try:
            conc = _build_concrete_function(keras_model, input_shape_hw)
            conv = tf.lite.TFLiteConverter.from_concrete_functions([conc], keras_model)
            conv._experimental_lower_to_saved_model = True
            tfl = _try(conv)
            with open(out_path, "wb") as f: f.write(tfl)
            print("[TFLite] INT8 export OK (ConcreteFunction):", out_path)
            return out_path
        except Exception as e:
            print("[TFLite][INT8-B] failed (ConcreteFunction):", e)
            traceback.print_exc(limit=2)

    # SELECT_TF_OPS（并不总和 INT8 相容，但作为最后尝试）
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        conv._experimental_lower_to_saved_model = True
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        if rep_data_gen is not None:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = rep_data_gen
        tfl = conv.convert()
        with open(out_path, "wb") as f: f.write(tfl)
        print("[TFLite] INT8 export OK (SELECT_TF_OPS):", out_path)
        return out_path
    except Exception as e:
        print("[TFLite][INT8-C] failed (SELECT_TF_OPS):", e)
        traceback.print_exc(limit=2)
        raise RuntimeError("INT8 export failed at all attempts.") from e

# 代表性数据（与训练前处理一致：resize + mobilenet_v2.preprocess_input → [-1,1]）
def rep_data_gen():
    n = min(REP_SAMPLES, len(x_train))
    for i in range(n):
        img = x_train[i]
        img = tf.image.resize(img, IMG_SIZE)
        img = mobilenet_v2.preprocess_input(tf.cast(img, tf.float32))
        img = tf.expand_dims(img, 0)  # [1,H,W,3]
        yield [img]

# 8) 导出 TFLite（稳健）
tflite_fp32_path = os.path.join(OUTPUT_DIR, f"mnv2_cifar10_{PRESET}_fp32.tflite")
if EXPORT_FP32:
    export_tflite_fp32_robust(model, tflite_fp32_path, input_shape_hw=IMG_SIZE)

if EXPORT_INT8:
    tflite_int8_path = os.path.join(OUTPUT_DIR, f"mnv2_cifar10_{PRESET}_int8.tflite")
    try:
        export_tflite_int8_robust(model, tflite_int8_path, rep_data_gen=rep_data_gen, input_shape_hw=IMG_SIZE)
    except Exception as e:
        print("[WARN] INT8 导出失败：", e)
        print("      可能的原因：部分算子不支持整型量化或代表性数据与预处理不匹配。")
