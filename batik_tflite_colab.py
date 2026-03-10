"""
Batik Wastra Nusantara – TFLite Pipeline (Optimized for Google Colab)
=====================================================================
Dataset : dwibudisantoso/batik-wastra-nusantara (Kaggle)
Pipeline : audit → filter → split → augment → train → evaluate → TFLite → benchmark
Run      : Execute all cells in Google Colab (GPU recommended)
"""

# ── Auto-install dependencies (Colab-safe) ────────────────────────────────────
import subprocess
import sys

def _pip_install(*packages: str) -> None:
    for pkg in packages:
        try:
            __import__(pkg.split("[")[0].replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_pip_install("kagglehub", "seaborn")

# ── Standard imports ──────────────────────────────────────────────────────────
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
KAGGLE_DATASET = "dwibudisantoso/batik-wastra-nusantara"
COLAB_ROOT = Path("/content")
OUTPUT_DIR = COLAB_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

# Filtering
MIN_IMAGES = 30
MAX_IMAGES = 50

# Training
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS_HEAD = 12         # head-only training
EPOCHS_FINE = 10         # fine-tuning
FINE_TUNE_LAST_N = 30    # unfreeze last N backbone layers

# Visualization
RANDOM_SAMPLES_PER_CLASS = 1
MAX_MISCLASSIFIED_TO_SHOW = 16
BENCHMARK_SAMPLES = 120

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_cmd(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def print_gpu_info() -> None:
    """Print GPU availability and memory info."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  ✅ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"     - {gpu.name}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("  ⚠️  No GPU detected — training will be slow!")
    print(f"  TensorFlow version: {tf.__version__}")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Download
# ══════════════════════════════════════════════════════════════════════════════
def download_kaggle_dataset(slug: str) -> Path:
    """
    Download from Kaggle.
    Priority 1: kagglehub (stable on Colab)
    Priority 2: kaggle CLI (needs kaggle.json)
    """
    try:
        import kagglehub
        path = Path(kagglehub.dataset_download(slug))
        print(f"  Downloaded via kagglehub → {path}")
        return path
    except Exception as e:
        print(f"  kagglehub failed ({e}), falling back to kaggle CLI...")
        zip_path = COLAB_ROOT / "batik_wastra.zip"
        out_dir = COLAB_ROOT / "batik_wastra"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(f"kaggle datasets download -d {slug} -p {COLAB_ROOT}")
        run_cmd(f"unzip -o {zip_path} -d {out_dir}")
        return out_dir


def find_class_root(root: Path) -> Path:
    """
    Auto-detect the folder that contains class sub-folders.
    Picks the directory with the most sub-dirs containing images.
    """
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    best_dir, best_score = root, -1

    for d in candidates:
        subdirs = [x for x in d.iterdir() if x.is_dir()]
        if not subdirs:
            continue
        score = sum(
            1 for s in subdirs
            if any(f.suffix.lower() in ALLOWED_EXT for f in s.iterdir() if f.is_file())
        )
        if score > best_score:
            best_score = score
            best_dir = d

    return best_dir


# ══════════════════════════════════════════════════════════════════════════════
# Data Audit & Selection
# ══════════════════════════════════════════════════════════════════════════════
def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXT]


def is_valid_image(path: str) -> bool:
    """Check image validity with both PIL and TensorFlow decoding."""
    # 1. PIL check
    try:
        with Image.open(path) as img:
            fmt = img.format
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return False
    # 2. Reject formats TensorFlow cannot decode
    if fmt and fmt.upper() not in {"JPEG", "PNG", "GIF", "BMP"}:
        return False
    # 3. TensorFlow decode check (catches mismatched magic bytes)
    try:
        raw = tf.io.read_file(path)
        tf.image.decode_image(raw, channels=3, expand_animations=False)
        return True
    except (tf.errors.InvalidArgumentError, Exception):
        return False


def build_metadata(class_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows, image_rows = [], []

    for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
        imgs = list_images(class_dir)
        class_rows.append({"class_name": class_dir.name, "count": len(imgs)})
        for img in imgs:
            image_rows.append({"filepath": str(img), "class_name": class_dir.name})

    class_df = pd.DataFrame(class_rows).sort_values("count", ascending=False).reset_index(drop=True)
    image_df = pd.DataFrame(image_rows)
    return class_df, image_df


def filter_invalid_images(image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_mask = image_df["filepath"].map(is_valid_image)
    return image_df[valid_mask].reset_index(drop=True), image_df[~valid_mask].reset_index(drop=True)


def select_classes_by_range(class_df: pd.DataFrame, min_n: int, max_n: int) -> list[str]:
    mask = (class_df["count"] >= min_n) & (class_df["count"] <= max_n)
    return class_df.loc[mask, "class_name"].tolist()


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, title: str, out: Path) -> None:
    plt.figure(figsize=(12, max(4, len(df) * 0.3)))
    sns.barplot(data=df, x="count", y="class_name", color="#2563EB")
    plt.title(title)
    plt.xlabel("Image Count")
    plt.ylabel("Class")
    save_plot(out)


def plot_sample_grid(
    image_df: pd.DataFrame, class_names: list[str], out: Path, per_class: int = 1
) -> None:
    selected = []
    for c in class_names:
        rows = image_df[image_df["class_name"] == c]
        if rows.empty:
            continue
        sampled = rows.sample(n=min(per_class, len(rows)), random_state=SEED)
        selected.extend(sampled.to_dict("records"))

    n = len(selected)
    if n == 0:
        return

    ncols = min(7, n)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(3 * ncols, 3 * nrows))

    for i, row in enumerate(selected, start=1):
        img = keras.utils.load_img(row["filepath"], target_size=IMAGE_SIZE)
        arr = keras.utils.img_to_array(img).astype(np.uint8)
        plt.subplot(nrows, ncols, i)
        plt.imshow(arr)
        plt.title(row["class_name"], fontsize=8)
        plt.axis("off")

    save_plot(out)


def plot_preprocess_demo(sample_path: str, out: Path) -> None:
    raw = keras.utils.load_img(sample_path)
    raw_arr = keras.utils.img_to_array(raw).astype(np.uint8)

    resized = keras.utils.load_img(sample_path, target_size=IMAGE_SIZE)
    resized_arr = keras.utils.img_to_array(resized).astype(np.float32)
    norm_arr = resized_arr / 255.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(raw_arr)
    plt.title(f"Original ({raw_arr.shape[1]}×{raw_arr.shape[0]})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(resized_arr.astype(np.uint8))
    plt.title(f"Resized ({IMAGE_SIZE[0]}×{IMAGE_SIZE[1]})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.hist(norm_arr.flatten(), bins=32, color="#0E9F6E")
    plt.title("Normalized Pixel Histogram")
    plt.xlabel("Pixel Value")

    save_plot(out)


# ══════════════════════════════════════════════════════════════════════════════
# Split & tf.data
# ══════════════════════════════════════════════════════════════════════════════
def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["class_name"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["class_name"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def plot_split_distribution(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out: Path
) -> None:
    merged = pd.concat(
        [
            train_df["class_name"].value_counts().rename("train"),
            val_df["class_name"].value_counts().rename("val"),
            test_df["class_name"].value_counts().rename("test"),
        ],
        axis=1,
    ).fillna(0).astype(int).sort_index()

    merged.plot(kind="bar", stacked=True, figsize=(14, 6), color=["#1D4ED8", "#16A34A", "#F59E0B"])
    plt.title("Train / Val / Test Distribution per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    save_plot(out)


def decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, try_recover_truncated=True)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def make_dataset(
    df: pd.DataFrame, class_to_idx: dict[str, int], batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    labels = df["class_name"].map(class_to_idx).values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, labels))
    ds = ds.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Augmentation ──────────────────────────────────────────────────────────────
def build_augmenter() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.15),
        ],
        name="augmenter",
    )


def apply_augmentation(ds: tf.data.Dataset, augmenter: keras.Sequential) -> tf.data.Dataset:
    return ds.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def plot_augmentation_demo(train_df: pd.DataFrame, augmenter: keras.Sequential, out: Path) -> None:
    sample_path = train_df.iloc[0]["filepath"]
    img = keras.utils.load_img(sample_path, target_size=IMAGE_SIZE)
    x = np.expand_dims(keras.utils.img_to_array(img).astype(np.float32), axis=0)

    plt.figure(figsize=(14, 6))
    plt.subplot(2, 4, 1)
    plt.imshow((x[0] / 255.0).clip(0, 1))
    plt.title("Original")
    plt.axis("off")

    for i in range(2, 9):
        aug = augmenter(x, training=True)[0].numpy()
        plt.subplot(2, 4, i)
        plt.imshow((aug / 255.0).clip(0, 1))
        plt.title(f"Aug #{i - 1}")
        plt.axis("off")

    save_plot(out)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base.trainable = False

    inp = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image")
    x = keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inp, out, name="batik_mobilenetv2")
    return model, base


def plot_trainable_overview(base_model: keras.Model, out: Path) -> None:
    frozen = sum(1 for l in base_model.layers if not l.trainable)
    trainable = sum(1 for l in base_model.layers if l.trainable)

    plt.figure(figsize=(6, 4))
    plt.bar(["Frozen", "Trainable"], [frozen, trainable], color=["#6B7280", "#2563EB"])
    plt.title("Backbone Layer Status")
    plt.ylabel("Layer Count")
    save_plot(out)


# ── Training history ──────────────────────────────────────────────────────────
def merge_histories(
    head_hist: keras.callbacks.History,
    fine_hist: keras.callbacks.History,
) -> pd.DataFrame:
    rows = []
    for i in range(len(head_hist.history["loss"])):
        lr_list = head_hist.history.get("learning_rate", head_hist.history.get("lr", [np.nan]))
        rows.append({
            "epoch": i + 1,
            "phase": "head",
            "loss": head_hist.history["loss"][i],
            "val_loss": head_hist.history["val_loss"][i],
            "accuracy": head_hist.history["accuracy"][i],
            "val_accuracy": head_hist.history["val_accuracy"][i],
            "lr": float(lr_list[i]) if i < len(lr_list) else np.nan,
        })

    base = len(rows)
    for i in range(len(fine_hist.history["loss"])):
        lr_list = fine_hist.history.get("learning_rate", fine_hist.history.get("lr", [np.nan]))
        rows.append({
            "epoch": base + i + 1,
            "phase": "fine_tune",
            "loss": fine_hist.history["loss"][i],
            "val_loss": fine_hist.history["val_loss"][i],
            "accuracy": fine_hist.history["accuracy"][i],
            "val_accuracy": fine_hist.history["val_accuracy"][i],
            "lr": float(lr_list[i]) if i < len(lr_list) else np.nan,
        })

    return pd.DataFrame(rows)


def plot_training_curves(history_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history_df["epoch"], history_df["loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["lr"], color="#7C3AED")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")

    save_plot(out)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    save_plot(out)


def plot_report_heatmap(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out: Path
) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metric_df = pd.DataFrame(report).T
    metric_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    per_class = metric_df.loc[class_names, ["precision", "recall", "f1-score"]]
    plt.figure(figsize=(8, max(4, len(class_names) * 0.35)))
    sns.heatmap(per_class, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Per-Class Precision / Recall / F1")
    save_plot(out)

    return metric_df


def plot_misclassified_samples(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    out: Path,
    max_items: int,
) -> None:
    y_true = test_df["class_name"].map(class_to_idx).values
    y_pred = np.argmax(probs, axis=1)
    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("  ✅ No misclassified samples!")
        return

    wrong_idx = wrong_idx[:max_items]
    n = len(wrong_idx)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, wi in enumerate(wrong_idx, start=1):
        row = test_df.iloc[wi]
        img = keras.utils.load_img(row["filepath"], target_size=IMAGE_SIZE)
        arr = keras.utils.img_to_array(img).astype(np.uint8)
        pred_idx = int(y_pred[wi])

        plt.subplot(nrows, ncols, i)
        plt.imshow(arr)
        plt.title(
            f"T: {row['class_name']}\nP: {idx_to_class[pred_idx]} ({probs[wi, pred_idx]:.2f})",
            fontsize=8,
        )
        plt.axis("off")

    save_plot(out)


# ══════════════════════════════════════════════════════════════════════════════
# TFLite Conversion & Benchmark
# ══════════════════════════════════════════════════════════════════════════════
def representative_dataset(ds: tf.data.Dataset):
    for images, _ in ds.take(100):
        for i in range(images.shape[0]):
            yield [tf.expand_dims(images[i], axis=0)]


def convert_tflite_models(
    saved_model_dir: Path, train_ds: tf.data.Dataset
) -> dict[str, Path]:
    tflite_paths: dict[str, Path] = {}

    # 1. Dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dyn = converter.convert()
    p = MODEL_DIR / "model_dynamic.tflite"
    p.write_bytes(dyn)
    tflite_paths["dynamic"] = p
    print(f"  ✅ Dynamic:  {bytes_to_mb(len(dyn)):.2f} MB")

    # 2. Float16 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    fp16 = converter.convert()
    p = MODEL_DIR / "model_float16.tflite"
    p.write_bytes(fp16)
    tflite_paths["float16"] = p
    print(f"  ✅ Float16:  {bytes_to_mb(len(fp16)):.2f} MB")

    # 3. INT8 full-integer quantization (may fail on some models)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(train_ds)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        int8 = converter.convert()
        p = MODEL_DIR / "model_int8.tflite"
        p.write_bytes(int8)
        tflite_paths["int8"] = p
        print(f"  ✅ INT8:     {bytes_to_mb(len(int8)):.2f} MB")
    except Exception as e:
        print(f"  ⚠️  INT8 conversion skipped: {e}")

    return tflite_paths


def preprocess_for_tflite(path: str) -> np.ndarray:
    img = keras.utils.load_img(path, target_size=IMAGE_SIZE)
    arr = keras.utils.img_to_array(img).astype(np.float32)
    return np.expand_dims(arr, axis=0)


def run_tflite_inference(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    x = input_array.copy()

    # Quantize input if needed (int8/uint8)
    if in_d["dtype"] in (np.int8, np.uint8):
        scale, zp = in_d["quantization"]
        scale = scale if scale != 0 else 1.0
        x = np.round(x / scale + zp).astype(in_d["dtype"])
    else:
        x = x.astype(in_d["dtype"])

    interpreter.set_tensor(in_d["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_d["index"])

    # Dequantize output if needed
    if out_d["dtype"] in (np.int8, np.uint8):
        scale, zp = out_d["quantization"]
        scale = scale if scale != 0 else 1.0
        out = (out.astype(np.float32) - zp) * scale

    return out


def benchmark_models(
    tf_model: keras.Model,
    test_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    tflite_paths: dict[str, Path],
    out: Path,
) -> pd.DataFrame:
    subset = test_df.sample(
        n=min(BENCHMARK_SAMPLES, len(test_df)), random_state=SEED
    ).reset_index(drop=True)
    y_true = subset["class_name"].map(class_to_idx).values

    # TF Keras model benchmark
    tf_preds, tf_times = [], []
    for p in subset["filepath"]:
        x = preprocess_for_tflite(p)
        t0 = time.perf_counter()
        out_v = tf_model(x, training=False).numpy()
        tf_times.append((time.perf_counter() - t0) * 1000)
        tf_preds.append(int(np.argmax(out_v[0])))

    rows = [{
        "model": "tensorflow",
        "accuracy": float((np.array(tf_preds) == y_true).mean()),
        "latency_ms": float(np.mean(tf_times)),
    }]

    # TFLite models benchmark
    for name, path in tflite_paths.items():
        preds, times = [], []
        for p in subset["filepath"]:
            x = preprocess_for_tflite(p)
            t0 = time.perf_counter()
            out_v = run_tflite_inference(path, x)
            times.append((time.perf_counter() - t0) * 1000)
            preds.append(int(np.argmax(out_v[0])))
        rows.append({
            "model": f"tflite_{name}",
            "accuracy": float((np.array(preds) == y_true).mean()),
            "latency_ms": float(np.mean(times)),
        })

    bench_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=bench_df, x="model", y="accuracy", ax=axes[0], color="#2563EB")
    axes[0].set_title("Accuracy: TF vs TFLite")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=bench_df, x="model", y="latency_ms", ax=axes[1], color="#16A34A")
    axes[1].set_title("Latency (ms): TF vs TFLite")
    axes[1].tick_params(axis="x", rotation=30)

    save_plot(out)
    return bench_df


def recursive_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def plot_model_sizes(
    saved_model_dir: Path, tflite_paths: dict[str, Path], out: Path
) -> pd.DataFrame:
    rows = [{"artifact": "saved_model", "size_mb": bytes_to_mb(recursive_size(saved_model_dir))}]
    for name, path in tflite_paths.items():
        rows.append({"artifact": f"tflite_{name}", "size_mb": bytes_to_mb(path.stat().st_size)})

    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="artifact", y="size_mb", color="#F59E0B")
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")
    plt.xticks(rotation=25)
    save_plot(out)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline (10 Steps)
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    t_start = time.time()
    set_seed(SEED)
    ensure_dirs()

    # ── Environment ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Batik Wastra Nusantara – TFLite Pipeline")
    print("=" * 60)
    print_gpu_info()
    print()

    # ── [1/10] Download ───────────────────────────────────────────────────────
    print("[1/10] Downloading dataset from Kaggle...")
    raw_root = download_kaggle_dataset(KAGGLE_DATASET)
    class_root = find_class_root(raw_root)
    print(f"  Dataset root: {class_root}")

    # ── [2/10] Data audit & filter ────────────────────────────────────────────
    print("[2/10] Auditing data and filtering classes ({}-{} images)...".format(MIN_IMAGES, MAX_IMAGES))
    class_df, image_df = build_metadata(class_root)
    class_df.to_csv(OUTPUT_DIR / "class_counts_all.csv", index=False)
    plot_class_distribution(class_df, "All Class Distribution", FIG_DIR / "01_class_distribution_all.png")

    selected_classes = select_classes_by_range(class_df, MIN_IMAGES, MAX_IMAGES)
    if len(selected_classes) < 2:
        raise ValueError(f"Only {len(selected_classes)} classes in range [{MIN_IMAGES}, {MAX_IMAGES}]. Adjust filter.")

    selected_df = image_df[image_df["class_name"].isin(selected_classes)].reset_index(drop=True)
    selected_df, invalid_df = filter_invalid_images(selected_df)
    if not invalid_df.empty:
        invalid_df.to_csv(OUTPUT_DIR / "invalid_images.csv", index=False)
        print(f"  ⚠️  Removed {len(invalid_df)} corrupt images  →  invalid_images.csv")

    selected_counts = (
        selected_df["class_name"]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if selected_counts["class_name"].nunique() < 2:
        raise ValueError("< 2 valid classes remaining after image validation.")

    selected_counts.to_csv(OUTPUT_DIR / "class_counts_selected.csv", index=False)
    plot_class_distribution(
        selected_counts,
        f"Selected Classes ({MIN_IMAGES}–{MAX_IMAGES} images)",
        FIG_DIR / "02_class_distribution_selected.png",
    )
    print(f"  ✅ {len(selected_classes)} classes, {len(selected_df)} images selected")

    # ── [3/10] Sample visualization ───────────────────────────────────────────
    print("[3/10] Generating sample visualization...")
    plot_sample_grid(
        selected_df, selected_classes,
        FIG_DIR / "03_sample_images_per_class.png",
        per_class=RANDOM_SAMPLES_PER_CLASS,
    )

    # ── [4/10] Preprocess demo ────────────────────────────────────────────────
    print("[4/10] Generating preprocess demo...")
    plot_preprocess_demo(selected_df.iloc[0]["filepath"], FIG_DIR / "04_preprocess_demo.png")

    # ── [5/10] Split dataset ──────────────────────────────────────────────────
    print("[5/10] Splitting dataset (80/10/10)...")
    train_df, val_df, test_df = split_dataset(selected_df)
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    plot_split_distribution(train_df, val_df, test_df, FIG_DIR / "05_split_distribution.png")

    class_names = sorted(selected_counts["class_name"].tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    (OUTPUT_DIR / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # ── [6/10] Build tf.data & augmentation ───────────────────────────────────
    print("[6/10] Building tf.data pipelines & augmentation demo...")
    train_ds = make_dataset(train_df, class_to_idx, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val_df, class_to_idx, BATCH_SIZE, shuffle=False)
    test_ds = make_dataset(test_df, class_to_idx, BATCH_SIZE, shuffle=False)

    augmenter = build_augmenter()
    plot_augmentation_demo(train_df, augmenter, FIG_DIR / "06_augmentation_demo.png")
    train_ds_aug = apply_augmentation(train_ds, augmenter)

    # ── [7/10] Build & train model ────────────────────────────────────────────
    print("[7/10] Building & training model...")
    model, base_model = build_model(num_classes=len(class_names))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_head.keras"), monitor="val_loss", save_best_only=True
        ),
    ]

    plot_trainable_overview(base_model, FIG_DIR / "07_backbone_status_phase_head.png")
    print("  Phase 1: Head-only training...")
    head_hist = model.fit(
        train_ds_aug, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks, verbose=1
    )

    # Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAST_N]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    plot_trainable_overview(base_model, FIG_DIR / "08_backbone_status_phase_finetune.png")
    print("  Phase 2: Fine-tuning last {} layers...".format(FINE_TUNE_LAST_N))
    fine_hist = model.fit(
        train_ds_aug, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks, verbose=1
    )

    history_df = merge_histories(head_hist, fine_hist)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    plot_training_curves(history_df, FIG_DIR / "09_training_curves.png")

    # ── [8/10] Evaluate ───────────────────────────────────────────────────────
    print("[8/10] Evaluating model on test set...")
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_df["class_name"].map(class_to_idx).values

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    plot_confusion(y_true, y_pred, class_names, FIG_DIR / "10_confusion_matrix.png")
    report_df = plot_report_heatmap(y_true, y_pred, class_names, FIG_DIR / "11_classification_report_heatmap.png")
    plot_misclassified_samples(
        test_df, probs, idx_to_class, class_to_idx,
        FIG_DIR / "12_misclassified_examples.png", MAX_MISCLASSIFIED_TO_SHOW,
    )

    # Save Keras model + SavedModel
    model.save(MODEL_DIR / "final_model.keras")
    saved_model_dir = MODEL_DIR / "saved_model"
    model.export(str(saved_model_dir))

    # ── [9/10] TFLite conversion ──────────────────────────────────────────────
    print("[9/10] Converting to TFLite (dynamic, float16, int8)...")
    tflite_paths = convert_tflite_models(saved_model_dir, train_ds)

    # ── [10/10] Benchmark & summary ───────────────────────────────────────────
    print("[10/10] Benchmarking & saving summary...")
    size_df = plot_model_sizes(saved_model_dir, tflite_paths, FIG_DIR / "13_model_size_comparison.png")
    bench_df = benchmark_models(
        model, test_df, class_to_idx, tflite_paths,
        FIG_DIR / "14_tf_vs_tflite_benchmark.png",
    )

    size_df.to_csv(OUTPUT_DIR / "model_size_comparison.csv", index=False)
    bench_df.to_csv(OUTPUT_DIR / "tf_vs_tflite_benchmark.csv", index=False)

    summary = {
        "dataset_slug": KAGGLE_DATASET,
        "dataset_root": str(class_root),
        "selected_classes": class_names,
        "num_selected_classes": len(class_names),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_f1": float(report_df.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report_df.loc["weighted avg", "f1-score"]),
        "tflite_models": {k: str(v) for k, v in tflite_paths.items()},
        "benchmark": bench_df.to_dict(orient="records"),
        "model_sizes": size_df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  ✅ Pipeline completed in {elapsed / 60:.1f} minutes")
    print(f"  📁 Figures : {FIG_DIR}")
    print(f"  📁 Models  : {MODEL_DIR}")
    print(f"  📁 Outputs : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
