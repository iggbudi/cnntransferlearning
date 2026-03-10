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


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("_DATASET")
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

MIN_IMAGES = 30
MAX_IMAGES = 50
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

EPOCHS_HEAD = 12
EPOCHS_FINE = 10
FINE_TUNE_LAST_N = 30

RANDOM_SAMPLES_PER_CLASS = 1
MAX_MISCLASSIFIED_TO_SHOW = 16
BENCHMARK_SAMPLES = 120

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(p)
    return files


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def filter_invalid_images(image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_mask = image_df["filepath"].map(is_valid_image)
    valid_df = image_df[valid_mask].reset_index(drop=True)
    invalid_df = image_df[~valid_mask].reset_index(drop=True)
    return valid_df, invalid_df


def recursive_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# -----------------------------
# Data audit and selection
# -----------------------------
def build_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows = []
    image_rows = []

    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        imgs = list_images(class_dir)
        class_rows.append({"class_name": class_dir.name, "count": len(imgs)})
        for img in imgs:
            image_rows.append({"filepath": str(img), "class_name": class_dir.name})

    class_df = pd.DataFrame(class_rows).sort_values("count", ascending=False).reset_index(drop=True)
    image_df = pd.DataFrame(image_rows)
    return class_df, image_df


def plot_class_distribution(df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(12, max(4, len(df) * 0.3)))
    sns.barplot(data=df, x="count", y="class_name", color="#2D7FF9")
    plt.title(title)
    plt.xlabel("Image Count")
    plt.ylabel("Class")
    save_plot(out_path)


def select_classes_by_range(class_df: pd.DataFrame, min_images: int, max_images: int) -> list[str]:
    mask = (class_df["count"] >= min_images) & (class_df["count"] <= max_images)
    return class_df.loc[mask, "class_name"].tolist()


def plot_sample_grid(image_df: pd.DataFrame, class_names: list[str], out_path: Path, per_class: int = 1) -> None:
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

    save_plot(out_path)


def plot_preprocess_demo(sample_path: str, out_path: Path) -> None:
    raw = keras.utils.load_img(sample_path)
    raw_arr = keras.utils.img_to_array(raw).astype(np.uint8)

    resized = keras.utils.load_img(sample_path, target_size=IMAGE_SIZE)
    resized_arr = keras.utils.img_to_array(resized).astype(np.float32)

    norm_arr = resized_arr / 255.0

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(raw_arr)
    plt.title(f"Original ({raw_arr.shape[1]}x{raw_arr.shape[0]})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(resized_arr.astype(np.uint8))
    plt.title(f"Resized ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.hist(norm_arr.flatten(), bins=32, color="#0E9F6E")
    plt.title("Normalized Pixel Histogram")
    plt.xlabel("Pixel Value")

    save_plot(out_path)


# -----------------------------
# Split and tf.data
# -----------------------------
def split_dataset(image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        image_df,
        test_size=0.2,
        random_state=SEED,
        stratify=image_df["class_name"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["class_name"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def plot_split_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path) -> None:
    train_counts = train_df["class_name"].value_counts().rename("train")
    val_counts = val_df["class_name"].value_counts().rename("val")
    test_counts = test_df["class_name"].value_counts().rename("test")

    merged = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    merged = merged.sort_index()

    merged.plot(kind="bar", stacked=True, figsize=(14, 6), color=["#1D4ED8", "#16A34A", "#F59E0B"])
    plt.title("Train/Val/Test Distribution per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    save_plot(out_path)


def decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def make_dataset(df: pd.DataFrame, class_to_idx: dict[str, int], batch_size: int, shuffle: bool) -> tf.data.Dataset:
    labels = df["class_name"].map(class_to_idx).values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, labels))
    ds = ds.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


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
    return ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)


def plot_augmentation_demo(train_df: pd.DataFrame, augmenter: keras.Sequential, out_path: Path) -> None:
    sample_path = train_df.iloc[0]["filepath"]
    img = keras.utils.load_img(sample_path, target_size=IMAGE_SIZE)
    x = keras.utils.img_to_array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 4, 1)
    plt.imshow((x[0] / 255.0).clip(0, 1))
    plt.title("Original")
    plt.axis("off")

    for i in range(2, 9):
        aug = augmenter(x, training=True)[0].numpy()
        plt.subplot(2, 4, i)
        plt.imshow((aug / 255.0).clip(0, 1))
        plt.title(f"Aug #{i-1}")
        plt.axis("off")

    save_plot(out_path)


# -----------------------------
# Model
# -----------------------------
def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image")
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name="batik_mobilenetv2")
    return model, base_model


def plot_trainable_overview(base_model: keras.Model, out_path: Path) -> None:
    frozen_count = sum([0 if l.trainable else 1 for l in base_model.layers])
    trainable_count = sum([1 if l.trainable else 0 for l in base_model.layers])

    plt.figure(figsize=(6, 4))
    plt.bar(["Frozen", "Trainable"], [frozen_count, trainable_count], color=["#6B7280", "#2563EB"])
    plt.title("Backbone Layer Status")
    plt.ylabel("Layer Count")
    save_plot(out_path)


def merge_histories(head_hist: keras.callbacks.History, fine_hist: keras.callbacks.History) -> pd.DataFrame:
    rows = []

    for i in range(len(head_hist.history["loss"])):
        rows.append(
            {
                "epoch": i + 1,
                "phase": "head",
                "loss": head_hist.history["loss"][i],
                "val_loss": head_hist.history["val_loss"][i],
                "accuracy": head_hist.history["accuracy"][i],
                "val_accuracy": head_hist.history["val_accuracy"][i],
                "lr": float(head_hist.history.get("learning_rate", [np.nan])[i]),
            }
        )

    base_epoch = len(rows)
    for i in range(len(fine_hist.history["loss"])):
        rows.append(
            {
                "epoch": base_epoch + i + 1,
                "phase": "fine_tune",
                "loss": fine_hist.history["loss"][i],
                "val_loss": fine_hist.history["val_loss"][i],
                "accuracy": fine_hist.history["accuracy"][i],
                "val_accuracy": fine_hist.history["val_accuracy"][i],
                "lr": float(fine_hist.history.get("learning_rate", [np.nan])[i]),
            }
        )

    return pd.DataFrame(rows)


def plot_training_curves(history_df: pd.DataFrame, out_path: Path) -> None:
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

    save_plot(out_path)


# -----------------------------
# Evaluation
# -----------------------------
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    save_plot(out_path)


def plot_report_heatmap(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_path: Path) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metric_df = pd.DataFrame(report).T
    metric_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    per_class = metric_df.loc[class_names, ["precision", "recall", "f1-score"]]

    plt.figure(figsize=(8, max(4, len(class_names) * 0.35)))
    sns.heatmap(per_class, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Per-Class Precision/Recall/F1")
    save_plot(out_path)

    return metric_df


def plot_misclassified_samples(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    out_path: Path,
    max_items: int,
) -> None:
    y_true = test_df["class_name"].map(class_to_idx).values
    y_pred = np.argmax(probs, axis=1)

    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
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

    save_plot(out_path)


# -----------------------------
# TFLite
# -----------------------------
def representative_dataset(ds: tf.data.Dataset):
    for images, _ in ds.take(100):
        for i in range(images.shape[0]):
            yield [tf.expand_dims(images[i], axis=0)]


def convert_tflite_models(saved_model_dir: Path, train_ds: tf.data.Dataset) -> dict[str, Path]:
    tflite_paths = {}

    # Dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_model = converter.convert()
    dynamic_path = MODEL_DIR / "model_dynamic.tflite"
    dynamic_path.write_bytes(dynamic_model)
    tflite_paths["dynamic"] = dynamic_path

    # Float16 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    float16_model = converter.convert()
    float16_path = MODEL_DIR / "model_float16.tflite"
    float16_path.write_bytes(float16_model)
    tflite_paths["float16"] = float16_path

    # Int8 quantization (fallback-safe)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(train_ds)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        int8_model = converter.convert()
        int8_path = MODEL_DIR / "model_int8.tflite"
        int8_path.write_bytes(int8_model)
        tflite_paths["int8"] = int8_path
    except Exception as e:
        print(f"[WARN] INT8 conversion skipped: {e}")

    return tflite_paths


def preprocess_for_tflite(path: str) -> np.ndarray:
    img = keras.utils.load_img(path, target_size=IMAGE_SIZE)
    arr = keras.utils.img_to_array(img).astype(np.float32)
    return np.expand_dims(arr, axis=0)


def run_tflite_inference(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    x = input_array.copy()

    # Quantize input if needed
    if in_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = in_detail["quantization"]
        if scale == 0:
            scale = 1.0
        x = np.round(x / scale + zp).astype(in_detail["dtype"])
    else:
        x = x.astype(in_detail["dtype"])

    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()

    out = interpreter.get_tensor(out_detail["index"])

    # Dequantize output if needed
    if out_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = out_detail["quantization"]
        if scale == 0:
            scale = 1.0
        out = (out.astype(np.float32) - zp) * scale

    return out


def benchmark_models(
    tf_model: keras.Model,
    test_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    tflite_paths: dict[str, Path],
    out_path: Path,
) -> pd.DataFrame:
    subset = test_df.sample(n=min(BENCHMARK_SAMPLES, len(test_df)), random_state=SEED).reset_index(drop=True)

    y_true = subset["class_name"].map(class_to_idx).values

    tf_preds = []
    tf_times = []
    for p in subset["filepath"]:
        x = preprocess_for_tflite(p)
        t0 = time.perf_counter()
        out = tf_model(x, training=False).numpy()
        tf_times.append((time.perf_counter() - t0) * 1000)
        tf_preds.append(int(np.argmax(out[0])))

    rows = [
        {
            "model": "tensorflow",
            "accuracy": float((np.array(tf_preds) == y_true).mean()),
            "latency_ms": float(np.mean(tf_times)),
        }
    ]

    for name, path in tflite_paths.items():
        preds = []
        times = []
        for p in subset["filepath"]:
            x = preprocess_for_tflite(p)
            t0 = time.perf_counter()
            out = run_tflite_inference(path, x)
            times.append((time.perf_counter() - t0) * 1000)
            preds.append(int(np.argmax(out[0])))

        rows.append(
            {
                "model": f"tflite_{name}",
                "accuracy": float((np.array(preds) == y_true).mean()),
                "latency_ms": float(np.mean(times)),
            }
        )

    bench_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=bench_df, x="model", y="accuracy", ax=axes[0], color="#2563EB")
    axes[0].set_title("Accuracy: TF vs TFLite")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=bench_df, x="model", y="latency_ms", ax=axes[1], color="#16A34A")
    axes[1].set_title("Latency (ms): TF vs TFLite")
    axes[1].tick_params(axis="x", rotation=30)

    save_plot(out_path)

    return bench_df


def plot_model_sizes(saved_model_dir: Path, tflite_paths: dict[str, Path], out_path: Path) -> pd.DataFrame:
    rows = [{"artifact": "saved_model", "size_mb": bytes_to_mb(recursive_size(saved_model_dir))}]
    for name, path in tflite_paths.items():
        rows.append({"artifact": f"tflite_{name}", "size_mb": bytes_to_mb(path.stat().st_size)})

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="artifact", y="size_mb", color="#F59E0B")
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")
    plt.xticks(rotation=25)
    save_plot(out_path)

    return df


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    set_seed(SEED)
    ensure_dirs()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    print("[1/10] Data audit...")
    class_df, image_df = build_metadata(DATA_DIR)
    class_df.to_csv(OUTPUT_DIR / "class_counts_all.csv", index=False)
    plot_class_distribution(class_df, "All Class Distribution", FIG_DIR / "01_class_distribution_all.png")

    selected_classes = select_classes_by_range(class_df, MIN_IMAGES, MAX_IMAGES)
    if len(selected_classes) < 2:
        raise ValueError("Selected class count is < 2. Adjust MIN_IMAGES / MAX_IMAGES.")

    selected_df = image_df[image_df["class_name"].isin(selected_classes)].reset_index(drop=True)
    selected_df, invalid_df = filter_invalid_images(selected_df)
    if not invalid_df.empty:
        invalid_df.to_csv(OUTPUT_DIR / "invalid_images.csv", index=False)
        print(f"[WARN] Removed invalid images: {len(invalid_df)} (saved to outputs/invalid_images.csv)")

    selected_counts = (
        selected_df["class_name"].value_counts().rename_axis("class_name").reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if selected_counts["class_name"].nunique() < 2:
        raise ValueError("Remaining valid classes are < 2 after invalid-image filtering.")

    selected_counts.to_csv(OUTPUT_DIR / "class_counts_selected.csv", index=False)
    plot_class_distribution(selected_counts, "Selected Class Distribution (30-50)", FIG_DIR / "02_class_distribution_selected.png")

    print("[2/10] Sample visualization...")
    plot_sample_grid(
        selected_df,
        selected_classes,
        FIG_DIR / "03_sample_images_per_class.png",
        per_class=RANDOM_SAMPLES_PER_CLASS,
    )

    print("[3/10] Preprocess demo...")
    first_sample = selected_df.iloc[0]["filepath"]
    plot_preprocess_demo(first_sample, FIG_DIR / "04_preprocess_demo.png")

    print("[4/10] Split dataset...")
    train_df, val_df, test_df = split_dataset(selected_df)
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    plot_split_distribution(train_df, val_df, test_df, FIG_DIR / "05_split_distribution.png")

    class_names = sorted(selected_counts["class_name"].tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    (OUTPUT_DIR / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print("[5/10] Build tf.data and augmentation...")
    train_ds = make_dataset(train_df, class_to_idx, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val_df, class_to_idx, BATCH_SIZE, shuffle=False)
    test_ds = make_dataset(test_df, class_to_idx, BATCH_SIZE, shuffle=False)

    augmenter = build_augmenter()
    plot_augmentation_demo(train_df, augmenter, FIG_DIR / "06_augmentation_demo.png")
    train_ds_aug = apply_augmentation(train_ds, augmenter)

    print("[6/10] Build and train model...")
    model, base_model = build_model(num_classes=len(class_names))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_head.keras"), monitor="val_loss", save_best_only=True),
    ]

    plot_trainable_overview(base_model, FIG_DIR / "07_backbone_status_phase_head.png")

    head_hist = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning stage
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAST_N]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    plot_trainable_overview(base_model, FIG_DIR / "08_backbone_status_phase_finetune.png")

    fine_hist = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = merge_histories(head_hist, fine_hist)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    plot_training_curves(history_df, FIG_DIR / "09_training_curves.png")

    print("[7/10] Evaluate model...")
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_df["class_name"].map(class_to_idx).values

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    plot_confusion(y_true, y_pred, class_names, FIG_DIR / "10_confusion_matrix.png")
    report_df = plot_report_heatmap(y_true, y_pred, class_names, FIG_DIR / "11_classification_report_heatmap.png")
    plot_misclassified_samples(
        test_df,
        probs,
        idx_to_class,
        class_to_idx,
        FIG_DIR / "12_misclassified_examples.png",
        MAX_MISCLASSIFIED_TO_SHOW,
    )

    model_path = MODEL_DIR / "final_model.keras"
    model.save(model_path)

    saved_model_dir = MODEL_DIR / "saved_model"
    model.export(str(saved_model_dir))

    print("[8/10] Convert to TFLite...")
    tflite_paths = convert_tflite_models(saved_model_dir, train_ds)

    print("[9/10] Compare model sizes and runtime...")
    size_df = plot_model_sizes(saved_model_dir, tflite_paths, FIG_DIR / "13_model_size_comparison.png")
    bench_df = benchmark_models(
        model,
        test_df,
        class_to_idx,
        tflite_paths,
        FIG_DIR / "14_tf_vs_tflite_benchmark.png",
    )

    print("[10/10] Save summary...")
    summary = {
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
    }

    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    size_df.to_csv(OUTPUT_DIR / "model_size_comparison.csv", index=False)
    bench_df.to_csv(OUTPUT_DIR / "tf_vs_tflite_benchmark.csv", index=False)

    print("Pipeline complete.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
