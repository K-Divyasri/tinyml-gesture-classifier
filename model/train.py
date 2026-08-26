"""
Train a tiny gesture classifier on the MPU6050 IMU log, quantize it to int8
TFLite, and export a C byte-array header the firmware can compile straight in.

Pipeline: CSV (long format, one row per IMU sample) -> windowed feature matrix
(one row per recording, 6 axes x 50 samples = 300 features) -> small dense
network -> post-training int8 quantization -> tflite_model.h

Everything here is real: real Keras training, real accuracy numbers, real
TFLite conversion, real quantized model size. Nothing is hand-waved.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "imu_gesture_log.csv"
MODEL_DIR = Path(__file__).parent
FIRMWARE_DIR = Path(__file__).parent.parent / "firmware" / "tinyml_gesture"
CLASSES = ["idle", "wave", "circle", "punch"]
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
SAMPLES_PER_WINDOW = 50


def load_recordings():
    recordings = defaultdict(list)
    labels = {}
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["recording_id"])
            labels[rid] = row["label"]
            recordings[rid].append([float(row[a]) for a in AXES])

    X, y = [], []
    for rid in sorted(recordings):
        samples = recordings[rid]
        assert len(samples) == SAMPLES_PER_WINDOW, f"recording {rid} has {len(samples)} samples"
        X.append(np.array(samples, dtype=np.float32).flatten())  # 300 features
        y.append(CLASSES.index(labels[rid]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def split(X, y, rng):
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def build_model(input_dim, num_classes, mean, std):
    inputs = tf.keras.Input(shape=(input_dim,), name="imu_window")
    x = tf.keras.layers.Normalization(mean=mean, variance=std ** 2)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def representative_dataset_gen(X_train):
    def gen():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i + 1]]
    return gen


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    X, y = load_recordings()
    print(f"loaded {len(X)} recordings, {X.shape[1]} features each")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split(X, y, rng)
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6

    model = build_model(X.shape[1], len(CLASSES), mean, std)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        verbose=2,
    )

    float_loss, float_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nfloat32 model test accuracy: {float_acc:.4f}")

    # --- convert to TFLite with full int8 post-training quantization ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    tflite_path = MODEL_DIR / "gesture_model_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"\nwrote {tflite_path} ({len(tflite_model)} bytes)")

    # --- evaluate the quantized model for real, not just assume it matches ---
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0
    for i in range(len(X_test)):
        interpreter.set_tensor(input_details["index"], X_test[i:i + 1])
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(output_details["index"])[0])
        if pred == y_test[i]:
            correct += 1
    quant_acc = correct / len(X_test)
    print(f"int8 tflite model test accuracy: {quant_acc:.4f}")

    # --- export as a C header byte array for the firmware to #include ---
    header_path = MODEL_DIR / "gesture_model_data.h"
    var_name = "g_gesture_model_data"
    with open(header_path, "w") as f:
        f.write("// Auto-generated by train.py. Do not edit by hand.\n")
        f.write("#ifndef GESTURE_MODEL_DATA_H_\n#define GESTURE_MODEL_DATA_H_\n\n")
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(tflite_model), 12):
            chunk = tflite_model[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f"const int {var_name}_len = {len(tflite_model)};\n\n")
        f.write("#endif  // GESTURE_MODEL_DATA_H_\n")
    print(f"wrote {header_path}")

    # the firmware sketch can only #include headers that live in its own sketch
    # folder, so the generated headers are copied there too -- same content,
    # two locations, single source of truth (this script)
    FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)
    (FIRMWARE_DIR / header_path.name).write_text(header_path.read_text())

    # --- write the class labels the firmware needs, in the same order as training ---
    labels_path = MODEL_DIR / "gesture_labels.h"
    with open(labels_path, "w") as f:
        f.write("// Auto-generated by train.py. Do not edit by hand.\n")
        f.write("#ifndef GESTURE_LABELS_H_\n#define GESTURE_LABELS_H_\n\n")
        f.write(f"const int kNumGestureClasses = {len(CLASSES)};\n")
        f.write("const char* kGestureLabels[] = {" + ", ".join(f'"{c}"' for c in CLASSES) + "};\n\n")
        f.write("#endif  // GESTURE_LABELS_H_\n")
    print(f"wrote {labels_path}")
    (FIRMWARE_DIR / labels_path.name).write_text(labels_path.read_text())

    metrics = {
        "float32_test_accuracy": float_acc,
        "int8_tflite_test_accuracy": quant_acc,
        "float32_test_loss": float_loss,
        "tflite_model_bytes": len(tflite_model),
        "num_train": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test),
        "classes": CLASSES,
    }
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")
    print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
