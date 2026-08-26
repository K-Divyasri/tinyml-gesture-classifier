import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model"
FIRMWARE_DIR = Path(__file__).parent.parent / "firmware" / "tinyml_gesture"


def test_metrics_file_exists_and_is_sane():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    assert metrics["classes"] == ["idle", "wave", "circle", "punch"]
    assert 0.0 <= metrics["int8_tflite_test_accuracy"] <= 1.0
    assert 0.0 <= metrics["float32_test_accuracy"] <= 1.0
    # a real int8 quantization pass costs at least a little accuracy on a model
    # this small; if it ever matches float exactly, something didn't quantize
    assert metrics["int8_tflite_test_accuracy"] <= metrics["float32_test_accuracy"]
    assert metrics["tflite_model_bytes"] > 0


def test_tflite_model_file_is_a_flatbuffer():
    tflite_path = MODEL_DIR / "gesture_model_int8.tflite"
    data = tflite_path.read_bytes()
    assert len(data) > 100
    # TFLite flatbuffers carry the "TFL3" identifier at bytes 4:8
    assert data[4:8] == b"TFL3"


def test_generated_headers_match_between_model_and_firmware_dirs():
    for name in ("gesture_model_data.h", "gesture_labels.h"):
        model_copy = (MODEL_DIR / name).read_text()
        firmware_copy = (FIRMWARE_DIR / name).read_text()
        assert model_copy == firmware_copy, f"{name} drifted between model/ and firmware/ -- rerun train.py"


def test_labels_header_lists_four_classes_in_training_order():
    labels_h = (MODEL_DIR / "gesture_labels.h").read_text()
    assert "kNumGestureClasses = 4" in labels_h
    assert '"idle", "wave", "circle", "punch"' in labels_h
