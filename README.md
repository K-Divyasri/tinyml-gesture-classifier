# TinyML Gesture Classifier

An end-to-end TinyML pipeline: an ESP32 reads an MPU6050 accelerometer/gyroscope over I2C, a quantized neural network classifies which of four gestures (idle, wave, circle, punch) just happened, and inference runs entirely on-device, no cloud round-trip, no internet connection.

## Pipeline

```
MPU6050 (I2C) --> ESP32 data_collector.ino --> CSV rows --> imu_gesture_log.csv
                                                                    |
                                                              model/train.py
                                                  (Keras train, int8 quantize,
                                                   export .tflite + C headers)
                                                                    |
                                       gesture_model_data.h / gesture_labels.h / metrics.json
                                                                    |
                                          firmware/tinyml_gesture/tinyml_gesture.ino
                                    (MPU6050 -> sliding 1s window -> TFLite Micro inference)
```

## What's verified for real, and what needs a bench

1. **Verified end to end on this machine**: the full ML pipeline (data generation, training, int8 quantization) and both firmware sketches' compilation. Real `arduino-cli compile` runs against `esp32:esp32:esp32` produced `tinyml_gesture.ino`: 621,992 bytes (47%) flash / 40,980 bytes (12%) RAM, and `data_collector.ino`: 318,460 bytes (24%) flash / 23,636 bytes (7%) RAM.
2. **Written against the real APIs, not yet run on hardware**: the on-device inference loop's behavior with a physically wired MPU6050, and the data-collector's recording flow. Both compile clean but have not been flashed to a physical board.
3. **Needs a physical board**: soldering/breadboarding, an I2C continuity check, performing gestures with a real board in hand, and power measurement with a multimeter.

Real numbers, measured on this pipeline: 1.0000 float32 test accuracy vs. 0.85 int8 test accuracy after quantization, a genuine tradeoff rather than a hand-wave. The training data itself is synthetic (four parametrically-generated gesture signatures, `data/generate_data.py`, seed=42) until it's retrained on real recordings, so the 1.0000 float32 accuracy reflects how cleanly separable synthetic data is, not a claim about recognizing a real person's gestures out of the box.

## Run it

```powershell
pip install -r requirements.txt

python data/generate_data.py   # deterministic synthetic IMU data (seed=42)
python model/train.py          # train, quantize to int8, export .tflite + C headers

pytest -q                      # tests for the data generator and model artifacts, no hardware needed
```

To compile the firmware (no board required to just compile):

```powershell
arduino-cli core install esp32:esp32
arduino-cli lib install "ArduTFLite" "Adafruit MPU6050"

arduino-cli compile --fqbn esp32:esp32:esp32 firmware/tinyml_gesture
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/data_collector
```

To run it on real hardware: wire an MPU6050 to an ESP32 per `wiring/WIRING_GUIDE.md`, then `arduino-cli upload -p <PORT> --fqbn esp32:esp32:esp32 <sketch>`.

## What this demonstrates

- The full TinyML pipeline: sensor data, windowing, a small neural net trained in TensorFlow/Keras, post-training int8 quantization, a C byte-array header, TensorFlow Lite Micro running inference on a microcontroller with no OS.
- I2C sensor integration (MPU6050 accelerometer + gyroscope) targeting a real ESP32, not a simulator.
- Firmware written against real library APIs (ArduTFLite / `Chirale_TensorFlowLite`, Adafruit MPU6050) and compiled for real against the ESP32 toolchain via `arduino-cli`.

## Repo layout

```
tinyml-gesture-classifier/
├── data/            synthetic data generator + generated CSV log
├── model/           Keras training script, exported .tflite model, C headers, metrics
├── firmware/         tinyml_gesture (on-device inference) and data_collector (logging) sketches
├── tests/            pytest suite for the data generator and model artifacts
├── wiring/           wiring guide for the MPU6050 + ESP32
└── hosting/          CI workflow, hosting guide, and a Streamlit data-viewer/prediction dashboard
```

## Hosting

See `hosting/HOSTING_GUIDE.md`. A Streamlit app lets you upload or paste a real IMU recording CSV and visualize it, and run live inference against the quantized model if TensorFlow is available in the hosting environment; it's free to host on Streamlit Community Cloud or Hugging Face Spaces. A GitHub Actions workflow runs the pytest suite and compile-checks both firmware sketches on every push; CI can compile firmware but cannot flash or test it on real hardware, since there's no ESP32 attached to a GitHub-hosted runner.

## Stretch goals

- Record real gesture data from a few different people and retrain, then compare the real-data accuracy honestly against the synthetic 1.0000/0.85 numbers.
- Add a fifth class, e.g. "shake" or "tap", and see how much the int8 accuracy drop grows as the classes get harder to separate.
- Swap the Dense-only architecture for a small 1D-CNN over the raw window and compare parameter count, latency, and accuracy.
- Add an on-device confidence-based "unknown gesture" rejection instead of always picking the argmax class.
