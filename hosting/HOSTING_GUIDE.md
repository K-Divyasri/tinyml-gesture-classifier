# Hosting Guide -- TinyML Gesture Classifier

This project doesn't have a natural "always-on cloud service" the way a web
app or an API would. What it actually produces is (1) a training pipeline
that turns an IMU CSV log into a quantized TFLite model, and (2) two Arduino
sketches that run on a physical ESP32. Neither of those is a thing a cloud
host runs continuously. So "hosting" this project means three different,
honest things:

1. A small **web dashboard** for looking at IMU recordings and dataset stats
   -- this genuinely can live on a free host, and does.
2. A **CI workflow** that proves the ML pipeline's tests pass and both
   firmware sketches compile clean, on every push -- this also genuinely runs
   in the cloud, for free.
3. **Flashing the firmware to a real ESP32** -- there is no cloud equivalent
   of this. This section is a guide to doing it on your own bench, and it
   says plainly that nobody has done this step for you yet.

Read `build_from_scratch/` first if you haven't -- `data/generate_data.py` ->
`model/train.py` -> the two sketches in `firmware/` -- so the rest of this
guide has something to point at.

---

## Part 1 -- The IMU log viewer webapp

### What it does

`hosting/webapp/app.py` is a Streamlit app with three tabs:

- **Plot a Recording** -- pick any `recording_id` out of a loaded CSV and see
  its accelerometer trace (ax/ay/az) and gyroscope trace (gx/gy/gz) over
  `t_ms`, plus the raw rows in a table.
- **Dataset Summary** -- recordings per class (bar chart), total
  recordings/samples/classes, and a samples-per-recording distribution (this
  matters because real hand-timed recordings from `data_collector.ino` won't
  always land on exactly 50 samples the way the synthetic generator does).
- **Live Prediction** -- loads the real
  `build_from_scratch/model/gesture_model_int8.tflite` through
  `tf.lite.Interpreter` and classifies a selected recording, **only if
  tensorflow is importable in the environment the app is running in.**

It accepts a CSV from three places: the built-in synthetic dataset
(`build_from_scratch/data/imu_gesture_log.csv`, loaded by default so the app
is useful with zero clicks), a file upload, or pasted text. All three go
through the same schema check for the 9 required columns:
`recording_id,label,t_ms,ax,ay,az,gx,gy,gz` -- exactly what
`data_collector.ino` prints over serial and what `generate_data.py` writes.

### Why Live Prediction isn't a hard dependency

`hosting/webapp/requirements.txt` does **not** include tensorflow or
tflite-runtime. This was a deliberate call, not an oversight:

- Full `tensorflow` is a several-hundred-MB install. The model it would be
  loading is 15,832 bytes. Asking a free-tier host (Streamlit Community
  Cloud's containers, Hugging Face Spaces' free CPU tier) to install a
  framework that large, just to run a model that small, means slower cold
  starts and a real risk of tripping the free tier's roughly 1GB memory
  ceiling during import -- for no accuracy or feature benefit.
- The obvious lighter alternative, `tflite-runtime`, isn't a safe substitute
  either: Google deprecated that PyPI package in 2023 in favor of
  `ai-edge-litert`, and it is no longer reliably installable across Python
  versions/platforms. (Worth noting while we're being honest about tooling:
  running the test suite on this machine with TensorFlow 2.21.0 installed
  still works today, but `tf.lite.Interpreter` itself now prints a
  deprecation warning pointing at `ai_edge_litert` as its eventual
  replacement -- this is a real warning we saw when running the tests below,
  not a hypothetical.)

So the app's code tries `import tensorflow` inside the Live Prediction tab,
catches `ImportError`, and shows a plain explanation instead of crashing when
it's missing. Visualization and dataset stats -- the two things the task
actually needs a hosted dashboard for -- depend on nothing but
pandas/matplotlib and work regardless.

If you want the Live Prediction tab: `pip install tensorflow` (you already
need it for `build_from_scratch/model/train.py`) and run the app locally.
It will pick it up automatically -- no code change needed.

### Run it locally

```bash
pip install -r hosting/webapp/requirements.txt
streamlit run hosting/webapp/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The
built-in synthetic dataset loads immediately; try uploading your own
`data_collector.ino` output once you've recorded some real gestures.

### Deploy it for free

**Streamlit Community Cloud:**

1. Push this project to GitHub (see Part 2's Step 0 below).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app**, pick the repo and branch, and set the main file path to:
   ```
   hosting/webapp/app.py
   ```
4. Streamlit Cloud auto-detects `hosting/webapp/requirements.txt` because it
   sits next to the entry file.
5. Click **Deploy**.

Because `build_from_scratch/data/imu_gesture_log.csv` and
`build_from_scratch/model/gesture_model_int8.tflite` are committed to the
repo, the dashboard works instantly on a cold container -- the built-in
dataset loads with nothing to generate first, and (if you separately choose
to install tensorflow on that deployment, which the default
`requirements.txt` deliberately does not do) the model file is already there
too.

**Hugging Face Spaces** works the same way: create a **Streamlit** Space,
push the repo, set the app file to `hosting/webapp/app.py`. Same behavior,
same caveat about the Live Prediction tab.

### Run the smoke test

```bash
pip install -r hosting/webapp/requirements.txt
pytest hosting/webapp/test_app.py -q
```

This uses `streamlit.testing.v1.AppTest` to run the entire `app.py` headlessly
(no browser, no server) and asserts it renders without raising, that the
default source loads all 800 recordings across 4 classes, that a malformed
CSV gets a clear schema error instead of a crash, and that the Live
Prediction tab produces *something* sensible in whichever environment it
runs in (a real prediction if tensorflow is present, an explanatory message
if not). This was actually run on this machine -- 5 passed, tensorflow
happened to be installed here so the real `tf.lite.Interpreter` path was
exercised, not just the fallback message.

---

## Part 2 -- The CI workflow

`hosting/github_actions/ci.yml` is a ready-made GitHub Actions workflow. Copy
it to `.github/workflows/ci.yml` at your repo root (GitHub only looks in
`.github/workflows` there) and push.

### What it does, step by step

1. Installs `build_from_scratch/requirements.txt` (tensorflow, numpy, pytest).
2. Runs `pytest -q` in `build_from_scratch/` -- 7 tests covering the data
   generator's determinism, the CSV schema/shape, a gravity-magnitude sanity
   check, TFLite flatbuffer validity, model/firmware header-copy consistency,
   and the "int8 accuracy can't exceed float32 accuracy" ordering check.
3. Installs `arduino-cli` via the official `arduino/setup-arduino-cli`
   GitHub Action (the equivalent shell command, if you're reproducing this
   outside GitHub Actions, is
   `curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh`
   -- this is literally how arduino-cli got installed on this development
   machine too, portably and without admin rights).
4. Adds the `esp32:esp32` board index and installs that core.
5. Installs the firmware's two libraries, `ArduTFLite` and
   `Adafruit MPU6050` (arduino-cli resolves each one's own dependencies --
   `Chirale_TensorFlowLite` for ArduTFLite, `Adafruit BusIO` +
   `Adafruit Unified Sensor` for the MPU6050 library -- automatically).
6. Runs `arduino-cli compile --fqbn esp32:esp32:esp32` on
   `build_from_scratch/firmware/tinyml_gesture`.
7. Runs the same compile command on
   `build_from_scratch/firmware/data_collector`.

Any of those steps failing turns the whole job red. A broken sketch, a
missing header, an accuracy regression, or a bad CSV schema change all get
caught before they ever reach a real board.

### What this proves, and what it explicitly does not

**Proves:** the Python pipeline's tests pass on a clean checkout, and both
`.ino` sketches compile cleanly against the real `ArduTFLite` and
`Adafruit_MPU6050` library APIs for the exact `esp32:esp32:esp32` board
target this project targets. That's a real, meaningful guarantee -- a
firmware change that doesn't compile, or a Python change that breaks a test,
cannot merge quietly.

**Does not prove, and cannot prove:** anything about a real board. GitHub's
hosted runners are plain Linux VMs with nothing plugged into them. This
workflow cannot flash a board, cannot confirm the I2C wiring is correct,
cannot confirm the MPU6050 returns sane values, and cannot confirm the
on-device sliding-window classifier actually recognizes a real "wave" versus
a real "punch." A green run means **"this will build,"** not **"this
works on a bench."** See Part 3 below and
`build_from_scratch/wiring/WIRING_GUIDE.md` for the steps that only a real
board can complete -- this is stated here and in the workflow file's own
header comment on purpose, not buried.

### Secrets

None needed. No API keys, no cloud credentials -- everything this workflow
touches is either pip packages or arduino-cli's own board/library indexes.

---

## Part 3 -- "Hosting" the firmware: flashing it to a real ESP32

There is no cloud host for embedded firmware -- the real-world equivalent of
"deploying" this project is flashing it onto a physical board over USB. This
section is that guide. **Nobody has run these steps on this machine** -- there
is no ESP32 or MPU6050 attached to it. Everything below is written against
the real `arduino-cli`/Arduino IDE APIs and the sketches that compiled clean
in CI, but the actual flash-and-run has to happen on your bench, not here.

### What you need

- An ESP32 dev board (e.g. an ESP32 DevKitC) and an MPU6050 breakout, wired
  per `build_from_scratch/wiring/WIRING_GUIDE.md` (VCC->3V3, GND->GND,
  SCL->GPIO22, SDA->GPIO21).
- A data-capable USB cable (not a charge-only one -- this trips up more
  people than it should).
- Either `arduino-cli` (same tool CI uses) or the Arduino IDE.

### Option A -- Arduino IDE (simplest if you already have it installed)

1. Install the ESP32 board package: **File > Preferences**, add
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   to "Additional Board Manager URLs," then **Tools > Board > Boards
   Manager**, search `esp32`, install the Espressif package.
2. Install the libraries: **Tools > Manage Libraries**, search and install
   `ArduTFLite` and `Adafruit MPU6050` (accept the dependency prompt for
   Adafruit BusIO / Adafruit Unified Sensor).
3. Open `build_from_scratch/firmware/data_collector/data_collector.ino`
   first -- not the inference sketch. It's the simpler of the two and is
   the right first flash to confirm the wiring works at all.
4. Plug in the board. **Tools > Board** -> pick your ESP32 board variant
   (e.g. "ESP32 Dev Module"). **Tools > Port** -> pick the new serial port
   that appeared when you plugged in (on Windows this looks like `COM3`,
   `COM7`, etc.; on macOS/Linux, `/dev/cu.usbserial-XXXX` or
   `/dev/ttyUSB0`). If you're not sure which port is the board, unplug it,
   check the port list, plug it back in, and see which one appears.
5. Click **Upload**. Once it finishes, open **Tools > Serial Monitor** at
   115200 baud. You should see the CSV header line and a "ready" message --
   if instead you see `MPU6050 not found`, go through the wiring guide's
   bring-up checklist (SDA/SCL swap and loose breadboard contacts are the two
   most common causes).
6. Once the data collector sketch works and you trust the wiring, repeat the
   same board/port/upload steps for
   `build_from_scratch/firmware/tinyml_gesture/tinyml_gesture.ino`.

### Option B -- arduino-cli (same tool CI uses, if you'd rather stay on the
command line)

```bash
# one-time setup -- board core and libraries (same as the CI workflow's steps)
arduino-cli config init --overwrite
arduino-cli config add board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install ArduTFLite
arduino-cli lib install "Adafruit MPU6050"

# find which serial port the board is on
arduino-cli board list
```

`board list` prints something like:

```
Port         Protocol Type              Board Name        FQBN            Core
/dev/ttyUSB0 serial   Serial Port (USB) Espressif ESP32... esp32:esp32:esp32 esp32:esp32
```

(On Windows this will show a `COMx` port instead of `/dev/ttyUSB0`.) If
nothing shows up, the board isn't enumerating -- check the USB cable is a
data cable, and that you've installed the board's USB-serial driver if it
needs one (CP2102 and CH340 are the two common ones on cheap ESP32 boards).

```bash
# flash the data collector first -- it's the simplest possible wiring check
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 \
  build_from_scratch/firmware/data_collector

# watch it stream (Ctrl+C to exit)
arduino-cli monitor -p /dev/ttyUSB0 -c baudrate=115200

# once wiring is confirmed, flash the real inference sketch
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 \
  build_from_scratch/firmware/tinyml_gesture
```

Replace `/dev/ttyUSB0` with whatever `arduino-cli board list` actually
printed for your machine.

### What to expect once it's flashed (and what you're checking for)

- `data_collector.ino`: hold the BOOT button (GPIO 0) and move the board --
  you should see `ax,ay,az,gx,gy,gz` values change sensibly (az near 1.0 when
  flat and still, since that's 1g of gravity; others near 0 at rest).
- `tinyml_gesture.ino`: after `"model ready. move the board to see
  predictions."` prints, perform one of the four gestures (idle/wave/circle/
  punch) and watch for a predicted label + confidence to print. Since the
  model was trained on **synthetic** data, don't expect it to reliably
  recognize your real arm motion out of the box -- see the note below.

### The honest state of this project right now

This project is deliberately built and documented around a three-way split
of what's actually verified:

1. **Verified for real, on a machine, by running the actual tools:** the
   entire ML pipeline (data generation, training, int8 quantization, the
   0.85 quantized test accuracy vs. 1.00 float32) and both firmware
   sketches' **compilation** against the real `esp32:esp32:esp32` target
   (621,992 bytes / 47% flash for `tinyml_gesture.ino`, 318,460 bytes / 24%
   flash for `data_collector.ino` -- real numbers from a real
   `arduino-cli compile`, not estimates).
2. **Written against the real library APIs, never run on a physical board:**
   the on-device inference loop's actual behavior with a real MPU6050 wired
   up, and the data-collector's real button-triggered recording flow. These
   compile clean but have never been flashed or exercised with a real
   sensor from this machine.
3. **Only you can do this, with a real board in your hands:** the wiring and
   continuity check, actually performing gestures and watching what the
   model predicts, and the power-measurement-with-a-multimeter step in
   `build_from_scratch/wiring/WIRING_GUIDE.md`.

**This guide, and the CI workflow above, do not change that split.** CI
proves the code compiles; this section explains how to flash it; neither one
is a substitute for actually doing it on a bench. If you flash this and try
it, the single most useful next step is recording ~20-50 real gestures per
class with `data_collector.ino` and retraining `model/train.py` on that real
data (see that script's `CSV_PATH` -- point it at your own log, mixed with or
replacing the synthetic one). Real accelerometer data from an actual human
arm will not hit the synthetic dataset's 100%/85% numbers, and no particular
number is claimed here for real data, because that number doesn't exist
until you generate it.

---

## What "done and live" looks like

- A public **dashboard URL** where anyone can drop in a `data_collector.ino`
  CSV and see the traces, without installing anything.
- A green **Actions** run showing 7 pytest tests + 2 successful firmware
  compiles.
- Your own bench, with a real prediction printing over serial when you wave
  your arm -- the one part of "done" that has to happen off this machine.
