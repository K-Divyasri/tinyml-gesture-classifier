"""Streamlit dashboard for Project 12 -- TinyML Gesture Classifier.

This is the companion web tool for a project that otherwise has no "always-on
service" to host -- the real deliverable is a training pipeline plus firmware
that runs on an ESP32. What IS genuinely useful to put online is a viewer for
the IMU CSV logs both the synthetic data generator and the real
data_collector.ino sketch produce: upload/paste a log, see the accelerometer
and gyroscope traces for any single recording, and check dataset-level shape
(recordings per class, samples per recording).

CSV schema expected (exactly what data_collector.ino prints over serial, and
what data/generate_data.py writes):

    recording_id,label,t_ms,ax,ay,az,gx,gy,gz

One row per IMU sample. ax/ay/az are accelerometer readings in g, gx/gy/gz are
gyroscope readings in deg/s -- the same units train.py trains on and the
firmware converts to on-device (see tinyml_gesture.ino's unit-conversion
comment).

Live Prediction tab: loads the real
model/gesture_model_int8.tflite (the actual 15.8KB int8
quantized model, not a stand-in) through tf.lite.Interpreter and runs it
against a selected recording, IF tensorflow is importable in this environment.
It deliberately is NOT a hard dependency of this app (see requirements.txt and
hosting/HOSTING_GUIDE.md for why) -- the tab degrades to a plain explanation
instead of crashing when tensorflow isn't installed.

Run locally:

    pip install -r hosting/webapp/requirements.txt
    streamlit run hosting/webapp/app.py

This file lives in hosting/webapp/ but reads real artifacts from
the repo root (the synthetic dataset in data/ and the trained model in model/), two
directories up and back down -- same idea as project 25's dashboard, just
without importing a Python package (there's no equivalent shared library here,
the "product" is a CSV schema and a .tflite file).
"""
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# hosting/webapp/app.py -> hosting/ -> repo root (data/, model/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "imu_gesture_log.csv"
TFLITE_MODEL = PROJECT_ROOT / "model" / "gesture_model_int8.tflite"

REQUIRED_COLUMNS = ["recording_id", "label", "t_ms", "ax", "ay", "az", "gx", "gy", "gz"]
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
# training order from data/generate_data.py -- the trained
# model's output softmax index order matches this list exactly.
CLASSES = ["idle", "wave", "circle", "punch"]
SAMPLES_PER_WINDOW = 50  # 1s @ 50Hz -- what train.py and the firmware require

# dataviz-skill validated categorical palette (light-surface, 3/4-slot sets).
# Slot order is fixed, not cycled: blue, aqua, yellow, green.
ACCEL_COLORS = {"ax": "#2a78d6", "ay": "#1baf7a", "az": "#eda100"}
GYRO_COLORS = {"gx": "#2a78d6", "gy": "#1baf7a", "gz": "#eda100"}
CLASS_COLORS = {"idle": "#2a78d6", "wave": "#1baf7a", "circle": "#eda100", "punch": "#008300"}
GRID_COLOR = "#e1e0d9"

st.set_page_config(page_title="TinyML Gesture Log Viewer", page_icon="\U0001F4C8", layout="wide")
st.title("TinyML Gesture Classifier -- IMU Log Viewer")
st.caption(
    "Project 12 companion dashboard. Upload or paste a CSV in the "
    "data_collector.ino serial-log schema (recording_id,label,t_ms,ax,ay,az,gx,gy,gz), "
    "plot the accelerometer/gyroscope traces for any recording, and check "
    "dataset-level stats (recordings per class, samples per recording)."
)


def _style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# --------------------------------------------------------------- load data ---
st.sidebar.header("Load data")
source = st.sidebar.radio(
    "Source",
    ["Use the built-in synthetic dataset", "Upload a CSV", "Paste CSV text"],
    index=0,
    help=(
        "The built-in dataset is data/imu_gesture_log.csv -- "
        "the real 800-recording, seed=42 synthetic set generate_data.py produces. "
        "Upload or paste your own to inspect a real data_collector.ino recording."
    ),
)

df = None
if source == "Use the built-in synthetic dataset":
    if SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
        st.sidebar.caption(
            f"Loaded `{SAMPLE_CSV.relative_to(PROJECT_ROOT)}` -- 800 recordings, "
            "200 per class x 4 classes, generated with a fixed seed."
        )
    else:
        st.sidebar.error(
            f"Can't find {SAMPLE_CSV}. Run "
            "`python data/generate_data.py` first, or switch "
            "to Upload/Paste."
        )
elif source == "Upload a CSV":
    uploaded = st.sidebar.file_uploader(
        "CSV file (recording_id,label,t_ms,ax,ay,az,gx,gy,gz)", type="csv"
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001 -- surface any parse error to the user
            st.sidebar.error(f"Couldn't parse that file as CSV: {exc}")
else:
    pasted = st.sidebar.text_area(
        "Paste CSV rows here, header included", height=220,
        placeholder="recording_id,label,t_ms,ax,ay,az,gx,gy,gz\n0,idle,0,0.05767,...",
    )
    if pasted.strip():
        try:
            df = pd.read_csv(io.StringIO(pasted))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Couldn't parse that text as CSV: {exc}")

if df is None:
    st.info(
        "Load a CSV from the sidebar to get started -- the built-in synthetic "
        "dataset works with no upload needed, or bring your own "
        "data_collector.ino log."
    )
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(
        f"This CSV is missing required column(s): {', '.join(missing)}. "
        f"Expected schema: {', '.join(REQUIRED_COLUMNS)}"
    )
    st.stop()

recordings_meta = (
    df[["recording_id", "label"]].drop_duplicates().sort_values("recording_id").reset_index(drop=True)
)
samples_per_recording = df.groupby("recording_id").size()

tab_plot, tab_summary, tab_predict = st.tabs(
    ["Plot a Recording", "Dataset Summary", "Live Prediction"]
)

# ----------------------------------------------------------------- plot tab --
with tab_plot:
    st.subheader("IMU traces for one recording")
    options = [f"{rid} ({label})" for rid, label in recordings_meta.itertuples(index=False)]
    choice = st.selectbox("Recording (id and label)", options)
    rid = int(choice.split(" ", 1)[0])
    rec = df[df["recording_id"] == rid].sort_values("t_ms").reset_index(drop=True)
    label = rec["label"].iloc[0]
    n_samples = len(rec)

    if n_samples != SAMPLES_PER_WINDOW:
        st.warning(
            f"This recording has {n_samples} samples, not the {SAMPLES_PER_WINDOW} "
            "the model and firmware expect (1 second @ 50Hz). That's normal for a "
            "hand-timed real recording from data_collector.ino -- a human releasing "
            "the button rarely lands on exactly 1.000s. It just means this "
            "particular recording isn't eligible for the Live Prediction tab as-is."
        )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for axis, color in ACCEL_COLORS.items():
            ax.plot(rec["t_ms"], rec[axis], color=color, linewidth=2, label=axis)
        ax.set_xlabel("t_ms")
        ax.set_ylabel("acceleration (g)")
        ax.set_title(f"Accelerometer -- recording {rid} ({label})")
        ax.legend()
        _style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for axis, color in GYRO_COLORS.items():
            ax2.plot(rec["t_ms"], rec[axis], color=color, linewidth=2, label=axis)
        ax2.set_xlabel("t_ms")
        ax2.set_ylabel("angular velocity (deg/s)")
        ax2.set_title(f"Gyroscope -- recording {rid} ({label})")
        ax2.legend()
        _style_axes(ax2)
        st.pyplot(fig2)
        plt.close(fig2)

    with st.expander("Raw samples for this recording"):
        st.dataframe(rec, width="stretch", hide_index=True)

# -------------------------------------------------------------- summary tab --
with tab_summary:
    st.subheader("Recordings per class")
    counts = df.groupby("label")["recording_id"].nunique().sort_index()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bar_colors = [CLASS_COLORS.get(c, "#898781") for c in counts.index]
    bars = ax3.bar(counts.index, counts.values, color=bar_colors)
    ax3.bar_label(bars)
    ax3.set_ylabel("recordings")
    _style_axes(ax3)
    st.pyplot(fig3)
    plt.close(fig3)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total recordings", int(df["recording_id"].nunique()))
    col2.metric("Total samples (rows)", int(len(df)))
    col3.metric("Classes present", int(df["label"].nunique()))

    st.subheader("Samples per recording")
    st.caption(
        f"The training pipeline (model/train.py) requires "
        f"exactly {SAMPLES_PER_WINDOW} samples per recording -- 1 second of IMU "
        "data at 50Hz. The synthetic dataset always hits this exactly; real "
        "recordings from data_collector.ino can be a sample or two off because "
        "the recording window is bounded by a human pressing/releasing a button."
    )
    st.write(
        f"min={int(samples_per_recording.min())}, "
        f"max={int(samples_per_recording.max())}, "
        f"mean={samples_per_recording.mean():.1f}"
    )
    dist = (
        samples_per_recording.value_counts()
        .sort_index()
        .rename_axis("samples_in_recording")
        .rename("recording_count")
        .reset_index()
    )
    st.dataframe(dist, width="stretch", hide_index=True)

# -------------------------------------------------------------- predict tab --
with tab_predict:
    st.subheader("Run the real quantized model against a recording")
    st.caption(
        "Loads model/gesture_model_int8.tflite -- the actual "
        "15,832-byte int8 quantized model that model/train.py "
        "produced, not a stand-in -- through tf.lite.Interpreter, and runs it "
        "against the selected recording's 300 features (50 samples x 6 axes), "
        "flattened in the same order train.py uses."
    )

    try:
        import tensorflow as tf  # noqa: PLC0415 -- deliberately deferred, see module docstring
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    if not TF_AVAILABLE:
        st.info(
            "tensorflow isn't installed in this environment, so this tab can't "
            "run right now. That's deliberate, not a bug: hosting/webapp/"
            "requirements.txt does not install tensorflow, because asking a "
            "free-tier host (Streamlit Community Cloud, Hugging Face Spaces) to "
            "install a several-hundred-MB machine learning framework just to run "
            "a 15.8KB model is a bad trade -- slow cold starts and a real risk of "
            "hitting the free tier's ~1GB memory ceiling. The lighter "
            "alternative, tflite-runtime, isn't a safe substitute either: Google "
            "deprecated that PyPI package in 2023 and it's no longer reliably "
            "installable across platforms. See hosting/HOSTING_GUIDE.md for the "
            "full reasoning. To use this tab: `pip install tensorflow` locally "
            "(you already need it for model/train.py) and run "
            "`streamlit run hosting/webapp/app.py` from your own machine."
        )
    elif not TFLITE_MODEL.exists():
        st.error(
            f"Can't find {TFLITE_MODEL}. Run "
            "`python model/train.py` first."
        )
    else:
        eligible = samples_per_recording[samples_per_recording == SAMPLES_PER_WINDOW].index
        eligible_meta = recordings_meta[recordings_meta["recording_id"].isin(eligible)]
        if len(eligible_meta) == 0:
            st.warning(
                f"No recording in this CSV has exactly {SAMPLES_PER_WINDOW} samples, "
                "so none can be fed to the model as-is."
            )
        else:
            options2 = [
                f"{rid} ({label})" for rid, label in eligible_meta.itertuples(index=False)
            ]
            choice2 = st.selectbox("Recording to classify", options2, key="predict_recording")
            rid2 = int(choice2.split(" ", 1)[0])
            rec2 = df[df["recording_id"] == rid2].sort_values("t_ms")
            true_label = rec2["label"].iloc[0]

            @st.cache_resource
            def load_interpreter(model_path: str):
                interp = tf.lite.Interpreter(model_path=model_path)
                interp.allocate_tensors()
                return interp

            interpreter = load_interpreter(str(TFLITE_MODEL))
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            feature_vec = rec2[AXES].to_numpy(dtype=np.float32).flatten().reshape(1, -1)
            interpreter.set_tensor(input_details["index"], feature_vec)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_details["index"])[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASSES[pred_idx]

            st.success(
                f"Predicted: **{pred_label}** ({probs[pred_idx] * 100:.1f}% confidence)"
            )
            if true_label in CLASSES and true_label != pred_label:
                st.caption(
                    f"CSV's own label column says this recording is '{true_label}' -- "
                    "the int8 model's real measured test accuracy is 0.85 "
                    "(see model/metrics.json), so an occasional "
                    "mismatch is expected, not evidence of a bug."
                )

            fig4, ax4 = plt.subplots(figsize=(6, 3))
            bar_colors2 = [CLASS_COLORS[c] for c in CLASSES]
            bars2 = ax4.bar(CLASSES, probs, color=bar_colors2)
            ax4.bar_label(bars2, fmt="%.2f")
            ax4.set_ylabel("softmax probability")
            ax4.set_ylim(0, 1)
            _style_axes(ax4)
            st.pyplot(fig4)
            plt.close(fig4)
