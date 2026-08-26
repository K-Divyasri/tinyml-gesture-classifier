"""
Deterministic synthetic IMU (MPU6050) gesture data generator.

Real physical data collection (see ../firmware/data_collector/) streams the exact
same CSV schema over serial: one row per IMU sample, columns
recording_id,label,t_ms,ax,ay,az,gx,gy,gz. This script produces data in that same
schema so the training pipeline in ../model/train.py never needs to know whether a
recording came from a real MPU6050 on a real board or from this generator.

Four gesture classes, modeled on how each one actually moves an MPU6050 strapped
to a wrist, not just abstract sine waves:
  - idle:   sensor at rest. accel ~= gravity on one axis + tiny sensor noise. gyro ~= 0.
  - wave:   side-to-side hand wave. ay oscillates, az carries gravity, small gyro-z.
  - circle: hand traces a circle. ax/ay are two sinusoids 90 degrees out of phase.
  - punch:  a fast forward jab and pullback. one sharp ax impulse, not periodic.

Every recording gets randomized phase, amplitude, gravity-axis tilt, and Gaussian
sensor noise so the four classes are separable but not trivially so -- the same
"real data is messy" property real accelerometer logs have.
"""
import csv
import math
import random
from pathlib import Path

SEED = 42
SAMPLE_RATE_HZ = 50
WINDOW_SECONDS = 1.0
SAMPLES_PER_WINDOW = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)  # 50
RECORDINGS_PER_CLASS = 200
GRAVITY_G = 1.0  # accel unit = g

CLASSES = ["idle", "wave", "circle", "punch"]

OUT_DIR = Path(__file__).parent
OUT_FILE = OUT_DIR / "imu_gesture_log.csv"


def _noise(rng, scale):
    return rng.gauss(0.0, scale)


def gen_idle(rng, n):
    tilt = rng.uniform(-0.15, 0.15)
    rows = []
    for i in range(n):
        ax = tilt + _noise(rng, 0.02)
        ay = rng.uniform(-0.05, 0.05) + _noise(rng, 0.02)
        az = GRAVITY_G + _noise(rng, 0.02)
        gx = _noise(rng, 1.5)
        gy = _noise(rng, 1.5)
        gz = _noise(rng, 1.5)
        rows.append((ax, ay, az, gx, gy, gz))
    return rows


def gen_wave(rng, n):
    freq = rng.uniform(1.5, 2.5)  # Hz, one wave cycle ~ 0.4-0.7s
    amp = rng.uniform(0.6, 1.0)
    phase = rng.uniform(0, 2 * math.pi)
    tilt = rng.uniform(-0.1, 0.1)
    rows = []
    for i in range(n):
        t = i / SAMPLE_RATE_HZ
        ay = amp * math.sin(2 * math.pi * freq * t + phase) + _noise(rng, 0.05)
        ax = tilt + _noise(rng, 0.05)
        az = GRAVITY_G * 0.9 + _noise(rng, 0.05)
        gz = amp * 120 * math.cos(2 * math.pi * freq * t + phase) + _noise(rng, 5)
        gx = _noise(rng, 3)
        gy = _noise(rng, 3)
        rows.append((ax, ay, az, gx, gy, gz))
    return rows


def gen_circle(rng, n):
    freq = rng.uniform(1.0, 1.8)
    amp = rng.uniform(0.5, 0.9)
    phase = rng.uniform(0, 2 * math.pi)
    direction = rng.choice([1, -1])
    rows = []
    for i in range(n):
        t = i / SAMPLE_RATE_HZ
        ax = amp * math.sin(2 * math.pi * freq * t + phase) + _noise(rng, 0.05)
        ay = direction * amp * math.cos(2 * math.pi * freq * t + phase) + _noise(rng, 0.05)
        az = GRAVITY_G * 0.85 + _noise(rng, 0.05)
        gx = -direction * amp * 100 * math.sin(2 * math.pi * freq * t + phase) + _noise(rng, 5)
        gy = amp * 100 * math.cos(2 * math.pi * freq * t + phase) + _noise(rng, 5)
        gz = direction * amp * 60 + _noise(rng, 5)
        rows.append((ax, ay, az, gx, gy, gz))
    return rows


def gen_punch(rng, n):
    impulse_center = rng.uniform(0.25, 0.45) * n / SAMPLE_RATE_HZ
    width = rng.uniform(0.06, 0.1)
    amp = rng.uniform(1.5, 2.5)
    tilt = rng.uniform(-0.1, 0.1)
    rows = []
    for i in range(n):
        t = i / SAMPLE_RATE_HZ
        # a punch is a sharp forward-then-back impulse, not periodic motion
        bump = amp * math.exp(-((t - impulse_center) ** 2) / (2 * width ** 2))
        recoil = -0.4 * amp * math.exp(-((t - impulse_center - 2.2 * width) ** 2) / (2 * width ** 2))
        ax = bump + recoil + tilt + _noise(rng, 0.05)
        ay = tilt * 0.3 + _noise(rng, 0.05)
        az = GRAVITY_G + _noise(rng, 0.05)
        gx = _noise(rng, 4)
        gy = amp * 80 * math.exp(-((t - impulse_center) ** 2) / (2 * width ** 2)) + _noise(rng, 5)
        gz = _noise(rng, 4)
        rows.append((ax, ay, az, gx, gy, gz))
    return rows


GENERATORS = {
    "idle": gen_idle,
    "wave": gen_wave,
    "circle": gen_circle,
    "punch": gen_punch,
}


def main():
    rng = random.Random(SEED)
    header = ["recording_id", "label", "t_ms", "ax", "ay", "az", "gx", "gy", "gz"]
    recording_id = 0
    total_rows = 0

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label in CLASSES:
            gen_fn = GENERATORS[label]
            for _ in range(RECORDINGS_PER_CLASS):
                samples = gen_fn(rng, SAMPLES_PER_WINDOW)
                for i, (ax, ay, az, gx, gy, gz) in enumerate(samples):
                    t_ms = round(i * (1000 / SAMPLE_RATE_HZ))
                    writer.writerow([
                        recording_id, label, t_ms,
                        round(ax, 5), round(ay, 5), round(az, 5),
                        round(gx, 3), round(gy, 3), round(gz, 3),
                    ])
                    total_rows += 1
                recording_id += 1

    print(f"wrote {OUT_FILE}")
    print(f"recordings: {recording_id} ({RECORDINGS_PER_CLASS} per class x {len(CLASSES)} classes)")
    print(f"rows: {total_rows} ({SAMPLES_PER_WINDOW} samples/recording @ {SAMPLE_RATE_HZ}Hz)")


if __name__ == "__main__":
    main()
