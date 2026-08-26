import csv
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "imu_gesture_log.csv"


def test_generator_is_deterministic(tmp_path):
    # run the generator twice into isolated copies and confirm byte-identical output
    script = DATA_DIR / "generate_data.py"
    for i in (1, 2):
        out = tmp_path / f"run{i}"
        out.mkdir()
        subprocess.run([sys.executable, str(script)], cwd=DATA_DIR, check=True)
        (out / "imu_gesture_log.csv").write_bytes(CSV_PATH.read_bytes())
    assert (tmp_path / "run1" / "imu_gesture_log.csv").read_bytes() == \
        (tmp_path / "run2" / "imu_gesture_log.csv").read_bytes()


def test_csv_schema_and_shape():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == [
        "recording_id", "label", "t_ms", "ax", "ay", "az", "gx", "gy", "gz",
    ]

    recordings = {}
    labels_seen = set()
    for row in rows:
        rid = int(row["recording_id"])
        recordings.setdefault(rid, []).append(row)
        labels_seen.add(row["label"])

    assert labels_seen == {"idle", "wave", "circle", "punch"}
    assert len(recordings) == 800  # 200 per class x 4 classes
    for rid, samples in recordings.items():
        assert len(samples) == 50  # 50Hz x 1 second


def test_gravity_axis_present_when_idle():
    # idle recordings should show ~1g total accel magnitude (gravity), not near-zero
    import math
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        idle_rows = [r for r in reader if r["label"] == "idle"]

    assert idle_rows
    mags = [
        math.sqrt(float(r["ax"]) ** 2 + float(r["ay"]) ** 2 + float(r["az"]) ** 2)
        for r in idle_rows
    ]
    avg_mag = sum(mags) / len(mags)
    assert 0.8 < avg_mag < 1.2
