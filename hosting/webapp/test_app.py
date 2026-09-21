"""Headless smoke test for the IMU gesture log viewer Streamlit app.

Runs the whole app.py script through Streamlit's own test harness -- no
browser, no server, no network -- and asserts it renders without raising.
Same idiom as data_engineer/25_domain_data_mesh/hosting/streamlit_app/test_app.py:
https://docs.streamlit.io/develop/api-reference/app-testing

Run:
    pytest hosting/webapp/test_app.py -q

Whether the Live Prediction tab exercises a real tf.lite.Interpreter
prediction or falls back to the "tensorflow not installed" message depends on
whatever environment pytest itself runs in -- this file checks for whichever
of the two actually happened, so it passes honestly either way instead of
assuming one environment.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent / "app.py"

try:
    import tensorflow  # noqa: F401
    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False


def test_app_runs_without_exception():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception


def test_default_source_loads_builtin_dataset():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    # the built-in synthetic dataset is the default radio choice so the app
    # is useful with zero clicks -- confirm that default and that it produced
    # at least one rendered table (raw samples / summary distribution).
    assert at.sidebar.radio[0].value == "Use the built-in synthetic dataset"
    assert len(at.dataframe) > 0


def test_summary_tab_reports_800_recordings_across_4_classes():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    # data/imu_gesture_log.csv: 800 recordings, 4 classes,
    # 200 per class -- the real, committed numbers from CANON, not assumed.
    assert metrics.get("Total recordings") == "800"
    assert metrics.get("Classes present") == "4"
    assert metrics.get("Total samples (rows)") == "40000"


def test_missing_columns_are_rejected_with_a_clear_error():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Paste CSV text").run()
    at.sidebar.text_area[0].set_value("a,b,c\n1,2,3\n").run()
    assert not at.exception
    assert any("missing required column" in e.value for e in at.error)


def test_live_prediction_tab_matches_whatever_environment_pytest_runs_in():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    if _HAS_TENSORFLOW:
        successes = [s.value for s in at.success]
        assert any("Predicted:" in s for s in successes)
    else:
        infos = [i.value for i in at.info]
        assert any("tensorflow isn't installed" in i for i in infos)
