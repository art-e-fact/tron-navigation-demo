import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from artefacts_toolkit.navigation import follow_route, load_route
from charts import path_chart, route_deviation_chart, speed_chart

ROOT = Path(__file__).resolve().parents[1]
ROUTE = os.environ.get("ROUTE") or None  # none: the route recorded last
RESULTS_DIR = Path(os.environ.get("ARTEFACTS_SCENARIO_UPLOAD_DIR", ROOT / "results"))
METRICS = ROOT / "metrics.json"  # artefacts.yaml's `metrics:`, read from where `artefacts run` runs
ROUTE_TIMEOUT_S = 400
STARTUP_TIMEOUT_S = 120  # the sim loading its world, then Nav2 coming up
SHUTDOWN_TIMEOUT_S = 30
MAX_FINAL_ERROR_M = 0.5
LOG_TAIL_LINES = 20
WALKING_MPS = 0.05  # slower than this counts as standing


@pytest.fixture
def robot(artefacts_params):
    """Sim + LimX controller + Nav2, up for one test. Yields the launch, its log and a dict
    for the test's metrics; the charts and metrics.json follow once the launch is down."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS.unlink(missing_ok=True)  # or a run that dies early reports the previous run's
    log = RESULTS_DIR / "launch.log"
    args = [f"headless:={artefacts_params.get('headless', 'true')}",
            f"rviz:={artefacts_params.get('rviz', 'false')}"] + ([f"route:={ROUTE}"] if ROUTE else [])
    with log.open("w") as out:
        proc = subprocess.Popen(["ros2", "launch", str(ROOT / "launch" / "tron1_nav2.launch.py"), *args],
                                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    metrics = {}
    yield proc, log, metrics

    os.killpg(proc.pid, signal.SIGINT)  # ros2 launch shuts its processes down in order
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    save_charts_and_metrics(metrics)


def save_charts_and_metrics(metrics):
    """The charts from the rosbag the launch just closed, then metrics.json: the test's and the charts'."""
    bag = max((ROOT / "rosbags").glob("rosbag2_*"))
    path = path_chart(bag, RESULTS_DIR)
    speed = speed_chart(bag, RESULTS_DIR)
    walked, deviation = route_deviation_chart(path, load_route(ROUTE), RESULTS_DIR)
    metrics.update(distance_walked_m=walked[-1], mean_walking_speed_mps=speed[speed > WALKING_MPS].mean(),
                   max_route_deviation_m=deviation.max(), mean_route_deviation_m=deviation.mean())
    METRICS.write_text(json.dumps({k: None if v is None else round(float(v), 3) for k, v in metrics.items()},
                                  indent=2))


def test_route_is_completed(robot):
    proc, log, metrics = robot
    result = follow_route(ROUTE, timeout_s=ROUTE_TIMEOUT_S, startup_timeout_s=STARTUP_TIMEOUT_S,
                          results_dir=RESULTS_DIR)
    metrics.update(route_duration_s=result.duration_s, final_pos_error_m=result.final_pos_error_m,
                   final_yaw_error_rad=result.final_yaw_error_rad)
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])

    assert proc.poll() is None, "the launch died:\n" + tail
    assert result.succeeded, f"{result.summary()}, missed {result.missed_waypoints}\n{tail}"
    assert result.final_pos_error_m is not None and result.final_pos_error_m < MAX_FINAL_ERROR_M
