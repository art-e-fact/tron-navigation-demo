import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from artefacts_toolkit.navigation import (
    assert_each_waypoint_reached, assert_final_pose, assert_max_recoveries, assert_min_clearance,
    assert_route_completed, follow_route, load_route)
from charts import path_chart, route_deviation_chart, speed_chart

ROOT = Path(__file__).resolve().parents[1]
ROUTE = os.environ.get("ROUTE") or None  # none: the route recorded last
RESULTS_DIR = Path(os.environ.get("ARTEFACTS_SCENARIO_UPLOAD_DIR", ROOT / "results"))
METRICS = ROOT / "metrics.json"  # artefacts.yaml's `metrics:`, read from where `artefacts run` runs
ROUTE_TIMEOUT_S = 400
STARTUP_TIMEOUT_S = 120  # the sim loading its world, then Nav2 coming up
SHUTDOWN_TIMEOUT_S = 30
MAX_WAYPOINT_ERROR_M = 0.5  # how far from a waypoint still counts as reaching it; Nav2's goal tolerance is 0.3
MIN_CLEARANCE_M = 0.25  # Nav2's robot_radius: a lidar return nearer than this is inside the robot's footprint
MAX_RECOVERIES = 0  # Nav2 spinning or backing up to get unstuck
LOG_TAIL_LINES = 20
WALKING_MPS = 0.05  # slower than this counts as standing
METRIC_DECIMALS = 3


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
    METRICS.write_text(json.dumps({name: round(value, METRIC_DECIMALS) for name, value in metrics.items()},
                                  indent=2))


def test_route_is_completed(robot):
    proc, log, metrics = robot
    result = follow_route(ROUTE, timeout_s=ROUTE_TIMEOUT_S, startup_timeout_s=STARTUP_TIMEOUT_S,
                          results_dir=RESULTS_DIR)
    metrics.update(result.metrics())
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])

    assert proc.poll() is None, "the launch died:\n" + tail
    assert_route_completed(result)
    assert_each_waypoint_reached(result, MAX_WAYPOINT_ERROR_M)
    assert_final_pose(result, MAX_WAYPOINT_ERROR_M)
    assert_min_clearance(result, MIN_CLEARANCE_M)
    assert_max_recoveries(result, MAX_RECOVERIES)
