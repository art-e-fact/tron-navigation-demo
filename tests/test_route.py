import os
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

from artefacts_toolkit.chart import make_chart
from artefacts_toolkit_navigation import follow_route, load_route

ROOT = Path(__file__).resolve().parents[1]
ROUTE = os.environ.get("ROUTE") or None
RESULTS_DIR = Path(os.environ.get("ARTEFACTS_SCENARIO_UPLOAD_DIR", ROOT / "results"))
ROUTE_TIMEOUT_S = 400
STARTUP_TIMEOUT_S = 120  # the sim loading its world, then Nav2 coming up
SHUTDOWN_TIMEOUT_S = 30
MAX_FINAL_ERROR_M = 0.5
LOG_TAIL_LINES = 20


@pytest.fixture
def robot(artefacts_params):
    """Sim + LimX controller + Nav2, up for one test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = RESULTS_DIR / "launch.log"
    args = [f"headless:={artefacts_params.get('headless', 'true')}",
            f"rviz:={artefacts_params.get('rviz', 'false')}"] + ([f"route:={ROUTE}"] if ROUTE else [])
    with log.open("w") as out:
        proc = subprocess.Popen(["ros2", "launch", str(ROOT / "launch" / "tron1_nav2.launch.py"), *args],
                                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    yield proc, log
    os.killpg(proc.pid, signal.SIGINT)  # ros2 launch shuts its processes down in order
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    bag = max((ROOT / "rosbags").glob("rosbag2_*"))  # the one the launch just closed
    path_chart(bag)
    speed_chart(bag)
    route_deviation_chart()

def path_chart(bag):
    """path.csv: where the robot walked, x vs y in the map frame."""
    make_chart(bag, "/amcl_pose.pose.pose.position.x", "/amcl_pose.pose.pose.position.y", field_unit="m",
               output_dir=RESULTS_DIR, chart_name="path", output_format="csv")


def speed_chart(bag):
    """speed.csv: how fast the robot walked forward, over time."""
    make_chart(bag, "time", "/odom.twist.twist.linear.x", field_unit="m/s",
               output_dir=RESULTS_DIR, chart_name="speed", output_format="csv")
    speed_csv = RESULTS_DIR / "speed.csv"  # make_chart keeps /odom's header stamps absolute: start them at 0
    header, speed = speed_csv.read_text().splitlines()[0], np.loadtxt(speed_csv, delimiter=",", skiprows=1, ndmin=2)
    speed[:, 0] -= speed[0, 0]
    np.savetxt(speed_csv, speed, delimiter=",", header=header, comments="", fmt="%.3f")


def route_deviation_chart():
    """route_deviation.csv: how far the robot strayed from the route line (start -> waypoints),
    along the way it walked. From path.csv, so after path_chart."""
    route = load_route(ROUTE)
    line = np.array([(p.x, p.y) for p in [route.initial_pose, *route.waypoints]])
    path = np.loadtxt(RESULTS_DIR / "path.csv", delimiter=",", skiprows=1, ndmin=2)
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    np.savetxt(RESULTS_DIR / "route_deviation.csv", np.column_stack([walked, distance_to_line(path, line)]),
               delimiter=",", header="distance walked (m),distance from route (m)", comments="", fmt="%.3f")


def distance_to_line(points, line):
    """Each point's distance to the nearest segment of the polyline."""
    start, seg = line[:-1], np.diff(line, axis=0)
    t = np.clip(((points[:, None] - start) * seg).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(points[:, None] - (start + t[..., None] * seg), axis=-1).min(axis=1)


def test_route_is_completed(robot):
    proc, log = robot
    result = follow_route(load_route(ROUTE), timeout_s=ROUTE_TIMEOUT_S, startup_timeout_s=STARTUP_TIMEOUT_S,
                          results_dir=RESULTS_DIR)
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])

    assert proc.poll() is None, "the launch died:\n" + tail
    assert result.succeeded, f"{result.summary()}, missed {result.missed_waypoints}\n{tail}"
    assert result.final_pos_error_m is not None and result.final_pos_error_m < MAX_FINAL_ERROR_M
