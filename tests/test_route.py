import os
import signal
import subprocess
from pathlib import Path

import pytest

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
def robot():
    """Sim + LimX controller + Nav2, up for one test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = RESULTS_DIR / "launch.log"
    args = [f"headless:={os.environ.get('HEADLESS', 'true')}",
            f"rviz:={os.environ.get('RVIZ', 'false')}"] + ([f"route:={ROUTE}"] if ROUTE else [])
    with log.open("w") as out:
        proc = subprocess.Popen(["ros2", "launch", str(ROOT / "launch" / "tron1_nav2.launch.py"), *args],
                                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    yield proc, log
    os.killpg(proc.pid, signal.SIGINT)  # ros2 launch shuts its processes down in order
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def test_route_is_completed(robot):
    proc, log = robot
    result = follow_route(load_route(ROUTE), timeout_s=ROUTE_TIMEOUT_S, startup_timeout_s=STARTUP_TIMEOUT_S,
                          results_dir=RESULTS_DIR)
    tail = "\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])

    assert proc.poll() is None, "the launch died:\n" + tail
    assert result.succeeded, f"{result.summary()}, missed {result.missed_waypoints}\n{tail}"
    assert result.final_pos_error_m is not None and result.final_pos_error_m < MAX_FINAL_ERROR_M
