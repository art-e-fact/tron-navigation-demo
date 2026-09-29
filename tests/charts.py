"""The run's charts: two-column CSVs, which the Artefacts dashboard shows as charts.
Each returns its data too, for the metrics."""

import numpy as np

from artefacts_toolkit.chart import make_chart


def path_chart(bag, out_dir):
    """path.csv: where the robot walked, x vs y in the map frame. Returns the (N, 2) points."""
    make_chart(bag, "/amcl_pose.pose.pose.position.x", "/amcl_pose.pose.pose.position.y", field_unit="m",
               output_dir=out_dir, chart_name="path", output_format="csv")
    return read_csv(out_dir / "path.csv")


def speed_chart(bag, out_dir):
    """speed.csv: how fast the robot walked forward, over time. Returns the speeds."""
    make_chart(bag, "time", "/odom.twist.twist.linear.x", field_unit="m/s",
               output_dir=out_dir, chart_name="speed", output_format="csv")
    speed = read_csv(out_dir / "speed.csv")
    speed[:, 0] -= speed[0, 0]  # make_chart keeps /odom's header stamps absolute: start them at 0
    write_csv(out_dir / "speed.csv", speed, "Time (s)", "/odom.twist.twist.linear.x (m/s)")
    return speed[:, 1]


def route_deviation_chart(path, route, out_dir):
    """route_deviation.csv: how far the robot strayed from the route line (start -> waypoints),
    along the way it walked. Returns (walked, deviation)."""
    line = np.array([(p.x, p.y) for p in [route.initial_pose, *route.waypoints]])
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    deviation = distance_to_line(path, line)
    write_csv(out_dir / "route_deviation.csv", np.column_stack([walked, deviation]),
              "distance walked (m)", "distance from route (m)")
    return walked, deviation


def distance_to_line(points, line):
    """Each point's distance to the nearest segment of the polyline."""
    start, seg = line[:-1], np.diff(line, axis=0)
    t = np.clip(((points[:, None] - start) * seg).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(points[:, None] - (start + t[..., None] * seg), axis=-1).min(axis=1)


def read_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def write_csv(path, data, x_title, y_title):
    np.savetxt(path, data, delimiter=",", header=f"{x_title},{y_title}", comments="", fmt="%.3f")
