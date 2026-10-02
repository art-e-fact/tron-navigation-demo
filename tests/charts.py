"""The run's charts: two-column CSVs, which the Artefacts dashboard shows as charts.
Each returns its data too, for the metrics."""

import numpy as np

from artefacts_toolkit.chart import make_chart

SPEED_FIELD = "/odom.twist.twist.linear.x"
CSV_FORMAT = "%.3f"  # millimetres, milliseconds
MIN_SEGMENT_LENGTH_SQ = 1e-9  # m^2


def path_chart(bag, out_dir):
    """path.csv: where the robot walked, x vs y in the map frame. Returns the (N, 2) points."""
    make_chart(bag, "/amcl_pose.pose.pose.position.x", "/amcl_pose.pose.pose.position.y", field_unit="m",
               output_dir=out_dir, chart_name="path", output_format="csv")
    return read_csv(out_dir / "path.csv")


def speed_chart(bag, out_dir):
    """speed.csv: how fast the robot walked forward, over time. Returns the speeds."""
    make_chart(bag, "time", SPEED_FIELD, field_unit="m/s",
               output_dir=out_dir, chart_name="speed", output_format="csv")
    speed_csv = out_dir / "speed.csv"
    stamps, speeds = read_csv(speed_csv).T
    elapsed = stamps - stamps[0]  # make_chart keeps /odom's header stamps absolute: start them at 0
    write_csv(speed_csv, np.column_stack([elapsed, speeds]), "Time (s)", f"{SPEED_FIELD} (m/s)")
    return speeds


def route_deviation_chart(path, route, out_dir):
    """route_deviation.csv: how far the robot strayed from the route line (start -> waypoints),
    along the way it walked. Returns (walked, deviation)."""
    line = np.array([(pose.x, pose.y) for pose in [route.initial_pose, *route.waypoints]])
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    deviation = distance_to_line(path, line)
    write_csv(out_dir / "route_deviation.csv", np.column_stack([walked, deviation]),
              "distance walked (m)", "distance from route (m)")
    return walked, deviation


def distance_to_line(points, line):
    """Each point's distance to the nearest segment of the polyline."""
    starts, segments = line[:-1], np.diff(line, axis=0)
    from_starts = points[:, None] - starts  # each point from each segment's start
    lengths_sq = np.maximum((segments * segments).sum(-1), MIN_SEGMENT_LENGTH_SQ)
    # how far along each segment its nearest point is, from 0 (its start) to 1 (its end)
    fractions = np.clip((from_starts * segments).sum(-1) / lengths_sq, 0, 1)
    return np.linalg.norm(from_starts - fractions[..., None] * segments, axis=-1).min(axis=1)


def read_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def write_csv(path, data, x_title, y_title):
    np.savetxt(path, data, delimiter=",", header=f"{x_title},{y_title}", comments="", fmt=CSV_FORMAT)
