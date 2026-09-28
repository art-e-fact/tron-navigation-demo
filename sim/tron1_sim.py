"""LimX Tron1 (point-foot biped) in Newton, played through LimX's own SDK.

    python sim/tron1_sim.py --world worlds/x.usda --x 0 --y 0 --yaw 0 [--headless]

The upstream controller (robot_hw pointfoot_node) talks limxsdk to "the robot" at
127.0.0.1; this script is that robot, like LimX's tron1-mujoco-sim/simulator.py:
it takes RobotCmd, applies tau = Kp*(q_cmd-q) + Kd*(dq_cmd-dq) + tau_ff and sends
back RobotState + IMU every physics step, paced to wall clock.
For Nav2 it publishes /odom, TF odom->base_link->livox_frame and a Mid-360 cloud.
"""

import argparse
import math
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import warp as wp
import newton
import newton.viewer
from newton.sensors import SensorIMU, SensorTiledCamera

import limxsdk.datatypes as datatypes
from limxsdk.robot.Robot import Robot
from limxsdk.robot.RobotType import RobotType

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

warnings.filterwarnings("ignore", message=".*margin.*zeroed")  # SolverMuJoCo, about the MJCF's 1 mm margins

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / f"limx_ws/src/robot-description/pointfoot/{os.environ['ROBOT_TYPE']}/xml/robot.xml"
ROBOT_IP = "127.0.0.1"  # where the controller looks for the robot: us

DT = 0.002  # 500 Hz, the controller's rate
NUM_JOINTS = 6  # abad_L, hip_L, knee_L, abad_R, hip_R, knee_R (MJCF order = limxsdk order)
TAU_MAX = 80.0  # MJCF motor ctrlrange
SPAWN_HEIGHT = 0.82  # MJCF base height, legs straight
HOLD_S = 2.0  # keep the base pinned until the controller has been commanding for this long...

# Newton's free base: joint_q = [position, quaternion xyzw, legs], joint_qd = [linear, angular, legs]
BASE_POS, BASE_QUAT, BASE_POSE, LEGS_Q = slice(0, 3), slice(3, 7), slice(0, 7), slice(7, None)
BASE_LIN_VEL, BASE_ANG_VEL, BASE_VEL, LEGS_QD = slice(0, 3), slice(3, 6), slice(0, 6), slice(6, None)

ODOM_PERIOD_S, LIDAR_PERIOD_S, RENDER_PERIOD_S, LOG_PERIOD_S = 0.02, 0.1, 0.05, 5.0

LIDAR_MOUNT = np.array([0.0, 0.0, 0.07])  # on the base_Link top plate (mesh top z=0.0165)
# Mid-360: 360 deg x -7..52 deg, 10 Hz, 0.1..40 m. A regular grid, with a 0 deg ring for /scan.
LIDAR_MIN_RANGE, LIDAR_MAX_RANGE = 0.1, 40.0
AZIMUTHS = np.radians(np.arange(-180.0, 180.0, 0.5))
ELEVATIONS = np.radians([-7.0] + list(range(-6, 53, 2)))

CAMERA_BEHIND, CAMERA_HEIGHT, CAMERA_PITCH = 1.5, 2.2, -60.0  # viewer start: behind the robot, above the walls


def quat_mul(q1, q2):  # xyzw
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def quat_inverse(quat):  # of a unit xyzw quaternion
    return quat * [-1, -1, -1, 1]


def rotate(quat, vector):  # rotate vector by the xyzw quaternion
    axis, w = np.asarray(quat[:3]), quat[3]
    return vector + 2.0 * np.cross(axis, np.cross(axis, vector) + w * vector)


def yaw_quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def fill(msg, values, fields="xyz"):  # e.g. fill(pose.position, [1, 2, 3])
    for field, value in zip(fields, values):
        setattr(msg, field, float(value))


class Tron1Sim:
    def __init__(self, world, x, y, yaw, headless):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(world)  # map_to_world's default USD: ground + 2 m walls, all colliders
        builder.add_mjcf(str(ROBOT_XML), ignore_names=("floor",))  # the world brings the ground
        self.spawn = np.array([x, y, SPAWN_HEIGHT, *yaw_quat(yaw)])
        builder.joint_q[BASE_POSE] = self.spawn.tolist()
        self.model = builder.finalize()

        self.imu = SensorIMU(self.model, sites="*/imu")  # before model.state(): needs body_qdd
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.joint_f = np.zeros(self.model.joint_dof_count, np.float32)
        self.physics_graph = None  # on CPU, simulate() runs directly
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.physics_graph = capture.graph

        # Mid-360: custom rays from the lidar frame, cast against every visible shape
        self.lidar = SensorTiledCamera(self.model, load_textures=False)
        elevation, azimuth = np.meshgrid(ELEVATIONS, AZIMUTHS, indexing="ij")
        self.ray_dirs = np.stack([np.cos(elevation) * np.cos(azimuth),
                                  np.cos(elevation) * np.sin(azimuth),
                                  np.sin(elevation)], -1)
        rays = np.stack([np.zeros_like(self.ray_dirs), self.ray_dirs], -2)[None]  # (1, H, W, origin/dir, xyz)
        self.rays = wp.array(rays.astype(np.float32), dtype=wp.vec3f)
        self.depth = self.lidar.utils.create_depth_image_output(len(AZIMUTHS), len(ELEVATIONS))
        self.hit_shape = self.lidar.utils.create_shape_index_image_output(len(AZIMUTHS), len(ELEVATIONS))
        # is_robot_shape[shape index]; the extra False at the end catches misses (index clamped to it)
        self.is_robot_shape = np.append(self.model.shape_body.numpy() >= 0, False)

        self.viewer = None
        if not headless:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
            eye = self.spawn[BASE_POS] + [-CAMERA_BEHIND * math.cos(yaw), -CAMERA_BEHIND * math.sin(yaw), CAMERA_HEIGHT]
            self.viewer.set_camera(wp.vec3(*eye), CAMERA_PITCH, math.degrees(yaw))

        self.robot = Robot(RobotType.PointFoot, True)
        if not self.robot.init(ROBOT_IP):
            raise SystemExit("limxsdk init failed")
        self.cmd = datatypes.RobotCmd()
        self.cmd.q, self.cmd.dq, self.cmd.tau, self.cmd.Kp, self.cmd.Kd = ([0.0] * NUM_JOINTS for _ in range(5))
        self.robot.subscribeRobotCmdForSim(self.on_cmd)
        self.robot_state, self.imu_data = datatypes.RobotState(), datatypes.ImuData()
        self.released = False
        self.nav_started = False

        # ROS
        self.node = rclpy.create_node("tron1_sim")
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.cloud_pub = self.node.create_publisher(PointCloud2, "/livox/lidar", 10)
        self.tf = TransformBroadcaster(self.node)
        self.static_tf = StaticTransformBroadcaster(self.node)
        self.static_tf.sendTransform(self.transform("base_link", "livox_frame", LIDAR_MOUNT, [0, 0, 0, 1]))
        self.node.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def simulate(self):  # one physics step
        self.state.clear_forces()
        self.solver.step(self.state, self.next_state, self.control, None, DT)
        self.state.assign(self.next_state)
        self.imu.update(self.state)

    def on_cmd(self, cmd):  # limxsdk thread
        self.cmd = cmd

    def on_cmd_vel(self, msg):
        self.nav_started |= any((msg.linear.x, msg.linear.y, msg.angular.z))

    def header(self, frame_id):
        return Header(stamp=self.node.get_clock().now().to_msg(), frame_id=frame_id)

    def transform(self, parent, child, position, quat):
        msg = TransformStamped(header=self.header(parent), child_frame_id=child)
        fill(msg.transform.translation, position)
        fill(msg.transform.rotation, quat, "xyzw")
        return msg

    def run(self):
        steps, commanded_steps, start = 0, 0, time.monotonic()
        log_wall, log_steps, slept = start, 0, 0.0

        def every(period_s):
            return steps % round(period_s / DT) == 0

        while rclpy.ok() and (self.viewer is None or self.viewer.is_running()):
            q, qd = self.state.joint_q.numpy(), self.state.joint_qd.numpy()

            # Hold the base at the spawn pose until the controller is walking and Nav2 is driving.
            if not self.released:
                commanded_steps = commanded_steps + 1 if max(self.cmd.Kp) > 0 else 0
                self.released = commanded_steps * DT >= HOLD_S and self.nav_started
                q[BASE_POSE], qd[BASE_VEL] = self.spawn, 0.0
                self.state.joint_q.assign(q)
                self.state.joint_qd.assign(qd)

            self.step_robot(q, qd)
            if self.physics_graph is not None:
                wp.capture_launch(self.physics_graph)
            else:
                self.simulate()
            steps += 1
            log_steps += 1

            if every(ODOM_PERIOD_S):
                self.publish_odom(q, qd)
            if every(LIDAR_PERIOD_S):
                self.publish_cloud(q)
            if self.viewer is not None and every(RENDER_PERIOD_S):
                self.render()
            if every(LOG_PERIOD_S):
                now = time.monotonic()
                print(f"[tron1_sim] t={steps * DT:.0f}s real-time factor {log_steps * DT / (now - log_wall):.2f}"
                      f" (busy {1 - slept / (now - log_wall):.0%}) base z={q[BASE_POS][2]:.2f}"
                      f" {'walking' if self.released else 'held'}", flush=True)
                log_wall, log_steps, slept = now, 0, 0.0

            # pace to wall clock (catches up after a slow step, e.g. a lidar scan)
            ahead = start + steps * DT - time.monotonic()
            if ahead > 0:
                time.sleep(ahead)
                slept += ahead

    def step_robot(self, q, qd):
        """limxsdk side, as in LimX's MuJoCo simulator.py."""
        cmd = self.cmd
        tau = (np.multiply(cmd.Kp, np.subtract(cmd.q, q[LEGS_Q]))
               + np.multiply(cmd.Kd, np.subtract(cmd.dq, qd[LEGS_QD]))
               + np.asarray(cmd.tau))
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        self.joint_f[LEGS_QD] = tau
        self.control.joint_f.assign(self.joint_f)

        state = self.robot_state
        state.q, state.dq, state.tau = q[LEGS_Q].tolist(), qd[LEGS_QD].tolist(), tau.tolist()
        state.stamp = time.time_ns()
        self.robot.publishRobotStateForSim(state)

        imu = self.imu_data
        x, y, z, w = q[BASE_QUAT]
        imu.quat = [w, x, y, z]  # limxsdk wants wxyz
        imu.gyro = self.imu.gyroscope.numpy()[0].tolist()
        imu.acc = self.imu.accelerometer.numpy()[0].tolist()
        imu.stamp = time.time_ns()
        self.robot.publishImuDataForSim(imu)

    def publish_odom(self, q, qd):
        # odom = the spawn pose, so odometry starts at identity like a real robot's
        spawn_inverse = quat_inverse(self.spawn[BASE_QUAT])
        spawn_ground = np.append(self.spawn[:2], 0.0)
        position = rotate(spawn_inverse, q[BASE_POS] - spawn_ground)
        orientation = quat_mul(spawn_inverse, q[BASE_QUAT])
        tf = self.transform("odom", "base_link", position, orientation)
        self.tf.sendTransform(tf)

        body_inverse = quat_inverse(q[BASE_QUAT])  # twist is in the body frame
        odom = Odometry(header=tf.header, child_frame_id="base_link")
        fill(odom.pose.pose.position, position)
        fill(odom.pose.pose.orientation, orientation, "xyzw")
        fill(odom.twist.twist.linear, rotate(body_inverse, qd[BASE_LIN_VEL]))
        fill(odom.twist.twist.angular, rotate(body_inverse, qd[BASE_ANG_VEL]))
        self.odom_pub.publish(odom)

    def publish_cloud(self, q):
        lidar_pos = q[BASE_POS] + rotate(q[BASE_QUAT], LIDAR_MOUNT)
        pose = wp.transformf(wp.vec3f(*lidar_pos), wp.quatf(*q[BASE_QUAT]))
        self.lidar.update(self.state, wp.array([[pose]], dtype=wp.transformf), self.rays,
                          depth_image=self.depth, shape_index_image=self.hit_shape)
        ranges = self.depth.numpy()[0, 0]
        hit = np.minimum(self.hit_shape.numpy()[0, 0].astype(np.int64), len(self.is_robot_shape) - 1)
        keep = (ranges >= LIDAR_MIN_RANGE) & (ranges <= LIDAR_MAX_RANGE) & ~self.is_robot_shape[hit]
        points = (self.ray_dirs[keep] * ranges[keep, None]).astype(np.float32)
        self.cloud_pub.publish(create_cloud_xyz32(self.header("livox_frame"), points))

    def render(self):
        self.viewer.begin_frame(time.monotonic())
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    rclpy.init()
    try:
        Tron1Sim(args.world, args.x, args.y, args.yaw, args.headless).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
