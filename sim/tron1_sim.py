"""LimX Tron1 (point-foot biped) in Newton, played through LimX's own SDK.

    python sim/tron1_sim.py --world worlds/x.usda --x 0 --y 0 --yaw 0 [--headless] [--video-dir results]

The upstream controller (robot_hw pointfoot_node) talks limxsdk to "the robot" at
127.0.0.1; this script is that robot, like LimX's tron1-mujoco-sim/simulator.py:
it takes RobotCmd, applies tau = Kp*(q_cmd-q) + Kd*(dq_cmd-dq) + tau_ff and sends
back RobotState + IMU every physics step, paced to wall clock.
For Nav2 it publishes /odom, TF odom->base_link->livox_frame and a Mid-360 cloud.
With --video-dir it also records follow.mp4 and birdseye.mp4, headless or not.
"""

import argparse
import os
import signal
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pyglet
from scipy.spatial.transform import Rotation
import warp as wp
import newton
import newton.viewer
from newton.sensors import SensorIMU

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

from utils.camera_utils import VIDEO_SIZE, VideoRecorder, follow_view
from utils.lidar_utils import LIDAR_MOUNT, Mid360

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
VIDEO_PERIOD_S = 0.1  # a multiple of RENDER_PERIOD_S: frames come from the viewer's latest state


def fill(msg, values, fields="xyz"):  # e.g. fill(pose.position, [1, 2, 3])
    for field, value in zip(fields, values):
        setattr(msg, field, float(value))


class Tron1Sim:
    def __init__(self, world, x, y, yaw, headless, video_dir):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(world)  # map_to_world's default USD: ground + 2 m walls, all colliders
        builder.add_mjcf(str(ROBOT_XML), ignore_names=("floor",))  # the world brings the ground
        self.spawn = np.array([x, y, SPAWN_HEIGHT, *Rotation.from_euler("z", yaw).as_quat()])
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

        self.lidar = Mid360(self.model)

        # The window, or a hidden (EGL) viewer when headless to render the videos with
        self.viewer = None
        if not headless or video_dir:
            pyglet.options["headless"] = headless
            self.viewer = newton.viewer.ViewerGL(*VIDEO_SIZE, headless=headless)
            self.viewer.set_model(self.model)
            self.viewer.set_camera(*follow_view(self.spawn[BASE_POS], yaw))
        self.recorder = VideoRecorder(self.viewer, video_dir, 1 / VIDEO_PERIOD_S, yaw) if video_dir else None

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
            if self.recorder and every(VIDEO_PERIOD_S):
                self.recorder.record(q[BASE_POS], q[BASE_QUAT])
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
        spawn_inverse = Rotation.from_quat(self.spawn[BASE_QUAT]).inv()
        body = Rotation.from_quat(q[BASE_QUAT])
        position = spawn_inverse.apply(q[BASE_POS] - np.append(self.spawn[:2], 0.0))
        tf = self.transform("odom", "base_link", position, (spawn_inverse * body).as_quat())
        self.tf.sendTransform(tf)

        odom = Odometry(header=tf.header, child_frame_id="base_link")
        fill(odom.pose.pose.position, position)
        odom.pose.pose.orientation = tf.transform.rotation
        fill(odom.twist.twist.linear, body.inv().apply(qd[BASE_LIN_VEL]))  # twist is in the body frame
        fill(odom.twist.twist.angular, body.inv().apply(qd[BASE_ANG_VEL]))
        self.odom_pub.publish(odom)

    def publish_cloud(self, q):
        points = self.lidar.scan(self.state, q[BASE_POS], q[BASE_QUAT])
        self.cloud_pub.publish(create_cloud_xyz32(self.header("livox_frame"), points))

    def render(self):
        self.viewer.begin_frame(time.monotonic())
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def close(self):
        if self.recorder:
            self.recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video-dir", help="record follow.mp4 and birdseye.mp4 here")
    args = parser.parse_args()
    rclpy.init()
    sim = Tron1Sim(args.world, args.x, args.y, args.yaw, args.headless, args.video_dir)
    # Ctrl-C arrives twice (from the terminal or test, then from ros2 launch): end the loop rather
    # than die, so close() can finish the mp4s. After Tron1Sim: limxsdk's init installs a handler that kills us.
    signal.signal(signal.SIGINT, lambda *_: rclpy.try_shutdown())
    try:
        sim.run()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
