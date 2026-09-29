"""Follow and birdseye videos of the robot, rendered with the sim's Newton viewer."""

import math
from pathlib import Path

import cv2
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

CAMERA_BEHIND, CAMERA_HEIGHT, CAMERA_PITCH = 1.5, 2.2, -60.0  # behind the robot, above the walls
BIRDSEYE_HEIGHT = 12.0
VIDEO_SIZE = (1280, 720)
HEADING_SMOOTHING = 0.1  # per frame, so the follow video does not wobble with the gait


def follow_view(position, yaw):
    """viewer.set_camera args: behind and above the robot, looking along yaw."""
    eye = position + [-CAMERA_BEHIND * math.cos(yaw), -CAMERA_BEHIND * math.sin(yaw), CAMERA_HEIGHT]
    return wp.vec3(*eye), CAMERA_PITCH, math.degrees(yaw)


class VideoRecorder:
    """follow.mp4 and birdseye.mp4 in video_dir, a frame of each per record()."""

    def __init__(self, viewer, video_dir, fps, yaw):
        self.viewer = viewer
        self.heading = np.array([math.cos(yaw), math.sin(yaw)])
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        h264 = cv2.VideoWriter_fourcc(*"avc1")  # ~3 ms a 720p frame; VP8 (webm) took ~35 and stalled the sim
        self.videos = {name: cv2.VideoWriter(str(Path(video_dir) / f"{name}.mp4"), h264, fps, VIDEO_SIZE)
                       for name in ("follow", "birdseye")}

    def record(self, position, quat):
        """Render both views off-screen, so the window keeps its own camera."""
        viewer, camera = self.viewer, self.viewer.camera
        own_view = wp.vec3(*camera.pos), camera.pitch, camera.yaw
        yaw = Rotation.from_quat(quat).as_euler("zyx")[0]
        self.heading += HEADING_SMOOTHING * (np.array([math.cos(yaw), math.sin(yaw)]) - self.heading)
        views = {"follow": follow_view(position, math.atan2(self.heading[1], self.heading[0])),
                 "birdseye": (wp.vec3(position[0], position[1], BIRDSEYE_HEIGHT), -90.0, 90.0)}  # down, map +y up
        for name, view in views.items():
            viewer.set_camera(*view)
            # the viewer's own draw call, without presenting it to the window
            viewer.renderer.render(camera, viewer.objects, viewer.lines, viewer.wireframe_shapes, viewer.arrows)
            frame = cv2.resize(viewer.get_frame().numpy(), VIDEO_SIZE)  # the window may have been resized
            self.videos[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        viewer.set_camera(*own_view)

    def close(self):
        for video in self.videos.values():
            video.release()  # finishes the files
