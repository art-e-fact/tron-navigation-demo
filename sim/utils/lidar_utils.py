"""A Livox Mid-360, cast with Newton's SensorTiledCamera from custom rays in the lidar frame."""

import numpy as np
import warp as wp
from newton.sensors import SensorTiledCamera
from scipy.spatial.transform import Rotation

LIDAR_MOUNT = np.array([0.0, 0.0, 0.07])  # on the base_Link top plate (mesh top z=0.0165)
# Mid-360: 360 deg x -7..52 deg, 10 Hz, 0.1..40 m. A regular grid, with a 0 deg ring for /scan.
LIDAR_MIN_RANGE, LIDAR_MAX_RANGE = 0.1, 40.0
AZIMUTHS = np.radians(np.arange(-180.0, 180.0, 0.5))
ELEVATIONS = np.radians([-7.0] + list(range(-6, 53, 2)))


class Mid360:
    def __init__(self, model):
        self.sensor = SensorTiledCamera(model, load_textures=False)
        elevation, azimuth = np.meshgrid(ELEVATIONS, AZIMUTHS, indexing="ij")
        self.ray_dirs = np.stack([np.cos(elevation) * np.cos(azimuth),
                                  np.cos(elevation) * np.sin(azimuth),
                                  np.sin(elevation)], -1)
        rays = np.stack([np.zeros_like(self.ray_dirs), self.ray_dirs], -2)[None]  # (1, H, W, origin/dir, xyz)
        self.rays = wp.array(rays.astype(np.float32), dtype=wp.vec3f)
        self.depth = self.sensor.utils.create_depth_image_output(len(AZIMUTHS), len(ELEVATIONS))
        self.hit_shape = self.sensor.utils.create_shape_index_image_output(len(AZIMUTHS), len(ELEVATIONS))
        # is_robot_shape[shape index]; the extra False at the end catches misses (index clamped to it)
        self.is_robot_shape = np.append(model.shape_body.numpy() >= 0, False)

    def scan(self, state, base_pos, base_quat):
        """Points (N, 3) in the lidar frame, without the robot's own shapes."""
        lidar_pos = base_pos + Rotation.from_quat(base_quat).apply(LIDAR_MOUNT)
        pose = wp.transformf(wp.vec3f(*lidar_pos), wp.quatf(*base_quat))
        self.sensor.update(state, wp.array([[pose]], dtype=wp.transformf), self.rays,
                           depth_image=self.depth, shape_index_image=self.hit_shape)
        ranges = self.depth.numpy()[0, 0]
        hit = np.minimum(self.hit_shape.numpy()[0, 0].astype(np.int64), len(self.is_robot_shape) - 1)
        keep = (ranges >= LIDAR_MIN_RANGE) & (ranges <= LIDAR_MAX_RANGE) & ~self.is_robot_shape[hit]
        return (self.ray_dirs[keep] * ranges[keep, None]).astype(np.float32)
