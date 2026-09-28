# Tron1 + Nav2 route test in Newton

## Setup

Needs Ubuntu 22.04+ and [pixi](https://pixi.sh). Newton runs on an NVIDIA GPU if
there is one, else on the CPU. The upstream LimX repos (`limx.repos`) are cloned into
`limx_ws/src` and built as-is:

```bash
pixi run build
```

## Usage

1. Save rviz map (yaml and pgm) to `maps/`
2. Record a route with

```bash
pixi run artefacts-route record --name my_route --map maps/my_map.yaml
```

* Set the start with **2D Pose Estimate**
* Add waypoints with **Publish Point**,
* press Enter in the terminal to save `routes/my_route.yaml`

3. Run Test with:

```bash
pixi run pytest -s                                  # Uses the last recorded route
HEADLESS=false RVIZ=true pixi run pytest -s         # watch it
```

## Install without pixi

The following ROS packages are required: 

```
ros-humble-desktop
ros-humble-navigation2
ros-humble-nav2-bringup
ros-humble-pointcloud-to-laserscan
ros-humble-controller-manager
ros-humble-controller-interface
ros-humble-hardware-interface
ros-humble-realtime-tools`
```

plus the `[pypi-dependencies]` of `pixi.toml` in a `--system-site-packages` venv.

```
vcs import --skip-existing limx_ws/src < limx.repos && scripts/get_onnxruntime.sh
"colcon --log-base rosbuild/log build --base-paths limx_ws/src --build-base rosbuild/build --install-base rosbuild/install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH=$PWD/third_party/onnxruntime/lib

source rosbuild/install/setup.bash
export ROBOT_TYPE=PF_TRON1A RL_TYPE=isaacgym
```

and run commands without `pixi run` prefix
