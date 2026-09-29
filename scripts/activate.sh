# pixi activation: source the upstream LimX workspace once it is built (pixi run build).
[ -f "$PIXI_PROJECT_ROOT/rosbuild/install/setup.sh" ] && . "$PIXI_PROJECT_ROOT/rosbuild/install/setup.sh"
true
