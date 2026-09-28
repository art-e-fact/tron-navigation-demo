#!/usr/bin/env bash
# onnxruntime 1.10 (the version upstream tron1-rl-deploy-ros2 documents), into third_party/.
set -e
cd "$(dirname "$0")/.."
[ -d third_party/onnxruntime ] && exit 0
mkdir -p third_party
curl -sSL https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz | tar xz -C third_party
mv third_party/onnxruntime-linux-x64-1.10.0 third_party/onnxruntime
