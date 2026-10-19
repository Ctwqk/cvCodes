#!/usr/bin/env bash
# Ubuntu container / CI setup, run as container root (see docker.sh).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends build-essential cmake libopencv-dev \
  python3-opencv python3-pytest python3-matplotlib python3-onnx python3-venv ca-certificates
python3 -m venv --system-site-packages /opt/pu-cvcodes-python
/opt/pu-cvcodes-python/bin/pip install --no-cache-dir \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
