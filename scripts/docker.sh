#!/usr/bin/env bash
# Reproducible Ubuntu/OpenCV environment; no host compiler or OpenCV installation needed.
# Run heavy work through your host's lock/resource policy when sharing a production host.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# == 0 )); then set -- bash scripts/verify.sh; fi
exec docker run --rm --name "pu-cvcodes-${$}" --memory 4g --memory-swap 4g --cpus 4 \
  -v "$PWD":/src -w /src -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e OPENCV_FOR_THREADS_NUM=2 \
  ubuntu:24.04 bash -c '
    set -e
    bash scripts/install-deps.sh
    export PATH="/opt/pu-cvcodes-python/bin:$PATH"
    export TMPDIR="$(mktemp -d /tmp/pu-cvcodes-run-XXXXXX)"
    chown "$LOCAL_UID:$LOCAL_GID" "$TMPDIR"
    export MPLCONFIGDIR="$TMPDIR/matplotlib"
    exec setpriv --reuid="$LOCAL_UID" --regid="$LOCAL_GID" --clear-groups "$@"
  ' bash "$@"
