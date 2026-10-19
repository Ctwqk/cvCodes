#!/usr/bin/env bash
set -euo pipefail
build=${1:-build}
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j4
ctest --test-dir "$build" --output-on-failure
for executable in "$build"/bin/*; do "$executable" --help > /dev/null; done
# Each coursework directory must also configure and build without the top level.
for project in project1 project2 project3 project4 project5 hw4 projectFinal projectFinal/SnakeGame projectFinal/SnakeGame_3d; do
  standalone="$build/standalone/${project//\//-}"
  cmake -S "$project" -B "$standalone" -DCVC_BUILD_TESTS=OFF
  cmake --build "$standalone" -j4
done
