# Planar AR snake

Original 2D snake logic and chessboard pose projection. This directory also owns the shared SIFT pattern database and tracking programs. See the [final-project guide](../README.md) for controls and commands.

Standalone build from the repository root:

```bash
cmake -S projectFinal/SnakeGame -B build/snake
cmake --build build/snake -j4
ctest --test-dir build/snake --output-on-failure
```
