# Original coursework experiments

These files preserve a handwritten experiment from the original coursework that
is separate from the maintained programs. They are excluded from CMake targets
and are kept for historical reference, without fixes or claims that they build.

`project4/sobel.cpp` includes the original `integralImg` implementation alongside
Sobel helpers. `project4/sobel.h` declares those functions, and
`project4/RobustFeature_bak` is the unfinished camera experiment that called
`integralImg`. The active Project 4 corner detector uses OpenCV Harris detection;
it does not implement this integral-image experiment.

The three files are preserved byte for byte from the repository's original
default-branch commit `ca45a2680d86d42d45fb84c630e268df49c9c1b1`.
