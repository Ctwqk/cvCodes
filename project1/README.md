# Project 1 — display and filters

Handwritten image operators plus OpenCV face detection. `imgDisplay` applies all filters without a display; `vidDisplay` accepts a camera index, video, or image sequence. Both use `task2/filter.cpp` and the shared filter table.

From the repository root, after the [top-level build](../README.md#run-everything):

```bash
build/bin/imgDisplay project1/cathedral.jpeg --all --max-width 600 --out results/filters
build/bin/vidDisplay --source 0
build/bin/timeBlur --help
```

Interactive image keys are listed by `imgDisplay --list`; `+`/`-` adjust brightness, `<`/`>` contrast, `a` resets, `s` saves, and `q` quits. `showFaces` demonstrates the included Haar cascade. Camera commands require a GUI; batch filters do not.

![Actual filters](../docs/gallery/filters.jpg)

`test_project1` checks reference blur/Sobel outputs, signed derivatives, borders, image types, quantisation and filter dispatch.
