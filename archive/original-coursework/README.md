# Original coursework source archive

This archive preserves every distinct original source file removed from its
previous location during the portfolio upgrade, including implementations that
now have maintained replacements. The source is copied byte for byte from
default-branch commit `ca45a2680d86d42d45fb84c630e268df49c9c1b1`.

There are 44 archived files covering C++ sources and headers, Python scripts,
text source backups, and handwritten CMake configurations. Four additional
original paths contain exact duplicates of files retained elsewhere in this
archive or the active tree. The original [Project 1 submission notes](../../project1/readme.txt)
and video URL are restored at their original path.

The [manifest](manifest.json) records all 49 original paths, their Git SHA-1 blob
IDs, the location of the preserved copy, whether that copy is shared with an
exact duplicate, and the maintained replacement paths where known. A matching
blob ID means the entire file is identical, including whitespace. For example,
run `git hash-object archive/original-coursework/project5/script.py` from the
repository root and compare it with that entry's `git_blob_sha1`.

These files are historical reference material, excluded from the active CMake
targets and test suites. They retain original unfinished code, old dependency
assumptions, paths, and bugs; the archived CMake files do not constitute a
self-contained supported build. Use the current project directories and their
READMEs to build and run the maintained programs.

| Original work | Maintained implementation |
| --- | --- |
| Project 1 display experiments and nested build configuration | `project1/imgDisplay.cpp`, `project1/task2/filter.cpp`, and `project1/CMakeLists.txt` |
| Shared CSV, distance, and image-list helpers | `common/csv_util.*`, `common/distance.*`, and `common/app_io.*` |
| Project 2 feature extraction | `project2/imgfeatures.*` and `project2/retrieval.*` |
| Project 3 segmentation, region features, and network examples | `project3/pipeline.*`, `project3/region.*`, `project3/regionfeatures.*`, and `project3/dnnEmbedding.*` |
| Project 4 calibration and overlay experiments | `project4/calibrate.cpp`, `project4/extension.cpp`, and `project4/ar_utils.*` |
| Project 5 digit inference, training, filter visualisation, and transfer learning | `project5/mnist.*`, `project5/mnistLive.cpp`, `project5/mnist_model.py`, `project5/filterviz.*`, and the current Python utilities |
| Final-project game backup and duplicated pattern tools | `projectFinal/SnakeGame/` and `projectFinal/SnakeGame_3d/snakegame.cpp` |

The integral-image experiment in `project4/sobel.cpp` and its caller
`project4/RobustFeature_bak` have no maintained equivalent. The active Project 4
corner detector uses OpenCV Harris detection; only the other Sobel helpers have
counterparts in Project 1.

Generated Makefiles, compiled executables, editor swap files, generated feature
databases and visualisations, and replaceable dataset downloads are outside the
source archive. Identical source copies are represented by one preserved file
and explicit mappings in the manifest.
