cmake_minimum_required(VERSION 3.16)
project(cvCodes
        VERSION 1.0.0
        DESCRIPTION "Computer-vision coursework: filters, CBIR, segmentation, calibration/AR, DNN, K-means"
        LANGUAGES CXX)

# Top-level build: every project is a sub-directory with its own CMakeLists.txt.
# Each of them can also be configured on its own, e.g.
#     cmake -S project2 -B build/project2 && cmake --build build/project2
option(CVC_BUILD_TESTS "Build the unit tests (CTest)" ON)
option(CVC_BUILD_PROJECTFINAL "Build projectFinal (AR snake game, SIFT pattern matching)" ON)

include(cmake/cvc.cmake)

add_subdirectory(project1)
add_subdirectory(project2)
add_subdirectory(project3)
add_subdirectory(project4)
add_subdirectory(project5)
add_subdirectory(hw4)
# (remaining projects are added below as they are migrated)
