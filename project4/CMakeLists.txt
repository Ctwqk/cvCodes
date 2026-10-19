cmake_minimum_required(VERSION 3.16)
project(project4 LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

# Calibration, pose estimation, AR drawing, Harris corners, synthetic chessboard views.
cvc_add_library(project4_ar SOURCES ar_utils.cpp LIBS cvcommon)

cvc_add_app(calibrate SOURCES calibrate.cpp LIBS project4_ar)
cvc_add_app(CalPos SOURCES CalPos.cpp LIBS project4_ar)
cvc_add_app(extension SOURCES extension.cpp LIBS project4_ar)
cvc_add_app(RobustFeature SOURCES RobustFeature.cpp LIBS project4_ar)
cvc_add_app(synthBoard SOURCES synthBoard.cpp LIBS project4_ar)

cvc_add_test(test_project4 SOURCES tests/test_ar.cpp LIBS project4_ar)
