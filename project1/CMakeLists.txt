cmake_minimum_required(VERSION 3.16)
project(project1 LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

# Filters written from scratch + Haar-cascade face detection + the filter table.
cvc_add_library(project1_filters
  SOURCES task2/filter.cpp task2/filterbank.cpp task2/FaceDetect/faceDetect.cpp
  LIBS cvcommon)
target_include_directories(project1_filters PUBLIC task2)

cvc_add_app(imgDisplay SOURCES imgDisplay.cpp LIBS project1_filters)
cvc_add_app(vidDisplay SOURCES task2/vidDisplay.cpp LIBS project1_filters)
cvc_add_app(timeBlur SOURCES task2/timeBlur.cpp LIBS project1_filters)
cvc_add_app(showFaces SOURCES task2/FaceDetect/showFaces.cpp LIBS project1_filters)

cvc_add_test(test_project1 SOURCES tests/test_filter.cpp LIBS project1_filters)
