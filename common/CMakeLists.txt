cmake_minimum_required(VERSION 3.16)
project(cvc_common LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

# Shared helpers: CSV feature files, distance metrics, CLI/IO helpers.
cvc_add_library(cvcommon SOURCES
  csv_util.cpp
  distance.cpp
  app_io.cpp)
target_compile_definitions(cvcommon PRIVATE CVC_REPO_ROOT="${CVC_ROOT}")

if(CVC_BUILD_TESTS)
  cvc_add_test(test_common SOURCES tests/test_csv_util.cpp tests/test_distance.cpp tests/test_app_io.cpp LIBS cvcommon)
endif()
