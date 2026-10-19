cmake_minimum_required(VERSION 3.16)
project(hw4 LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

cvc_add_library(hw4_kmeans SOURCES kmeans.cpp colorquant.cpp)

cvc_add_app(colors SOURCES colors.cpp LIBS hw4_kmeans cvcommon)

cvc_add_test(test_hw4 SOURCES tests/test_kmeans.cpp LIBS hw4_kmeans cvcommon)
