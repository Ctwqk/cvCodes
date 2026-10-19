cmake_minimum_required(VERSION 3.16)
project(project2 LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

# Feature extractors + feature registry / database building / ranking.
cvc_add_library(project2_cbir SOURCES imgfeatures.cpp retrieval.cpp LIBS cvcommon)

cvc_add_app(establishCsv SOURCES establishCsv.cpp LIBS project2_cbir)
cvc_add_app(imgMatch SOURCES imgMatch.cpp LIBS project2_cbir)

cvc_add_test(test_project2 SOURCES tests/test_features.cpp tests/test_retrieval.cpp LIBS project2_cbir)
