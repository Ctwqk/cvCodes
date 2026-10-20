cmake_minimum_required(VERSION 3.16)
project(project3 LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

# Pipeline: threshold -> denoise -> regions -> shape features -> nearest neighbour.
cvc_add_library(project3_vision
  SOURCES threshold.cpp denoise.cpp region.cpp regionfeatures.cpp pipeline.cpp
          recognition.cpp dnnEmbedding.cpp dbtools.cpp
  LIBS cvcommon)

cvc_add_app(segment SOURCES segment.cpp LIBS project3_vision)
cvc_add_app(recognize SOURCES recognize.cpp LIBS project3_vision)
cvc_add_app(buildDb SOURCES buildDb.cpp LIBS project3_vision)
cvc_add_app(evalDb SOURCES evalDb.cpp LIBS project3_vision)
cvc_add_app(vidDis SOURCES vidDis.cpp LIBS project3_vision)
cvc_add_app(dnnexample SOURCES dnnexample.cpp LIBS project3_vision)

cvc_add_test(test_project3
  SOURCES tests/test_threshold.cpp tests/test_denoise.cpp tests/test_region.cpp
          tests/test_features_recognition.cpp tests/test_pipeline.cpp tests/test_embedding.cpp
  LIBS project3_vision)
