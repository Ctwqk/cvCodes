cmake_minimum_required(VERSION 3.16)
project(project5 LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/cvc.cmake")

# MNIST / ONNX inference with OpenCV DNN, IDX + NPY readers, filter visualisation.
cvc_add_library(project5_dnn SOURCES npy.cpp idx.cpp mnist.cpp dnnEmbedding.cpp filterviz.cpp LIBS cvcommon)

cvc_add_app(mnistEval SOURCES mnistEval.cpp LIBS project5_dnn)
cvc_add_app(mnistLive SOURCES mnistLive.cpp LIBS project5_dnn)
cvc_add_app(filterViz SOURCES filterViz_main.cpp LIBS project5_dnn)
cvc_add_app(dnnForward SOURCES dnnForward.cpp LIBS project5_dnn)

cvc_add_test(test_project5 SOURCES tests/test_npy_idx.cpp tests/test_mnist.cpp LIBS project5_dnn)
