cmake_minimum_required(VERSION 3.16)
project(SnakeGame3D LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/cvc.cmake")
cvc_add_app(snakegame3d SOURCES snakegame.cpp CalPos.cpp LIBS cvcommon)
# Pattern tools are shared with the planar game.
if(NOT TARGET final_patterns)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../SnakeGame" "${CMAKE_BINARY_DIR}/snake2d")
endif()
