cmake_minimum_required(VERSION  2.8)
project(netWork)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(dnn dnnexample.cpp)
target_link_Libraries(dnn ${OpenCV_LIBS})
