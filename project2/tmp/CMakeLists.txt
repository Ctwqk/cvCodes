
cmake_minimum_required(VERSION  2.8)
project(prject2)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(sy sy.cpp)
target_link_Libraries(sy ${OpenCV_LIBS})

