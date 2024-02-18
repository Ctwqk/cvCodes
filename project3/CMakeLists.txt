cmake_minimum_required(VERSION  2.8)
project(prject3)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})



add_executable(main main.cpp threshold.cpp denoise.cpp region.cpp)
target_link_Libraries(main ${OpenCV_LIBS})


add_executable(debug debug.cpp region.cpp)
target_link_Libraries(debug ${OpenCV_LIBS})
