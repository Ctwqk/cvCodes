cmake_minimum_required(VERSION 2.8)
project( hw1 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(colors colors.cpp kmeans.cpp )
target_link_Libraries(colors  ${OpenCV_LIBS})
