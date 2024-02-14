cmake_minimum_required(VERSION 2.8)
project( fourier )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( four four.cpp)
target_link_Libraries(four ${OpenCV_LIBS})
