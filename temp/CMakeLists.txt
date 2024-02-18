cmake_minimum_required(VERSION 2.8)
project( fourier )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( four four.cpp)
target_link_Libraries(four ${OpenCV_LIBS})

add_executable( myFour myFour.cpp)
target_link_Libraries(myFour ${OpenCV_LIBS})

add_executable(fourier fourier.cpp)
target_link_Libraries(fourier ${OpenCV_LIBS})
