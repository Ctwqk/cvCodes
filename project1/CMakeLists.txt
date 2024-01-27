cmake_minimum_required(VERSION 2.8)
project( project1 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( imgDisplay imgDisplay.cpp task2/filter.cpp task2/FaceDetect/faceDetect.cpp)
target_link_Libraries(imgDisplay ${OpenCV_LIBS})

add_executable( play play.cpp task2/filter.cpp task2/FaceDetect/faceDetect.cpp)
target_link_Libraries(play ${OpenCV_LIBS})
