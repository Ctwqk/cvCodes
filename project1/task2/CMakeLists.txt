cmake_minimum_required(VERSION 2.8)
project( project1 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( vidDisplay vidDisplay.cpp filter.cpp FaceDetect/faceDetect.cpp)
target_link_libraries( vidDisplay ${OpenCV_LIBS} )

add_executable( timeBlur timeBlur.cpp filter.cpp)
target_link_Libraries(timeBlur ${OpenCV_LIBS})

