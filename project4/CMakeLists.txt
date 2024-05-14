cmake_minimum_required(VERSION 2.8)
project( project4 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(main main.cpp)
target_link_Libraries(main ${OpenCV_LIBS})

add_executable(CalPos CalPos.cpp)
target_link_Libraries(CalPos ${OpenCV_LIBS})

add_executable(RobustFeature RobustFeature.cpp)
target_link_Libraries(RobustFeature ${OpenCV_LIBS})

add_executable(extension extension.cpp)
target_link_Libraries(extension ${OpenCV_LIBS})
