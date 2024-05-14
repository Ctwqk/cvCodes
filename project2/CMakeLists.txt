cmake_minimum_required(VERSION  2.8)
project(prject2)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(establishCsv readfile.cpp establishCsv.cpp features.cpp csv_util.cpp )
target_link_Libraries(establishCsv ${OpenCV_LIBS})


add_executable(imgMatch imgMatch.cpp csv_util.cpp dist.cpp features.cpp ../project1/task2/filter.cpp)
target_link_Libraries(imgMatch ${OpenCV_LIBS})


