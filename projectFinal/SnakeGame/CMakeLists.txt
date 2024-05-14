cmake_minimum_required(VERSION 2.8)
project( projectFinal )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(snakegame snakegame.cpp CalPos.cpp)
target_link_Libraries(snakegame ${OpenCV_LIBS})

add_executable( main main.cpp saveOrReadPattern.cpp)
target_link_Libraries(main ${OpenCV_LIBS})

add_executable( match_tmp match_tmp.cpp patternMatch.cpp saveOrReadPattern.cpp)
target_link_Libraries(match_tmp ${OpenCV_LIBS})
