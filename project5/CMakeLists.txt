cmake_minimum_required(VERSION 2.8)
project( project5 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CNPY_PATH "/home/taiwei/code/pCode/cnpy")
include_directories(${CNPY_PATH})
link_directories(${CNPY_PATH}/build)

add_executable( filter filter.cpp)
target_link_Libraries(filter ${OpenCV_LIBS} cnpy z)

add_executable( main main.cpp dnnEmbedding.cpp preprocess.cpp)
target_link_Libraries(main ${OpenCV_LIBS})


add_executable( dnn dnnexample.cpp dnnEmbedding.cpp)
target_link_Libraries(dnn ${OpenCV_LIBS})
