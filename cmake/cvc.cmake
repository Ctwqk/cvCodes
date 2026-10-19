# Shared build setup for every project in this repository.
#
# Included by the top-level CMakeLists.txt and by each project's own
# CMakeLists.txt (so a project can be built stand-alone). Safe to include
# more than once.
include_guard(GLOBAL)

get_filename_component(CVC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CVC_BUILD_TESTS)
  option(CVC_BUILD_TESTS "Build the unit tests (CTest)" ON)
endif()

# Every OpenCV module used anywhere in the repository (OpenCV 4.x).
find_package(OpenCV 4.4 REQUIRED COMPONENTS
             core imgproc imgcodecs highgui videoio calib3d objdetect dnn features2d video)

# One output directory for all executables, whichever project they belong to.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR CMAKE_SOURCE_DIR STREQUAL CVC_ROOT)
  if(CVC_BUILD_TESTS)
    enable_testing()
  endif()
endif()

# Warnings for our own targets only (never for third_party).
function(cvc_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endfunction()

# cvc_add_library(<name> SOURCES a.cpp ... [LIBS dep ...])
function(cvc_add_library name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_library(${name} STATIC ${ARG_SOURCES})
  target_include_directories(${name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${name} PUBLIC ${ARG_LIBS} ${OpenCV_LIBS})
  target_include_directories(${name} SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
  cvc_warnings(${name})
endfunction()

# cvc_add_app(<exe> SOURCES a.cpp ... [LIBS dep ...])
# The CMake target is named after the project directory to keep names unique in
# the top-level build; the executable keeps the plain name.
function(cvc_add_app exe)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  get_filename_component(_proj "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  set(_target "${_proj}_${exe}")
  add_executable(${_target} ${ARG_SOURCES})
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${exe})
  target_link_libraries(${_target} PRIVATE ${ARG_LIBS} ${OpenCV_LIBS})
  target_include_directories(${_target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  cvc_warnings(${_target})
endfunction()

# cvc_add_test(<name> SOURCES test_*.cpp [LIBS dep ...])
# Builds a doctest executable and registers it with CTest.
function(cvc_add_test name)
  if(NOT CVC_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${ARG_SOURCES} "${CVC_ROOT}/tests/test_main.cpp")
  target_include_directories(${name} SYSTEM PRIVATE "${CVC_ROOT}/third_party/doctest")
  target_link_libraries(${name} PRIVATE ${ARG_LIBS} ${OpenCV_LIBS})
  target_include_directories(${name} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  cvc_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

if(NOT TARGET cvcommon)
  add_subdirectory("${CVC_ROOT}/common" "${CMAKE_BINARY_DIR}/common")
endif()
