# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/taiwei/code/cvCode/project3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/taiwei/code/cvCode/project3

# Include any dependencies generated for this target.
include CMakeFiles/threshold.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/threshold.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/threshold.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/threshold.dir/flags.make

CMakeFiles/threshold.dir/threshold.cpp.o: CMakeFiles/threshold.dir/flags.make
CMakeFiles/threshold.dir/threshold.cpp.o: threshold.cpp
CMakeFiles/threshold.dir/threshold.cpp.o: CMakeFiles/threshold.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/threshold.dir/threshold.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/threshold.dir/threshold.cpp.o -MF CMakeFiles/threshold.dir/threshold.cpp.o.d -o CMakeFiles/threshold.dir/threshold.cpp.o -c /home/taiwei/code/cvCode/project3/threshold.cpp

CMakeFiles/threshold.dir/threshold.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/threshold.dir/threshold.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/threshold.cpp > CMakeFiles/threshold.dir/threshold.cpp.i

CMakeFiles/threshold.dir/threshold.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/threshold.dir/threshold.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/threshold.cpp -o CMakeFiles/threshold.dir/threshold.cpp.s

# Object files for target threshold
threshold_OBJECTS = \
"CMakeFiles/threshold.dir/threshold.cpp.o"

# External object files for target threshold
threshold_EXTERNAL_OBJECTS =

threshold: CMakeFiles/threshold.dir/threshold.cpp.o
threshold: CMakeFiles/threshold.dir/build.make
threshold: /usr/local/lib/libopencv_gapi.so.4.9.0
threshold: /usr/local/lib/libopencv_highgui.so.4.9.0
threshold: /usr/local/lib/libopencv_ml.so.4.9.0
threshold: /usr/local/lib/libopencv_objdetect.so.4.9.0
threshold: /usr/local/lib/libopencv_photo.so.4.9.0
threshold: /usr/local/lib/libopencv_stitching.so.4.9.0
threshold: /usr/local/lib/libopencv_video.so.4.9.0
threshold: /usr/local/lib/libopencv_videoio.so.4.9.0
threshold: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
threshold: /usr/local/lib/libopencv_dnn.so.4.9.0
threshold: /usr/local/lib/libopencv_calib3d.so.4.9.0
threshold: /usr/local/lib/libopencv_features2d.so.4.9.0
threshold: /usr/local/lib/libopencv_flann.so.4.9.0
threshold: /usr/local/lib/libopencv_imgproc.so.4.9.0
threshold: /usr/local/lib/libopencv_core.so.4.9.0
threshold: CMakeFiles/threshold.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable threshold"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/threshold.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/threshold.dir/build: threshold
.PHONY : CMakeFiles/threshold.dir/build

CMakeFiles/threshold.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/threshold.dir/cmake_clean.cmake
.PHONY : CMakeFiles/threshold.dir/clean

CMakeFiles/threshold.dir/depend:
	cd /home/taiwei/code/cvCode/project3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3/CMakeFiles/threshold.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/threshold.dir/depend

